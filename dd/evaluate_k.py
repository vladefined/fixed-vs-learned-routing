"""Variant-K specific evaluation (hard-concrete shift gates).  python -m dd.evaluate_k runs5/K25_s0 [data5]
Runs dd.evaluate5 (common metrics) first if eval5.json is missing, then writes <run>/evalk.json."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

from .evaluate import DEV, by_bins, ce_of, load_run, load_split
from .evaluate5 import answer_eval


def set_sample(model, on):
    for b in model.blocks:
        b.hc_sample = on


def main(run, data="data5"):
    if not os.path.exists(os.path.join(run, "eval5.json")):
        from . import evaluate5
        evaluate5.main(run, data)            # needs gradients (probes): outside no_grad
    with torch.no_grad():
        _k_eval(run, data)


def _k_eval(run, data):
    model, cfg = load_run(run)
    assert cfg.gate_type == "hc"
    te, va = load_split(data, "test"), load_split(data, "val")
    rows = torch.tensor(te["tok"].astype(np.int64), device=DEV)
    vrows = torch.tensor(va["tok"][:1500].astype(np.int64), device=DEV)
    L, G, d = cfg.n_layers, cfg.depth_groups, cfg.d
    gs = d // G
    res = {}

    # -------- hard threshold (deployment) vs sampled gates (training mode)
    hard = answer_eval(model, rows, te, n_rows=2500)
    hard["val_ce"] = ce_of(model, vrows)
    set_sample(model, True)
    torch.manual_seed(0)
    samp = answer_eval(model, rows, te, n_rows=2500)
    samp["val_ce"] = ce_of(model, vrows)
    set_sample(model, False)
    res["hard"], res["sampled"] = hard, samp

    # -------- shift statistics on test tokens, both modes
    n_rows = 1500
    stats = {}
    for mode in ("hard", "sampled"):
        set_sample(model, mode == "sampled")
        torch.manual_seed(1)
        shift = np.zeros((L, G))
        shift_frac = np.zeros(L)
        cont = np.zeros(L)              # fraction of strictly fractional gates (0 < m < 1)
        p_open = np.zeros(L)
        ntok = 0
        rel_moving = []
        max_stand_change = 0.0
        per_group_rate = np.zeros((L, G))
        ans_shifts, ans_m = [], []
        for i in range(0, n_rows, 128):
            idx = rows[i:i + 128]
            B = idx.shape[0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, aux = model(idx, collect=True)
            msk = idx != 0
            n = int(msk.sum())
            ntok += n
            for l in range(L):
                mm = aux["dd"][l][msk]                                 # n, G
                shift_frac[l] += float((mm > 0).float().sum())
                cont[l] += float(((mm > 0) & (mm < 1)).float().sum())
                p_open[l] += float(aux["rc"][l][msk].sum())
                per_group_rate[l] += (mm > 0).float().sum(0).cpu().numpy()
                x0 = aux["xs"][l].float().view(B, -1, G, gs)[msk]          # n, G, gs
                x1 = aux["xs"][l + 1].float().view(B, -1, G, gs)[msk]
                dx = (x1 - x0)
                st = mm == 0
                if st.any():
                    max_stand_change = max(max_stand_change, float(dx[st].abs().max()))
                mv = mm > 0
                if mv.any() and mode == "hard":
                    r = dx.pow(2).mean(-1).sqrt() / x0.pow(2).mean(-1).sqrt().clamp_min(1e-12)
                    rel_moving.append(r[mv].cpu().numpy())
            if mode == "hard":
                aq = torch.tensor(te["ansq"][i:i + B] == 1, device=DEV)
                if aq.any():
                    tot = torch.stack([aux["dd"][l] for l in range(L)]).sum(0).sum(-1)   # B,T: shifted groups
                    ans_shifts.append(tot[aq].cpu().numpy())
                    ans_m.append(te["qm"][i:i + B][te["ansq"][i:i + B] == 1])
        stats[mode] = dict(shift_frac_layer=(shift_frac / (ntok * G)).tolist(),
                           shift_frac=float(shift_frac.sum() / (ntok * G * L)),
                           fractional_gate_frac_layer=(cont / (ntok * G)).tolist(),
                           p_open_layer=(p_open / (ntok * G)).tolist(),
                           max_abs_change_of_standing_group=max_stand_change)
        if mode == "hard":
            rate = per_group_rate / ntok
            stats[mode]["always_shift_frac_layer"] = (rate > 0.99).mean(1).tolist()
            stats[mode]["never_shift_frac_layer"] = (rate < 0.01).mean(1).tolist()
            stats[mode]["group_rate"] = rate.tolist()
            rm = np.concatenate(rel_moving)
            stats[mode]["moving_rel_change_quantiles"] = {q: float(np.quantile(rm, q)) for q in
                                                          (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9)}
            stats[mode]["moving_rel_change_hist"] = np.histogram(np.clip(rm, 0, 2), bins=40, range=(0, 2))[0].tolist()
            stats[mode]["moving_small_frac"] = {str(t): float((rm < t).mean()) for t in (0.01, 0.05, 0.1, 0.2)}
            a_s, a_m = np.concatenate(ans_shifts), np.concatenate(ans_m)
            stats[mode]["answer_shifts_by_m"] = by_bins(a_s, a_m, range(7))
    set_sample(model, False)
    res["stats"] = stats
    res["mu_final"] = model.mu.tolist()
    json.dump(res, open(os.path.join(run, "evalk.json"), "w"), indent=1)
    h, s = stats["hard"], stats["sampled"]
    print(f"answer hard {hard['acc']:.4f} sampled {samp['acc']:.4f} | val CE hard {hard['val_ce']:.4f} "
          f"sampled {samp['val_ce']:.4f}")
    print("shift frac hard", np.round(h["shift_frac_layer"], 3).tolist(), "sampled",
          np.round(s["shift_frac_layer"], 3).tolist(), "fractional (sampled)",
          np.round(s["fractional_gate_frac_layer"], 3).tolist())
    print("max |dx| of standing groups (hard):", h["max_abs_change_of_standing_group"],
          "| always/never:", np.round(h["always_shift_frac_layer"], 2).tolist(),
          np.round(h["never_shift_frac_layer"], 2).tolist())
    print("moving rel change quantiles", {k: round(v, 3) for k, v in h["moving_rel_change_quantiles"].items()})
    print("shifted groups at answer by m", {k: round(v, 2) for k, v in h["answer_shifts_by_m"].items()})


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
