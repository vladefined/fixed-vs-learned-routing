"""Round 6, part 2: grid pilots (2000 steps of the 20k schedule) -> report/r6_grid_pilots.md / .json.
For every pilot: val CE at 500/1000/1500/2000 (from its training log; killed pilots have only part of it), the
loss and gradient norm at step 0, and -- when the model was saved -- per state: RMS, share of values at the grid
edges (|level| = K), share of exact zeros, number of levels holding 99% of the values.
python -m dd.ts_grid_check [pilot_ts] [report]"""
from __future__ import annotations

import json
import os
import re
import sys

import torch

from .ts_eval import batches, load_run, val_starts
from .ts_train import load_stream

# name -> (log file, description); order = order of the table
PILOTS = [
    ("A_s0", "logs/r6ts_A_s0.txt", "A (без сетки)"),
    ("Q0_d103", "logs/pts_Q0_d103.txt", "Q0, общий Δ = 1.03, K = 15 (правило раунда 4)"),
    ("Q0_d0355", "logs/pts_Q0_d0355.txt", "Q0, общий Δ = 0.355, K = 44"),
    ("QG25_d0355", "logs/pts_QG25_d0355.txt", "QG25, общий Δ = 0.355, K = 44"),
    ("Q0_d011", "logs/pts_Q0_d011.txt", "Q0, общий Δ = 0.11, K = 141"),
    ("QG25_d011", "logs/pts_QG25_d011.txt", "QG25, общий Δ = 0.11, K = 141"),
    ("Q0_d005", "logs/pts_Q0_d005.txt", "Q0, общий Δ = 0.05, K = 309"),
    ("QG25_d005", "logs/pts_QG25_d005.txt", "QG25, общий Δ = 0.05, K = 309"),
    ("Q0_prof", "logs/pts_Q0_prof.txt", "Q0, профиль Δ_l = RMS_A/3.2, K = 8"),
    ("QG25_prof", "logs/pts_QG25_prof.txt", "QG25, профиль, K = 8"),
    ("Q0_prof_sqrtd", "logs/pts_Q0_prof_sqrtd.txt", "Q0, профиль, эмбеддинг ×√d, K = 8"),
    ("Q0_prof_sqrtd_K26", "logs/pts_Q0_prof_sqrtd_K26.txt", "Q0, профиль, эмбеддинг ×√d, K = 26"),
    ("QG25_prof_sqrtd_K26", "logs/pts_QG25_prof_sqrtd_K26.txt", "QG25, профиль, эмбеддинг ×√d, K = 26"),
    ("Q0_prof_sqrtd_K40", "logs/pts_Q0_prof_sqrtd_K40.txt", "Q0, профиль, эмбеддинг ×√d, K = 40"),
]


def parse_log(path):
    out = dict(val={})
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"\s+\[val\] step (\d+) ce ([\d.]+)", line)
        if m and int(m.group(1)) <= 2000:
            out["val"][int(m.group(1))] = float(m.group(2))
        m = re.match(r"\s+0 loss ([\d.e+inf]+) ce [\d.e+inf]+ gn ([\d.e+inf]+)", line)
        if m:
            out["loss0"], out["gn0"] = float(m.group(1)), float(m.group(2))
    return out


@torch.no_grad()
def edges(run, val, starts):
    m, cfg = load_run(run)
    acc = None
    for idx in batches(val, starts):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, aux = m(idx, collect=True)
        xs = [x.float().flatten() for x in aux["xs"]]
        acc = [[x] for x in xs] if acc is None else [a + [x] for a, x in zip(acc, xs)]
    K = cfg.q_levels
    res = []
    for i, xl in enumerate(acc):
        x = torch.cat(xl)
        D = cfg.qd(i)
        lv = torch.round(x / D).clamp(-K, K)
        h = torch.bincount((lv + K).long(), minlength=2 * K + 1).float()
        p = (h / h.sum()).sort(descending=True).values
        res.append(dict(delta=D, range=D * K, rms=float(x.pow(2).mean().sqrt()), edge=float((lv.abs() == K).float().mean()),
                        zero=float((lv == 0).float().mean()), levels99=int((p.cumsum(0) < 0.99).sum()) + 1))
    return dict(K=K, emb_scale=cfg.emb_scale, states=res)


def main(base="pilot_ts", out="report"):
    val = load_stream(os.path.join("data_ts", "val.bin"), "cuda")
    starts = val_starts(val)[:64]
    R = []
    for name, log, desc in PILOTS:
        r = dict(name=name, desc=desc, **parse_log(log))
        run = os.path.join(base, name)
        if os.path.exists(os.path.join(run, "model.pt")):
            r["grid"] = edges(run, val, starts)
        R.append(r)
    json.dump(R, open(os.path.join(out, "r6_grid_pilots.json"), "w"), indent=1)
    A = R[0]["val"]
    L = ["### Пробы сетки (2000 шагов того же расписания на 20k)\n",
         "| проба | потеря / ‖g‖ на шаге 0 | val CE 500 / 1000 / 1500 / 2000 | ΔCE к A на 2000 | max доля на краях (слой) | "
         "состояния: доля на краях, % (emb, b0 … b7) |", "|---|---|---|---|---|---|"]
    for r in R:
        v = r["val"]
        vs = " / ".join(f"{v[s]:.3f}" if s in v else "—" for s in (500, 1000, 1500, 2000))
        d = f"{v[2000] - A[2000]:+.3f}" if 2000 in v and r["name"] != "A_s0" else "—"
        if "grid" in r:
            e = [s["edge"] for s in r["grid"]["states"]]
            i = max(range(len(e)), key=lambda j: e[j])
            mx = f"{100 * e[i]:.2f}% ({'emb' if i == 0 else f'b{i - 1}'})"
            es = " ".join(f"{100 * x:.2f}" for x in e)
        else:
            mx = es = "модель не сохранена" if r["name"] != "A_s0" else "—"
        g0 = f"{r.get('loss0', float('nan')):.1f} / {r.get('gn0', float('nan')):.3g}"
        L.append(f"| {r['desc']} | {g0} | {vs} | {d} | {mx} | {es} |")
    L.append("")
    open(os.path.join(out, "r6_grid_pilots.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:])
