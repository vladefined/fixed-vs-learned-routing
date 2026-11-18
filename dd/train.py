"""Train one variant.   python -m dd.train --variant B --lam 0.03 --seed 0 --steps 20000"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

from .model import Cfg, Model, lm_loss, update_scale_ema

FUNNEL_NARROW = tuple(tuple(range(l, 8)) for l in range(8))           # block l writes groups l..7 (36 writes)
FUNNEL_WIDE = tuple(tuple(range(0, l + 1)) for l in range(8))         # block l writes groups 0..l (36 writes)


def static36(G=8, per_block=(5, 5, 5, 5, 4, 4, 4, 4)):
    """round 18: the same 36 group writes spread evenly -- groups taken round-robin, every group written 4-5 times"""
    out, p = [], 0
    for n in per_block:
        out.append(tuple(sorted((p + j) % G for j in range(n))))
        p = (p + n) % G
    return tuple(out)


def variant_cfg(name, d, H):
    """Variant registry.  Normalised-state scheme is the main one; *_legacy = original formulation."""
    N = dict(budget=True, norm_state=True)
    table = {
        "A": dict(budget=False),
        "A_norm": dict(budget=False, norm_state=True),
        "B": dict(**N, depth_groups=d, update="min"),
        "B_dd": dict(**N, depth_groups=d, update="dd"),
        "B_pull": dict(**N, depth_groups=d, update="pull"),
        "C": dict(**N, depth_groups=d, update="min", update_src="ffn"),
        "E": dict(**N, depth_groups=H, norm_groups=H, update="min"),
        "F": dict(**N, depth_groups=1, update="min"),
        "G": dict(**N, depth_groups=H, norm_groups=H, update="min", attn_depth_bias=True),
        "B_legacy": dict(budget=True, depth_groups=d, update="full", logit_clamp=0.0),
        # round 3 (groups = heads)
        "A_groupnorm": dict(budget=False, norm_state=True, norm_groups=H),
        "H": dict(**N, depth_groups=H, norm_groups=H, update="min", budget_global=True),
        "I": dict(budget=True, depth_groups=H, update="min", budget_global=True, rel_cost=True),
        "K": dict(budget=True, gate_type="hc", depth_groups=H, norm_state=True, norm_groups=H, update="min",
                  attn_depth_bias=True),
        "KL": dict(budget=True, gate_type="hc", depth_groups=H, norm_state=True, norm_groups=H, update="min",
                   attn_depth_bias=True, per_block_mu=True),
        "Kfree": dict(budget=True, gate_type="hc", depth_groups=H, norm_state=False, update="min",
                      attn_depth_bias=True, hc_tanh=False),
        "Knorm": dict(budget=True, gate_type="hc", depth_groups=H, norm_state=True, norm_groups=H, update="min",
                      attn_depth_bias=True, hc_tanh=False),
        "J": dict(budget=True, depth_groups=H, update="min", budget_global=True, rel_cost=True,
                  state_noise=0.02),
        # round 4
        "Q0": dict(quant=True, q_cost="none"),
        "QL1": dict(quant=True, q_cost="l1"),
        "QL0": dict(quant=True, q_cost="l0"),
        "Ip": dict(budget=True, depth_groups=H, update="min", budget_global=True, rel_cost=True,
                   rel_scale="median"),
        "KSTE": dict(budget=True, gate_type="ste", depth_groups=H, norm_state=True, norm_groups=H, update="min",
                     attn_depth_bias=True),
        "A_groupnorm_free": dict(budget=False, norm_state=True, norm_groups=H, free_per_group=2),
        # round 5: quantised state + per-group decision before the FFN
        "QG": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H),
        "QGsplit": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H, qg_split=True),
        "QGsplitFFN": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H, qg_split=True, qg_split_parts="ffn"),
        "QGsplitHead": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H, qg_split=True, qg_split_parts="head"),
        "QGnogate": dict(quant=True, q_cost="l0", qg="nogate", depth_groups=H),
        "E_free": dict(**N, depth_groups=H, norm_groups=H, update="min", free_per_group=2),
        "G_free": dict(**N, depth_groups=H, norm_groups=H, update="min", attn_depth_bias=True, free_per_group=2),
        # round 7: QG with a mask that is not learned (gate controls) and the QG-split cut without gate and grid
        "QGrandom": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H, qg_mask="random"),
        "QGstatic": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H, qg_mask="static"),
        "QGposition": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H, qg_mask="position"),
        "QGtoken": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H, qg_gate_in="token"),
        # round 8: start from the static mask / one static group always open + adaptive rest
        "QGstaticinit": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H, qg_static_init=True),
        "QGhybrid": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H, qg_force_open=1),
        "QGctrl": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H, qg_ctrl=64),
        # round 9: round only the update
        "Q0upd": dict(quant=True, q_cost="none", q_upd_round=True),
        "QGupd": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H, q_upd_round=True),
        "QGstaticupd": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H, qg_mask="static", q_upd_round=True),
        # round 9: H-Net-style gate training (write fraction s(p), no straight-through on m in the write path)
        "QGsmoothhnet": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H, qg_smooth="hnet"),
        "QGsmoothdimmer": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H, qg_smooth="dimmer"),
        "QGsmoothhnetfix": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H, qg_smooth="hnet", qg_smooth_closed_ste=True),
        "QGsmoothdimmerfix": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H, qg_smooth="dimmer",
                                  qg_smooth_closed_ste=True),
        # round 10: no grid (open groups write x + a + u unrounded, closed groups still not written); ordered gate
        "QGnogrid": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H, q_nogrid=True),
        "QGstaticnogrid": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H, qg_mask="static", q_nogrid=True),
        "QGordered": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H, qg_ordered=True),
        # round 11: ordered gate with a target and mu per block; no gate: random k (prefix of the queue, --randk_probs)
        # or a fixed but unequal k per block (--static_k)
        "QGorderedperblock": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H, qg_ordered=True,
                                  qg_mu_per_block=True),
        # round 13: ordered-perblock with k in [1, 3]; the same without the grid
        "QGorderedperblockk13": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H, qg_ordered=True,
                                     qg_mu_per_block=True, qg_k_range=(1, 3)),
        "QGorderedperblockk13nogrid": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H, qg_ordered=True,
                                           qg_mu_per_block=True, qg_k_range=(1, 3), q_nogrid=True),
        "QGorderedperblocknogrid": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H, qg_ordered=True,
                                        qg_mu_per_block=True, q_nogrid=True),
        # round 14: VQ state on top of A (no grid, no gate, no cost)
        "Avqshared": dict(budget=False, depth_groups=H, vq="shared", vq_k=1024),
        "Avqdelta": dict(budget=False, depth_groups=H, vq="delta", vq_k=256),
        # round 18 (the user's r17): funnel -- fixed per-block written groups, no gates / cost / grid
        "Afunnelnarrow": dict(budget=False, depth_groups=8, write_groups=FUNNEL_NARROW),
        "Afunnelwide": dict(budget=False, depth_groups=8, write_groups=FUNNEL_WIDE),
        "Astatic36": dict(budget=False, depth_groups=8, write_groups=static36()),
        "Afunnelnarrowmat": dict(budget=False, depth_groups=8, write_groups=FUNNEL_NARROW, mat_weight=0.2),
        # round 20: A with early-exit pressure by depth (main head after a random block)
        "Amat": dict(budget=False, exit_weight=0.2),
        "Amat01": dict(budget=False, exit_weight=0.1),
        # round 20: strict prefix head (the main head cut to the first k groups), lambda 0.2 / 0.1
        "Afunnelnarrowmatstrict": dict(budget=False, depth_groups=8, write_groups=FUNNEL_NARROW, mat_weight=0.2,
                                       mat_strict=True),
        "Afunnelnarrowmatstrict01": dict(budget=False, depth_groups=8, write_groups=FUNNEL_NARROW, mat_weight=0.1,
                                         mat_strict=True),
        # round 16: books-shared-attractor (needs --attract_scale; fine-tuned from A)
        "Aattract": dict(budget=False, depth_groups=H, attract=True, vq_k=1024),
        # round 15: normalised (cosine) VQ, separate length per group, beta 0.05
        "Avqsharednorm": dict(budget=False, depth_groups=H, vq="shared", vq_k=1024, vq_norm=True, vq_beta=0.05),
        "Avqdeltanorm": dict(budget=False, depth_groups=H, vq="delta", vq_k=256, vq_norm=True, vq_beta=0.05),
        "QGrandk": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H, qg_mask="randk"),
        "QGstaticunequal": dict(quant=True, q_cost="l0", qg="gate", depth_groups=H, qg_mask="static"),
        "Asplit": dict(budget=False, depth_groups=H, qg_split=True),
        "Q0split": dict(quant=True, q_cost="none", depth_groups=H, qg_split=True),
    }
    return table[name]




def load_rows(path, device):
    return torch.from_numpy(np.load(path).astype(np.int32)).to(device)


@torch.no_grad()
def evaluate(model, rows, bs=256, max_rows=None):
    model.eval()
    n = rows.shape[0] if max_rows is None else min(max_rows, rows.shape[0])
    tot_ce, tot_n, cost_sum = 0.0, 0, None
    for i in range(0, n, bs):
        idx = rows[i:i + bs].long()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, ce, cost, per_layer, _ = lm_loss(model, idx, lam=0.0)
        k = int((idx[:, 1:] != 0).sum())
        tot_ce += float(ce) * k
        tot_n += k
        pl = torch.stack(per_layer) if per_layer else torch.zeros(1, device=idx.device)
        cost_sum = pl * k if cost_sum is None else cost_sum + pl * k
    model.train()
    return tot_ce / tot_n, (cost_sum / tot_n).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="A")
    ap.add_argument("--name", default=None)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--ffn_hidden", type=int, default=344)
    ap.add_argument("--gate_bias_init", type=float, default=0.0)
    ap.add_argument("--min_step", type=float, default=0.0)
    ap.add_argument("--logit_clamp", type=float, default=None)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--data", default="data5")
    ap.add_argument("--noise", type=float, default=None)
    ap.add_argument("--target", type=float, default=0.25, help="K: final target share of shifts")
    ap.add_argument("--target_frac_steps", type=float, default=0.2, help="K: share of steps to anneal 1.0->target")
    ap.add_argument("--mu_eta", type=float, default=0.05, help="K: dual ascent step for mu")
    ap.add_argument("--sched", default="ramp", help="ramp: K schedule (1.0 -> target over target_frac_steps); "
                    "late: no constraint until hold_frac, then linear to target at end_frac")
    ap.add_argument("--hold_frac", type=float, default=0.3)
    ap.add_argument("--end_frac", type=float, default=0.6)
    ap.add_argument("--q_delta", type=float, default=None)
    ap.add_argument("--ckpt_every", type=int, default=0)
    ap.add_argument("--stop_at", type=int, default=None, help="stop after this many steps (LR schedule of --steps)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--max_minutes", type=float, default=None)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--resume", default=None,
                    help="continue from a checkpoint with 'step' (weights + mu; fresh AdamW, data stream fast-forwarded)")
    ap.add_argument("--sched_steps", type=int, default=None,
                    help="length the LR cosine and the cost-target schedule are computed for (default --steps); "
                         "past it the LR stays at its floor (0.1 lr)")
    ap.add_argument("--rewarm", type=int, default=500, help="with --resume: linear LR re-warmup over this many steps")
    a = ap.parse_args()
    sched_steps = a.sched_steps or a.steps

    name = a.name or f"{a.variant}_lam{a.lam:g}_s{a.seed}"
    out = os.path.join(a.out, name)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")

    meta = json.load(open(os.path.join(a.data, "meta.json")))
    vc = variant_cfg(a.variant, a.d, a.n_heads)
    if a.logit_clamp is not None:
        vc["logit_clamp"] = a.logit_clamp
    if a.noise is not None:
        vc["state_noise"] = a.noise
    if a.q_delta is not None:
        vc["q_delta"] = a.q_delta
    cfg = Cfg(vocab=len(meta["vocab"]), d=a.d, n_layers=a.n_layers, n_heads=a.n_heads,
              ffn_hidden=a.ffn_hidden, lam=a.lam, gate_bias_init=a.gate_bias_init,
              min_step=a.min_step, **vc)
    model = Model(cfg).to(dev)
    start = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev)
        model.load_state_dict(ck["state"])
        start = int(ck["step"])
        print(f"resumed from {a.resume} at step {start}, mu={model.mu.tolist()}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {n_params / 1e6:.2f}M params", flush=True)

    train = load_rows(os.path.join(a.data, "train_tok.npy"), dev)
    val = torch.from_numpy(np.load(os.path.join(a.data, "val.npz"))["tok"].astype(np.int32)).to(dev)
    from .evaluate import quick_struct
    te = {k: v for k, v in np.load(os.path.join(a.data, "test.npz")).items()}

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() == 2 else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=True)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = min(1.0, (s - a.warmup) / max(1, sched_steps - a.warmup))
        lr = a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))
        if start and s - start < a.rewarm:
            lr *= (s - start + 1) / a.rewarm
        return lr

    step_fn = torch.compile(lambda idx: lm_loss(model, idx)) if a.compile else (lambda idx: lm_loss(model, idx))
    g = torch.Generator(device=dev).manual_seed(a.seed)
    perm, ptr = torch.randperm(train.shape[0], device=dev, generator=g), 0
    log, t0 = [], time.time()
    hist = []
    targeted = cfg.gate_type in ("hc", "ste") or cfg.q_cost != "none"
    cost_ema, tgt_start, tgt = None, None, float("nan")

    def target_at(step):
        """Target for the dual-ascent constraint; None = no constraint (mu held at 0)."""
        nonlocal tgt_start
        if a.sched == "ramp":
            ramp = max(1, int(a.target_frac_steps * sched_steps))
            return a.target + (1.0 - a.target) * max(0.0, 1.0 - step / ramp)
        h, e = int(a.hold_frac * sched_steps), int(a.end_frac * sched_steps)
        if step < h:
            return None
        if tgt_start is None:      # K-STE: share starts at 1.0 (spec); Q: the cost reached when unconstrained
            tgt_start = 1.0 if cfg.gate_type in ("hc", "ste") else max(cost_ema, a.target)
        f = min(1.0, (step - h) / max(1, e - h))
        return tgt_start + (a.target - tgt_start) * f
    for _ in range(start):                       # fast-forward the data stream of a resumed run
        if ptr + a.bs > perm.numel():
            perm, ptr = torch.randperm(train.shape[0], device=dev, generator=g), 0
        ptr += a.bs
    for step in range(start, a.steps):
        if ptr + a.bs > perm.numel():
            perm, ptr = torch.randperm(train.shape[0], device=dev, generator=g), 0
        idx = train[perm[ptr:ptr + a.bs]].long()
        ptr += a.bs
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, ce, cost, per_layer, aux = step_fn(idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if cfg.rel_cost or cfg.state_noise > 0:
            update_scale_ema(model, aux)
        if targeted:
            with torch.no_grad():
                if cfg.quant:
                    fl = torch.stack([f.detach() for f in aux["q_l1_layer" if cfg.q_cost == "l1" else "q_l0_layer"]])
                else:
                    fl = torch.stack([f.detach() for f in aux["frac_layer"]])
                cm = float(fl.mean())
                cost_ema = cm if cost_ema is None else 0.99 * cost_ema + 0.01 * cm
                tgt_v = target_at(step)
                tgt = float("nan") if tgt_v is None else tgt_v
                if tgt_v is None:
                    model.mu.zero_()
                elif cfg.per_block_mu:
                    model.mu.add_(a.mu_eta * (fl - tgt_v)).clamp_(min=0.0)
                else:
                    model.mu.add_(a.mu_eta * (fl.mean() - tgt_v)).clamp_(min=0.0)
        if step % 100 == 0 or step == a.steps - 1:
            rec = dict(step=step, loss=float(loss.detach()), ce=float(ce.detach()), cost=float(cost.detach()),
                       gn=float(gn), lr=lr_at(step), t=time.time() - t0)
            if cfg.quant:
                rec["q_l1_layer"] = [float(c.detach()) for c in aux["q_l1_layer"]]
                rec["q_l0_layer"] = [float(c.detach()) for c in aux["q_l0_layer"]]
                if targeted:
                    rec["mu"] = model.mu.tolist()
                    rec["target"] = tgt
            if cfg.budget:
                with torch.no_grad():
                    m = (idx != 0).unsqueeze(-1)
                    rec["dd_layer"] = [float(c) for c in per_layer]
                    rec["frac0_layer"] = [float(((dd < 0.01) & m).sum() / (m.sum() * dd.shape[-1]))
                                          for dd in aux["dd"]]
                    if "rc_layer" in aux:
                        rec["rc_layer"] = [float(c.detach()) for c in aux["rc_layer"]]
                    if cfg.gate_type in ("hc", "ste"):
                        rec["mu"] = model.mu.tolist()
                        rec["p_open_layer"] = [float(f.detach()) for f in aux["frac_layer"]]
                        rec["target"] = tgt
            hist.append(rec)
            if step % 500 == 0:
                extra = ""
                if cfg.budget:
                    extra = " dd=" + ",".join(f"{c:.2f}" for c in rec["dd_layer"]) + \
                            " z=" + ",".join(f"{c:.2f}" for c in rec["frac0_layer"])
                if cfg.qg:
                    extra = " paid=" + ",".join(f"{c:.3f}" for c in rec["q_l0_layer"]) + \
                            " real=" + ",".join(f"{c:.3f}" for c in rec["q_l1_layer"])
                elif cfg.quant:
                    extra = " n=" + ",".join(f"{c:.3f}" for c in rec["q_l1_layer"]) + \
                            " moved=" + ",".join(f"{c:.3f}" for c in rec["q_l0_layer"])
                if targeted:
                    extra += f" mu={float(model.mu.mean()):.3f} tgt={tgt:.3f}"
                print(f"{step:6d} loss {rec['loss']:.4f} ce {rec['ce']:.4f} cost {rec['cost']:.3f} "
                      f"gn {rec['gn']:.2f} {rec['t']:.0f}s{extra}", flush=True)
        if (step + 1) % a.eval_every == 0 or step == a.steps - 1:
            vce, vcost = evaluate(model, val, max_rows=2000)
            qs = quick_struct(model, te)
            if cfg.norm_state and cfg.budget:
                with torch.no_grad():
                    for b in model.blocks:
                        b.track = True
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        model(train[perm[:64]].long())
                    qs["pre_rms_min"] = min(b.pre_rms_min for b in model.blocks)
                    for b in model.blocks:
                        b.track = False
            log.append(dict(step=step + 1, val_ce=vce, val_cost_layer=vcost, t=time.time() - t0, **qs))
            print(f"  [val] step {step + 1} ce {vce:.4f} ppl {math.exp(vce):.3f} "
                  f"cost {sum(vcost):.3f} " + " ".join(f"{k} {v:.3f}" for k, v in qs.items()), flush=True)
        if a.ckpt_every and (step + 1) % a.ckpt_every == 0:
            torch.save({"cfg": cfg.to_dict(), "state": model.state_dict(), "step": step + 1},
                       os.path.join(out, f"ckpt_{step + 1:06d}.pt"))
        if a.max_minutes and time.time() - t0 > a.max_minutes * 60:
            print("time limit reached", flush=True)
            break
        if a.stop_at and step + 1 >= a.stop_at:
            print("stop_at reached", flush=True)
            break
    train_time = time.time() - t0
    vce, vcost = evaluate(model, val)
    res = dict(name=name, variant=a.variant, lam=a.lam, seed=a.seed, steps=step + 1,
               train_time=train_time, val_ce=vce, val_ppl=math.exp(vce), val_cost_layer=vcost,
               n_params=n_params, cfg=cfg.to_dict(), args=vars(a))
    torch.save({"cfg": cfg.to_dict(), "state": model.state_dict()}, os.path.join(out, "model.pt"))
    json.dump(dict(result=res, val_log=log, train_log=hist), open(os.path.join(out, "log.json"), "w"))
    print(json.dumps({k: v for k, v in res.items() if k not in ("cfg", "args")}), flush=True)


if __name__ == "__main__":
    main()
