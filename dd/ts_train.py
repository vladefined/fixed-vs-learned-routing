"""Round 6, part 2: train one variant on TinyStories (token stream from dd.ts_data, random windows of ctx+1 tokens).
Same Model / variant registry / lm_loss / mu dual ascent as dd.train; only the data and the evaluation differ.
python -m dd.ts_train --variant QG --target 0.25 --seed 0 --name QG25_s0 [--q_delta D --q_levels K]"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")    # before CUDA is initialised
import torch  # noqa: E402

from .model import Cfg, Model, lm_loss, vq_kmeans_init, vq_update
from .train import variant_cfg


def load_stream(path, dev, chunk=1 << 26):
    """uint16 token file -> int16 tensor on the GPU (vocab < 32768), or int32 when the ids do not fit (GPT-2 BPE),
    copied in chunks to keep host memory small."""
    mm = np.memmap(path, dtype=np.uint16, mode="r")
    wide = bool(mm.max() >= 32768)
    out = torch.empty(mm.size, dtype=torch.int32 if wide else torch.int16, device=dev)
    for s in range(0, mm.size, chunk):
        a = np.asarray(mm[s:s + chunk])
        out[s:s + chunk] = torch.from_numpy(a.astype(np.int32) if wide else a.view(np.int16)).to(dev)
    return out


def windows(stream, offs, ctx):
    return stream[offs[:, None] + torch.arange(ctx + 1, device=stream.device)].long()


@torch.no_grad()
def val_loss(model, val, offs, ctx, bs=32):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, offs.numel(), bs):
        idx = windows(val, offs[i:i + bs], ctx)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, ce, *_ = lm_loss(model, idx, lam=0.0)
        k = int((idx[:, 1:] != 0).sum())
        tot += float(ce) * k
        n += k
    model.train()
    return tot / n


def init_from_checkpoint(model, path, train, ctx, bs, dev, kmeans_windows=0):
    """round 15/16: load a finished model (e.g. A_s0) into a model with extra buffers/parameters (VQ codebooks,
    attractor book) and fill those from the base model's activations on one batch, before step 0"""
    cfg = model.cfg
    ck = torch.load(path, map_location=dev, weights_only=False)
    missing, unexpected = model.load_state_dict(ck["state"], strict=False)
    assert not unexpected and all(k.startswith(("vq_", "att_")) for k in missing), (missing, unexpected)
    print(f"init from {path}: missing (new) {missing}", flush=True)
    if cfg.vq and kmeans_windows:                # round 17: k-means on the base model's directions
        dev = torch.device(dev)
        offs = torch.randint(0, train.numel() - ctx - 1, (kmeans_windows,), device=dev,
                             generator=torch.Generator(device=dev).manual_seed(998))
        nb = model.vq_E.shape[0]
        Zb = [[] for _ in range(nb)]
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            for i in range(0, kmeans_windows, 16):
                _, aux0 = model(windows(train, offs[i:i + 16], ctx), ctl={"vq_off": True})
                for bk, z in zip(aux0["vq_book"], aux0["vq_z"]):
                    Zb[bk].append(z.reshape(-1, cfg.depth_groups, cfg.d // cfg.depth_groups))
        Zb = [torch.cat(z) for z in Zb]
        calls = len(aux0["vq_book"]) // nb
        vq_kmeans_init(model, Zb, [bs * (ctx + 1) * calls] * nb)
        print(f"codebooks initialised by spherical k-means on {kmeans_windows} train windows "
              f"({Zb[0].shape[0]} vectors per book and group)", flush=True)
        return missing
    if cfg.vq or cfg.attract:
        dev = torch.device(dev)
        offs = torch.randint(0, train.numel() - ctx - 1, (bs,), device=dev,
                             generator=torch.Generator(device=dev).manual_seed(999))
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            _, aux0 = model(windows(train, offs, ctx), ctl={"attract_off": True} if cfg.attract else None)
            if cfg.vq:
                vq_update(model, aux0)
                print("codebooks initialised from the base model's activations", flush=True)
            if cfg.attract:                      # book = random directions of the base model's states (all blocks)
                G = cfg.depth_groups
                Xh = torch.cat([z.reshape(-1, G, cfg.d // G) for z in aux0["att_xh"]])
                r = torch.stack([torch.randperm(Xh.shape[0], device=dev)[:cfg.vq_k] for _ in range(G)], 1)
                model.att_C.copy_(Xh[r, torch.arange(G, device=dev)[None, :]].transpose(0, 1))
                print("attractor book initialised from the base model's state directions", flush=True)
    return missing


@torch.no_grad()
def mu_step(model, paid_layer, tgt, eta, step_max=None):
    """Dual ascent on mu: one shared mu (mean paid share vs the target) or, with cfg.qg_mu_per_block (round 11),
    mu_l += eta * (paid_l - target) for every block; mu >= 0, |change| <= step_max per step if given."""
    fl = torch.stack([f.detach().float() for f in paid_layer])
    dmu = eta * ((fl if model.cfg.qg_mu_per_block else fl.mean()) - tgt)
    if step_max is not None:
        dmu = dmu.clamp(-step_max, step_max)
    model.mu.add_(dmu).clamp_(min=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="A")
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--ffn_hidden", type=int, default=1408)
    ap.add_argument("--depth_groups", type=int, default=None, help="size ladder: QG groups G (default = n_heads)")
    ap.add_argument("--depth_lookup_recompute", type=int, default=0, help="size ladder: recompute depth lookups in backward")
    ap.add_argument("--q_delta", type=float, default=None)
    ap.add_argument("--q_levels", type=int, default=None)
    ap.add_argument("--q_delta_layers", default=None, help="grid step per state: emb,block0,...,block{L-1} (comma list)")
    ap.add_argument("--emb_scale", type=float, default=1.0, help="embedding multiplier before quantisation")
    ap.add_argument("--q_emb_init", type=float, default=None, help="embedding init std in grid steps (default 1)")
    ap.add_argument("--mask_p", type=float, default=None, help="QGrandom / QGstatic: share of open groups")
    ap.add_argument("--pos_profile", default=None, help="QGposition: json with edges, probs, p0")
    ap.add_argument("--init_from", default=None, help="round 15: start from this checkpoint (model.pt); new buffers "
                                                         "(VQ codebooks) are filled from its activations before step 0")
    ap.add_argument("--kmeans_windows", type=int, default=0, help="round 17: k-means codebook init on N train windows")
    ap.add_argument("--vq_gamma_steps", type=int, default=0, help="round 17: insert the book gradually over N steps")
    ap.add_argument("--report_insert", type=int, default=0, help="round 17: val CE right after insertion (gamma 0, 1)")
    ap.add_argument("--attract_scale", default=None, help="round 16: per-block group norm (comma list)")
    ap.add_argument("--attract_beta", default="1,30", help="round 16: beta from,to (linear over the steps)")
    ap.add_argument("--static_k", default=None, help="round 11 QGstaticunequal: open groups per block, comma list")
    ap.add_argument("--randk_probs", default=None, help="round 11 QGrandk: pmf of k = 0..G, comma list")
    ap.add_argument("--target", type=float, default=0.25)
    ap.add_argument("--target_frac_steps", type=float, default=0.0625, help="cost target 1.0 -> target over this share")
    ap.add_argument("--mu_eta", type=float, default=0.05)
    ap.add_argument("--mu_step_max", type=float, default=None, help="round 9 (QG25-mulimit): |change of mu| per step <= this")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--ckpt_every", type=int, default=5000)
    ap.add_argument("--val_windows", type=int, default=256)
    ap.add_argument("--data", default="data_ts")
    ap.add_argument("--out", default="runs_ts")
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--stop_at", type=int, default=None)
    ap.add_argument("--sampler", default="random", choices=["random", "perm"],
                    help="random: window offsets drawn with replacement (TinyStories); perm: non-overlapping windows, "
                         "each used once (needs steps * bs windows of ctx + 1 tokens)")
    a = ap.parse_args()

    name = a.name or f"{a.variant}_s{a.seed}"
    out = os.path.join(a.out, name)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    vc = variant_cfg(a.variant, a.d, a.depth_groups or a.n_heads)
    if a.q_delta is not None:
        vc["q_delta"] = a.q_delta
    if a.q_levels is not None:
        vc["q_levels"] = a.q_levels
    if a.q_delta_layers:
        vc["q_delta_layers"] = tuple(float(v) for v in a.q_delta_layers.split(","))
        assert len(vc["q_delta_layers"]) == a.n_layers + 1
    if a.emb_scale != 1.0:
        vc["emb_scale"] = a.emb_scale
    if a.q_emb_init is not None:
        vc["q_emb_init"] = a.q_emb_init
    if a.mask_p is not None:
        vc["qg_mask_p"] = a.mask_p
    assert (a.variant == "QGstaticunequal") == bool(a.static_k) and (a.variant == "QGrandk") == bool(a.randk_probs)
    if a.attract_scale:
        vc["attract_scale"] = tuple(float(v) for v in a.attract_scale.split(","))
    if a.vq_gamma_steps:
        vc["vq_mix"] = True
    if a.depth_lookup_recompute:
        vc["depth_lookup_recompute"] = True
    if a.static_k:
        vc["qg_static_k"] = tuple(int(v) for v in a.static_k.split(","))
    if a.randk_probs:
        pk = [float(v) for v in a.randk_probs.split(",")]
        vc["qg_randk_probs"] = tuple(v / sum(pk) for v in pk)
    if a.pos_profile:
        pp = json.load(open(a.pos_profile))
        vc.update(qg_pos_edges=tuple(pp["edges"]), qg_pos_probs=tuple(pp["probs"]), qg_pos_p0=pp["p0"])
    cfg = Cfg(vocab=meta["vocab"], d=a.d, n_layers=a.n_layers, n_heads=a.n_heads, ffn_hidden=a.ffn_hidden,
              max_len=a.ctx + 1, **vc)
    model = Model(cfg).to(dev)
    if cfg.vq:                                   # round 14: exact fp32 distances for the nearest codebook vector
        torch.backends.cuda.matmul.allow_tf32 = False
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {n_params / 1e6:.2f}M params, cfg {json.dumps(vc)}", flush=True)

    train = load_stream(os.path.join(a.data, "train.bin"), dev)
    val = load_stream(os.path.join(a.data, "val.bin"), dev)
    gv = torch.Generator(device=dev).manual_seed(1234)                 # same validation windows for every run
    voffs = torch.randint(0, val.numel() - a.ctx - 1, (a.val_windows,), device=dev, generator=gv)
    g = torch.Generator(device=dev).manual_seed(a.seed)
    if a.sampler == "perm":
        n_win = (train.numel() - 1) // (a.ctx + 1)
        assert n_win >= a.steps * a.bs, f"{n_win} windows < {a.steps * a.bs} needed: the data would repeat"
        perm = torch.randperm(n_win, generator=torch.Generator().manual_seed(a.seed))[:a.steps * a.bs].to(dev) * (a.ctx + 1)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() == 2 and "mat_w" not in n else no_decay).append(p)     # mat_w: RMSNorm_k weights
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=True)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = min(1.0, (s - a.warmup) / max(1, a.steps - a.warmup))
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    targeted = cfg.q_cost != "none" and cfg.quant and not cfg.qg_mask     # round 7 controls: mu stays 0
    ramp = max(1, int(a.target_frac_steps * a.steps))

    def target_at(s):
        return a.target + (1.0 - a.target) * max(0.0, 1.0 - s / ramp)

    if a.init_from:                              # round 15/16: fine-tuning from a finished model
        init_from_checkpoint(model, a.init_from, train, a.ctx, a.bs, dev, a.kmeans_windows)
    insert_ce = None
    if a.report_insert and cfg.vq:               # round 17: what the book alone costs, before any fine-tuning
        insert_ce = {}
        for gm in (0.0, 1.0):
            model.vq_gamma.fill_(gm)
            insert_ce[f"gamma_{gm:g}"] = val_loss(model, val, voffs, a.ctx)
        model.vq_gamma.fill_(1.0)
        print(f"val CE right after insertion: {insert_ce}", flush=True)
    att_b0, att_b1 = (float(v) for v in a.attract_beta.split(","))
    step_fn = torch.compile(lambda idx: lm_loss(model, idx)) if a.compile else (lambda idx: lm_loss(model, idx))
    log, hist, t0 = [], [], time.time()
    tgt = float("nan")
    for step in range(a.steps):
        if a.sampler == "perm":
            offs = perm[step * a.bs:(step + 1) * a.bs]
        else:
            offs = torch.randint(0, train.numel() - a.ctx - 1, (a.bs,), device=dev, generator=g)
        idx = windows(train, offs, a.ctx)
        if cfg.vq_mix:                           # round 17: gamma 0 -> 1 over vq_gamma_steps
            model.vq_gamma.fill_(min(1.0, step / max(1, a.vq_gamma_steps)))
        if cfg.attract:                          # round 16: beta grows linearly
            model.att_beta.fill_(att_b0 + (att_b1 - att_b0) * step / max(1, a.steps - 1))
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, ce, cost, per_layer, aux = step_fn(idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        vq_stats = vq_update(model, aux) if cfg.vq else None
        if targeted:
            tgt = target_at(step)
            mu_step(model, aux["q_l0_layer"], tgt, a.mu_eta, a.mu_step_max)
        if step % 100 == 0 or step == a.steps - 1:
            rec = dict(step=step, loss=float(loss.detach()), ce=float(ce.detach()), cost=float(cost.detach()),
                       gn=float(gn), lr=lr_at(step), t=time.time() - t0)
            extra = ""
            if cfg.attract:
                rec["att_beta"] = float(model.att_beta)
                rec["att_wmax"] = [float(w.mean()) for w in aux["att_wmax"]]
                extra = f" beta={rec['att_beta']:.1f} wmax=" + ",".join(f"{v:.2f}" for v in rec["att_wmax"])
            if cfg.vq_mix:
                rec["vq_gamma"] = float(model.vq_gamma)
            if cfg.vq:                           # used vectors this step / restarted, per book
                rec["vq_used"], rec["vq_restart"] = [u for u, _ in vq_stats], [r for _, r in vq_stats]
                extra = f" commit={rec['cost']:.4f} used=" + ",".join(map(str, rec["vq_used"])) + \
                        " restart=" + ",".join(map(str, rec["vq_restart"]))
            if cfg.quant:
                rec["q_l1_layer"] = [float(c.detach()) for c in aux["q_l1_layer"]]
                rec["q_l0_layer"] = [float(c.detach()) for c in aux["q_l0_layer"]]
                if cfg.qg:
                    extra = " paid=" + ",".join(f"{c:.2f}" for c in rec["q_l0_layer"]) + \
                            " real=" + ",".join(f"{c:.2f}" for c in rec["q_l1_layer"])
                else:
                    extra = " moved=" + ",".join(f"{c:.2f}" for c in rec["q_l0_layer"])
                if targeted:
                    rec["mu"], rec["target"] = model.mu.tolist(), tgt
                    extra += (" mu=" + ",".join(f"{v:.2f}" for v in rec["mu"]) if cfg.qg_mu_per_block
                              else f" mu={float(model.mu[0]):.3f}") + f" tgt={tgt:.3f}"
            hist.append(rec)
            if step % 500 == 0:
                rec["mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
                print(f"{step:6d} loss {rec['loss']:.4f} ce {rec['ce']:.4f} gn {rec['gn']:.2f} lr {rec['lr']:.2e} "
                      f"{rec['t']:.0f}s mem {rec['mem_gb']:.1f}G{extra}", flush=True)
        if (step + 1) % a.eval_every == 0 or step == a.steps - 1:
            vce = val_loss(model, val, voffs, a.ctx)
            log.append(dict(step=step + 1, val_ce=vce, t=time.time() - t0))
            print(f"  [val] step {step + 1} ce {vce:.4f} ppl {math.exp(vce):.3f}", flush=True)
        if a.ckpt_every and (step + 1) % a.ckpt_every == 0:
            torch.save({"cfg": cfg.to_dict(), "state": model.state_dict(), "step": step + 1},
                       os.path.join(out, f"ckpt_{step + 1:06d}.pt"))
        if a.stop_at and step + 1 >= a.stop_at:
            break
    train_time = time.time() - t0
    vce = log[-1]["val_ce"] if log and log[-1]["step"] == step + 1 else val_loss(model, val, voffs, a.ctx)
    res = dict(name=name, variant=a.variant, seed=a.seed, steps=step + 1, tokens=(step + 1) * a.bs * a.ctx,
               train_time=train_time, val_ce=vce, val_ppl=math.exp(vce), n_params=n_params, cfg=cfg.to_dict(),
               insert_val_ce=insert_ce,
               args=vars(a))
    torch.save({"cfg": cfg.to_dict(), "state": model.state_dict()}, os.path.join(out, "model.pt"))
    json.dump(dict(result=res, val_log=log, train_log=hist), open(os.path.join(out, "log.json"), "w"))
    print(json.dumps({k: v for k, v in res.items() if k not in ("cfg", "args")}), flush=True)


if __name__ == "__main__":
    main()
