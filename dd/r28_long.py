"""Round 28, stage 3 (home): does the gain of the mixture over dense-cont grow together with the specialisation of its
experts, and does either reach a plateau? perlayer (а) and dense-cont continued from 10k to 40k steps (saves 20k / 30k
/ 40k); the same for hash.

Runs (runs_home/r28s3/<model>; smoke: runs_home/r28s3t):
  dense  dense-cont again from A-mix, 0 -> 10k (the r21 recipe in the r28 loop; the r21 run kept only its 10k model),
         saves 2k / 5k / 10k, then 10k -> 40k
  pl     а = runs_home/r28s1/r28_base (its saves 2k / 5k / 10k), 10k -> 40k
  hash   hash routing (r21 MoE-hash tables in the R28 model), 0 -> 10k with saves, then 10k -> 40k
First phase (0 -> 10k): exactly stage 1 (warmup 200, cosine to 0.1 x peak at 10k, data_mix windows after the 20k
steps of A-mix, permutation data_seed). Continuation (10k -> 40k), the same for every model: fresh AdamW (stage 1 kept
no optimizer state), 200 warmup steps to the lr at the end of stage 1 (0.1 x peak; blocks 1..N x 0.1 as there), then
constant; windows of data_mix_ext (documents after those of data_mix: nothing of data_mix is repeated), its own
permutation (data_seed), the same windows in the same order for every model.
At every save (2k, 5k, 10k, 20k, 30k, 40k): full val CE per window (paired differences); MoE models: specialisation per
block — foreign / zero and dense-cont / zero with dense-cont AT THE SAME STEP —, the share removed by the refit of the
down, the cosine of the expert updates since step 0 (dd.r28_measure.measure_all: 1024 val windows, 512 calibration
windows, 2 foreign experts per expert for the refit).
    python -m dd.r28_long train --model dense --phase first
    python -m dd.r28_long train --model pl --phase cont
    python -m dd.r28_long measure --model pl
    python -m dd.r28_long report
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .r21_moe import CTX, E, Data, MoE, eval_ce, load_base, load_moe
from .r28_train import R28, loss_fn
from .ts_train import load_stream, windows

OUT = "runs_home/r28s3"
FIRST_SAVES = (2000, 5000, 10000)
CONT_SAVES = (20000, 30000, 40000)
PL_FIRST = "runs_home/r28s1/r28_base"                    # а, stage 1: saves 2k / 5k / 10k
MODELS = ("dense", "pl", "hash")
RU = dict(dense="dense-cont", pl="а) perlayer", hash="hash")


def args_defaults(a):
    for k, v in dict(data="data_mix", ext="data_mix_ext", init_from="runs_vast/Amix_s0/model.pt",
                     books="runs_vast/r21_books/books.pt", calib="data_mix/calib_train.bin", data_seed=0,
                     skip_steps=20000, bs=32, lr=3e-4, warmup=200, wd=0.1, start_mult=0.1).items():
        if not hasattr(a, k):
            setattr(a, k, v)
    return a


def saves_of(a):
    if a.smoke:
        return (10, 20, 40), (60, 80)
    return FIRST_SAVES, CONT_SAVES


def run_dir(a, model):
    if model == "pl" and not a.smoke:
        return None                                       # first phase lives in PL_FIRST, continuation in OUT/pl
    return os.path.join(a.out, model)


def ckpt_path(a, model, step):
    first, _ = saves_of(a)
    if model == "pl" and step in first:
        return os.path.join(a.pl_first, f"model_{step}_bf16.pt")
    return os.path.join(a.out, model, f"model_{step}_bf16.pt")


# ------------------------------------------------------------------------------------------------ models
def new_model(a, model, dev, D=None):
    books = torch.load(a.books, map_location="cpu", weights_only=False)
    base = load_base(a.init_from, dev)
    if model == "dense":
        m = MoE(base, "dense", books["N"], books=books).to(dev)
    else:
        m = R28(base, "base" if model == "pl" else "hash", books).to(dev)
        m.init_vecs = [[torch.cat([p[k].detach().reshape(-1).float() for p in m.expert_params(j)]).cpu()
                        for k in range(E)] for j in range(m.M)]
    return m, books


def save(m, model, path):
    pn = {n for n, _ in m.named_parameters()}
    st = {k: (v.to(torch.bfloat16) if k in pn else v) for k, v in m.state_dict().items()}
    if model == "dense":                                  # the r21 format: dd.r21_moe.load_moe reads it
        torch.save(dict(cfg=m.base.cfg.to_dict(), variant="dense", N=m.N, state=st), path)
    else:                                                 # the r28 format (as dd.r28_train saves)
        torch.save(dict(variant="base" if model == "pl" else "hash", state=st), path)


def load(a, model, step, dev):
    m, _ = new_model(a, model, dev)
    ck = torch.load(ckpt_path(a, model, step), map_location="cpu", weights_only=False)
    m.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in ck["state"].items()})
    return m.eval()


# ------------------------------------------------------------------------------------------------ training
def lr_first(a, s, steps):
    if s < a.warmup:
        return a.lr * (s + 1) / a.warmup
    p = min(1.0, (s - a.warmup) / max(1, steps - a.warmup))
    return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def lr_cont(a, s):
    """s counted from the start of the continuation: warmup to the end-of-stage-1 lr, then constant"""
    end = a.lr * 0.1
    return end * (s + 1) / a.warmup if s < a.warmup else end


def optimizer(a, m):
    start = {id(p) for p in m.base.blocks[:m.N].parameters()}
    groups = {}
    for n, p in m.named_parameters():
        if p.requires_grad:
            groups.setdefault((a.start_mult if id(p) in start else 1.0, a.wd if p.dim() >= 2 else 0.0), []).append(p)
    return torch.optim.AdamW([dict(params=v, lr=a.lr * mu, weight_decay=w, mult=mu) for (mu, w), v in groups.items()],
                             lr=a.lr, betas=(0.9, 0.95), fused=True)


def train_cmd(a):
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    first, cont = saves_of(a)
    out = os.path.join(a.out, a.model)
    os.makedirs(out, exist_ok=True)
    rp = os.path.join(out, "train.json")
    res = json.load(open(rp)) if os.path.exists(rp) else {}
    key = a.phase
    if res.get(key, {}).get("done"):
        print(f"{a.model} {a.phase}: done before, skipped", flush=True)
        return
    D = Data(a.data, dev)
    gv = torch.Generator(device=dev).manual_seed(1234)                       # as r21 / r28: the same val windows
    voffs = torch.randint(0, D.val.numel() - CTX - 1, (16 if a.smoke else 256,), device=dev, generator=gv)
    if a.phase == "first":
        assert not (a.model == "pl" and not a.smoke), "а's first phase is stage 1 (runs_home/r28s1/r28_base)"
        m, _ = new_model(a, a.model, dev)
        stream = load_stream(os.path.join(a.data, "train.bin"), dev)
        n_win = (stream.numel() - 1) // (CTX + 1)
        steps, s0, saves = first[-1], 0, first
        perm = torch.randperm(n_win, generator=torch.Generator().manual_seed(a.data_seed))[
            a.skip_steps * a.bs:(a.skip_steps + steps) * a.bs]
        lr_at = (lambda s: lr_first(a, s, steps))                                  # noqa: E731
    else:
        m = load(a, a.model, first[-1], dev)
        stream = load_stream(os.path.join(a.ext, "train.bin"), dev)
        n_win = (stream.numel() - 1) // (CTX + 1)
        s0, steps, saves = first[-1], cont[-1] - first[-1], cont
        assert steps * a.bs <= n_win, f"data_mix_ext has {n_win} windows, {steps * a.bs} needed"
        perm = torch.randperm(n_win, generator=torch.Generator().manual_seed(a.data_seed))[:steps * a.bs]
        lr_at = (lambda s: lr_cont(a, s))                                          # noqa: E731
    perm = perm.to(dev) * (CTX + 1)
    m.train()
    vce0, src0 = eval_ce(m, D, voffs)
    # start checks: the first phase computes the A-mix function; the continuation starts where the first phase ended
    if a.phase == "first":
        books = torch.load(a.books, map_location="cpu", weights_only=False)
        d0 = MoE(load_base(a.init_from, dev), "dense", books["N"], books=books).to(dev)
        ref, what = eval_ce(d0, D, voffs)[0], "A-mix"
        del d0
    else:
        lp = (os.path.join(a.pl_first, "r28.json") if a.model == "pl" and not a.smoke else rp)
        lg = json.load(open(lp))
        lg = lg["val_log"] if "val_log" in lg else lg["first"]["val_log"]
        ref, what = [e["val_ce"] for e in lg if e["step"] == first[-1]][-1], f"the first phase at {first[-1]}"
    print(f"  start check: {vce0:.5f} vs {what} {ref:.5f}", flush=True)
    assert abs(vce0 - ref) < 2e-3, f"start differs from {what}: {vce0:.5f} vs {ref:.5f}"
    r = res.setdefault(key, {})
    r.update(start_step=s0, steps=steps, val_log=[dict(step=s0, val_ce=vce0, val_ce_src=src0)])
    print(f"{a.model} {a.phase}: steps {s0} -> {s0 + steps}; val CE at start {vce0:.5f}", flush=True)
    opt = optimizer(a, m)
    t0 = time.time()
    m.train()
    for s in range(steps):
        idx = windows(stream, perm[s * a.bs:(s + 1) * a.bs], CTX)
        for g in opt.param_groups:
            g["lr"] = lr_at(s) * g["mult"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ce, _ = loss_fn(m, idx)
        opt.zero_grad(set_to_none=True)
        ce.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
        opt.step()
        g_step = s0 + s + 1
        if s % 500 == 0:
            print(f"{g_step - 1:6d} ce {float(ce):.4f} gn {float(gn):.2f} lr {lr_at(s):.2e} {time.time() - t0:.0f}s mem "
                  f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G", flush=True)
        if g_step % (20 if a.smoke else 1000) == 0 or s == steps - 1:
            vce, vsrc = eval_ce(m, D, voffs)
            m.train()
            r["val_log"].append(dict(step=g_step, val_ce=vce, val_ce_src=vsrc, t=time.time() - t0))
            json.dump(res, open(rp, "w"), indent=1)
            print(f"  [val] step {g_step} ce {vce:.4f}", flush=True)
        if g_step in saves:
            save(m, a.model, os.path.join(out, f"model_{g_step}_bf16.pt"))
            print(f"  [save] step {g_step}", flush=True)
    r.update(done=True, s_per_step=(time.time() - t0) / steps)
    json.dump(res, open(rp, "w"), indent=1)
    print(f"{a.model} {a.phase}: done ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------------------------------------ measurements
@torch.no_grad()
def measure_cmd(a):
    """every save of a.model not measured yet: full val per window; MoE: measure_all with dense-cont of the same step"""
    from . import r28_measure
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    first, cont = saves_of(a)
    out = os.path.join(a.out, a.model)
    os.makedirs(out, exist_ok=True)
    rp = os.path.join(out, "measure.json")
    res = json.load(open(rp)) if os.path.exists(rp) else {}
    D = Data(a.data, dev)
    offs = D.starts(64) if a.smoke else D.starts()
    for step in first + cont:
        if str(step) in res:
            continue
        if not os.path.exists(ckpt_path(a, a.model, step)):
            print(f"{a.model} {step}: no checkpoint, skipped", flush=True)
            continue
        t0 = time.time()
        m = load(a, a.model, step, dev)
        tot, src, S, C = eval_ce(m, D, offs, per_window=True)
        np.savez(os.path.join(out, f"val_full_{step}.npz"), S=S, C=C)
        r = dict(val_full=dict(ce=tot, src=src))
        if a.model != "dense":
            dn = load_moe(os.path.join(a.out, "dense"), dev=dev, fname=f"model_{step}_bf16.pt")
            r28_measure._REF.clear()
            r28_measure._REF["dense"] = [tuple(getattr(dn.base.blocks[b], w).weight.detach().float()
                                               for w in ("w1", "w3", "w2")) for b in range(m.N, m.L)]
            del dn
            a.m_windows, a.fit_windows, a.n_pairs, a.all_blocks = (16, 32, 2, False) if a.smoke else (1024, 512, 2, False)
            r["measure"] = r28_measure.measure_all(m, D, a)
            r28_measure._REF.clear()
        if a.model == "hash" and step == first[-1] and not a.smoke:      # the r21 hash run, for the reproduction check
            h21 = load_moe("runs_vast/mix_hash", dev=dev)
            _, _, S2, C2 = eval_ce(h21, D, offs, per_window=True)
            np.savez(os.path.join(out, "val_full_r21.npz"), S=S2, C=C2)
            del h21
        r["seconds"] = time.time() - t0
        res[str(step)] = r
        json.dump(res, open(rp, "w"), indent=1)
        sm = r.get("measure", {}).get("summary", {})
        print(f"{a.model} {step}: full val CE {tot:.4f}" + ("; foreign/zero " + " / ".join(
            f"{sm[str(j)]['spec_foreign']:.3f}" for j in range(4)) if sm else "") + f" ({r['seconds']:.0f}s)", flush=True)
        del m
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------ report
def ratio_se(num, den):
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    r = num.sum() / den.sum()
    n = len(den)
    return float(r), float(math.sqrt(((num - r * den) ** 2).sum() / (n - 1) / n) / den.mean())


def wins(out, model, step):
    p = os.path.join(out, model, f"val_full_{step}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z["S"].sum(1), z["C"].sum(1)


def report_cmd(a):
    from .r28_report1 import spec_se
    first, cont = saves_of(a)
    steps = first + cont
    M = {mo: (json.load(open(os.path.join(a.out, mo, "measure.json")))
              if os.path.exists(os.path.join(a.out, mo, "measure.json")) else {}) for mo in MODELS}
    T = {mo: (json.load(open(os.path.join(a.out, mo, "train.json")))
              if os.path.exists(os.path.join(a.out, mo, "train.json")) else {}) for mo in MODELS}
    L = ["# Раунд 28, этап 3: растёт ли выигрыш смеси вместе со специализацией (дома)\n"]
    L.append("а (perlayer) и dense-cont — 0 → 10k как этап 1 (dense-cont обучен заново: в r21 сохранена только 10k), "
             "10k → 40k: свежий AdamW, 200 шагов разогрева до lr конца этапа 1 (0.1 от пика), дальше постоянный; окна "
             "`data_mix_ext` (документы после `data_mix`, без повторов), одни и те же для всех моделей. Выигрыш смеси = "
             "CE dense-cont − CE смеси на полном val, парно по окнам, ± 2 SE. Специализация — 1024 окна val; dense-cont / "
             "ноль — с dense-cont того же шага.\n")
    # reproduction checks
    chk = []
    d10 = wins(a.out, "dense", first[-1])
    rd = os.path.join("runs_home/r28s1/refs", "dense_val_full_windows.npz")
    if d10 is not None and os.path.exists(rd) and np.load(rd)["S"].shape[0] == d10[0].shape[0]:
        z = np.load(rd)
        d = ratio_se(d10[0] - z["S"].sum(1), d10[1])
        chk.append(f"dense-cont заново против dense-cont r21 на 10k: {d[0]:+.4f} ± {2 * d[1]:.4f}")
    h10 = wins(a.out, "hash", first[-1])
    rh = os.path.join(a.out, "hash", "val_full_r21.npz")
    if h10 is not None and os.path.exists(rh) and np.load(rh)["S"].shape[0] == h10[0].shape[0]:
        z = np.load(rh)
        d = ratio_se(h10[0] - z["S"].sum(1), h10[1])
        chk.append(f"hash заново против hash r21 на 10k: {d[0]:+.4f} ± {2 * d[1]:.4f}")
    if chk:
        L.append("Проверки воспроизведения: " + "; ".join(chk) + ".\n")
    for mo in ("pl", "hash"):
        if not M[mo]:
            continue
        L.append(f"\n## {RU[mo]}\n")
        L.append("| шаг | val CE смеси | val CE dense-cont | выигрыш смеси ± 2 SE | чужой / ноль: блоки 5 / 6 / 7 / 8 (± 2 SE "
                 "блока 5) | dense-cont / ноль: блоки 5 / 6 / 7 / 8 | снято подбором: 5 / 6 / 7 / 8 | косинус сдвигов | "
                 "‖сдвиг‖ / ‖A-mix‖ |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for st in steps:
            r, dr = M[mo].get(str(st)), M["dense"].get(str(st))
            if not r:
                continue
            w, wd = wins(a.out, mo, st), wins(a.out, "dense", st)
            gain = ratio_se(wd[0] - w[0], w[1]) if (w is not None and wd is not None) else None
            ms = r.get("measure")
            if ms:
                q = [spec_se(ms["spec"][str(j)], ms["load"][j]) for j in range(4)]
                sm = ms["summary"]
                spec = " / ".join(f"{x['spec']:.3f}" for x in q) + f" (± {2 * q[0]['spec_se']:.3f})"
                dspec = " / ".join(f"{x['kspec']:.3f}" for x in q)
                rf = " / ".join(f"{sm[str(j)]['refit_share']:.2f}" for j in range(4))
                cs = " / ".join(f"{sm[str(j)]['cos_upd']:.2f}" for j in range(4))
                up = f"{np.mean([sm[str(j)]['upd_rel'] for j in range(4)]):.3f}"
            else:
                spec = dspec = rf = cs = up = "—"
            L.append(f"| {st} | {r['val_full']['ce']:.4f} | {dr['val_full']['ce'] if dr else float('nan'):.4f} | "
                     + (f"{gain[0]:+.4f} ± {2 * gain[1]:.4f}" if gain else "—") + f" | {spec} | {dspec} | {rf} | {cs} | {up} |")
    # val log over training (256 windows), all models
    L.append("\n## Ход val CE (256 окон, каждые 1000 шагов)\n")
    L.append("| шаг | " + " | ".join(RU[mo] for mo in MODELS) + " |")
    L.append("|---|" + "---|" * len(MODELS))
    logs = {}
    for mo in MODELS:
        for ph in ("first", "cont"):
            for e in T[mo].get(ph, {}).get("val_log", []):
                logs.setdefault(e["step"], {})[mo] = e["val_ce"]
    if a.model_pl_log and os.path.exists(os.path.join(a.pl_first, "r28.json")):
        for e in json.load(open(os.path.join(a.pl_first, "r28.json"))).get("val_log", []):
            logs.setdefault(e["step"], {}).setdefault("pl", e["val_ce"])
    for st in sorted(logs):
        if st % (5 if a.smoke else 5000) == 0 or st in steps:
            L.append(f"| {st} | " + " | ".join(f"{logs[st][mo]:.4f}" if mo in logs[st] else "—" for mo in MODELS) + " |")
    plot(a, M, steps)
    L.append("\n![выигрыш смеси и специализация по шагам](r28_long.png)")
    txt = "\n".join(L) + "\n"
    name = "r28_long" + ("_smoke" if a.smoke else "") + ".md"
    open(os.path.join("report", name), "w", encoding="utf-8").write(txt)
    print(txt)


def plot(a, M, steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surf, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), dpi=110)
    fig.patch.set_facecolor(surf)
    for ax in axs:
        ax.set_facecolor(surf)
        ax.grid(True, color=grid, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=ink2)
        ax.set_xlabel("шаг", color=ink2)
    for mo, ls in (("pl", "-"), ("hash", "--")):
        xs, g, ge = [], [], []
        for st in steps:
            w, wd = wins(a.out, mo, st), wins(a.out, "dense", st)
            if w is None or wd is None:
                continue
            r = ratio_se(wd[0] - w[0], w[1])
            xs.append(st)
            g.append(r[0])
            ge.append(2 * r[1])
        if xs:
            axs[0].errorbar(xs, g, yerr=ge, fmt="o" + ls, color=ink if mo == "pl" else ink2, lw=2, ms=6, mec=surf,
                            mew=1.5, capsize=3, label=RU[mo])
        for j, c in enumerate(cols):
            sx = [st for st in steps if M[mo].get(str(st), {}).get("measure")]
            if sx:
                axs[1].plot(sx, [M[mo][str(st)]["measure"]["summary"][str(j)]["spec_foreign"] for st in sx], "o" + ls,
                            color=c, lw=2, ms=6, mec=surf, mew=1.5, label=f"{RU[mo]}, блок {5 + j}")
    axs[0].axhline(0, color=ink2, lw=1)
    axs[0].set_title("выигрыш смеси: CE dense-cont − CE смеси (± 2 SE)", color=ink, fontsize=11, loc="left")
    axs[1].set_title("специализация: чужой эксперт / ноль", color=ink, fontsize=11, loc="left")
    for ax in axs:
        ax.axvline(10000 if not a.smoke else 40, color=grid, lw=6, zorder=0)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=8, labelcolor=ink, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join("report", "r28_long" + ("_smoke" if a.smoke else "") + ".png"), facecolor=surf)


EXT_EXPECTED = "runs_home/r28s3/ext_expected.json"         # sha256 of data_mix_ext/train.bin built at home


def vast_cmd(a):
    """on a vast machine (data_mix built and verified by the driver): build data_mix_ext from HF and check its sha256
    against the home build, then the first phase (dense, hash) and the continuation of a.model"""
    from . import mix_data_ext
    from .wiki_data import sha256
    exp = json.load(open(EXT_EXPECTED))["sha256"]["train.bin"]
    tb = os.path.join(a.ext, "train.bin")
    if not os.path.exists(tb):
        mix_data_ext.main(a.ext)
    got = sha256(tb)
    assert got == exp, f"data_mix_ext differs from the home build: {got} vs {exp}"
    print(f"data_mix_ext verified ({got[:12]})", flush=True)
    if a.model != "pl":
        a.phase = "first"
        train_cmd(a)
    a.phase = "cont"
    train_cmd(a)


def fetchwait_cmd(a):
    """at home: wait until the driver has fetched the vast job r28s3_<model> (runs_vast/<job>/status.json), then copy
    its checkpoints and train.json to runs_home/r28s3/<model>; exit 1 if the job failed or after a.wait_h hours"""
    import shutil
    import sys
    job = f"r28s3_{a.model}"
    src, dst = os.path.join("runs_vast", job), os.path.join(a.out, a.model)
    first, cont = saves_of(a)
    t0 = time.time()
    while not (os.path.exists(os.path.join(src, "status.json")) and os.path.exists(os.path.join(src, "vast.json"))):
        if time.time() - t0 > a.wait_h * 3600:
            sys.exit(f"{job}: no results after {a.wait_h} h")
        time.sleep(60)
    st = json.load(open(os.path.join(src, "status.json")))
    if st.get("state") != "done":
        sys.exit(f"{job}: state {st.get('state')} (stage {st.get('stage')}, rc {st.get('rc')})")
    need = [f"model_{x}_bf16.pt" for x in ((first if a.model != "pl" else ()) + cont)] + ["train.json"]
    miss = [f for f in need if not os.path.exists(os.path.join(src, f))]
    if miss:
        sys.exit(f"{job}: missing {miss}")
    os.makedirs(dst, exist_ok=True)
    for f in need:
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    tj = json.load(open(os.path.join(dst, "train.json")))
    assert tj.get("cont", {}).get("done") and (a.model == "pl" or tj.get("first", {}).get("done")), list(tj)
    print(f"{job}: copied {len(need)} files to {dst} (waited {(time.time() - t0) / 60:.0f} min)", flush=True)


def smoke_cmd(a):
    """the whole chain in small (runs_home/r28s3t): first phase 0 -> 40 of every model, continuation 40 -> 80 on
    data_mix_ext, measurements at every save, report/r28_long_smoke.md"""
    for mo in MODELS:
        a.model, a.phase = mo, "first"
        train_cmd(a)
    for mo in MODELS:
        a.model, a.phase = mo, "cont"
        train_cmd(a)
    for mo in MODELS:
        a.model = mo
        measure_cmd(a)
    report_cmd(a)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "measure", "report", "smoke", "vast", "fetchwait"])
    ap.add_argument("--wait_h", type=float, default=8.0)
    ap.add_argument("--model", default="pl", choices=MODELS)
    ap.add_argument("--phase", default="cont", choices=["first", "cont"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--pl_first", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "smoke":
        a.smoke = True
    args_defaults(a)
    a.out = a.out or ("runs_home/r28s3t" if a.smoke else OUT)
    a.pl_first = a.pl_first or (os.path.join(a.out, "pl") if a.smoke else PL_FIRST)
    a.model_pl_log = not a.smoke
    dict(train=train_cmd, measure=measure_cmd, report=report_cmd, smoke=smoke_cmd, vast=vast_cmd,
         fetchwait=fetchwait_cmd)[a.cmd](a)


if __name__ == "__main__":
    main()
