"""Round 10: no-grid QG / static and the ordered gate (QG25-ordered), on finished runs (no training).

    python -m dd.r10_diag kdist runs_vast/QG25ordered_s0      # k per (token, block): prefix check, distribution
    python -m dd.r10_diag state runs_vast/QG25nogrid_s0       # state scale per block (no clip) vs the gridded run
    python -m dd.r10_diag drift runs_vast/QG25ordered_s0      # decision flips 10k / 15k / 20k (needs ckpt_010000/015000.pt)
    python -m dd.r10_diag report                              # report/r10_tables.md
"""
import json
import os
import sys

import numpy as np
import torch

from .model import Cfg

RUNS = {
    "A": ["runs_ts/A_s0", "runs_vast/A_s1"],
    "static": ["runs_vast/QG25static_s0", "runs_vast/QG25static_s1", "runs_vast/QG25static_s2"],
    "QG25": ["runs_ts/QG25_s0", "runs_ts/QG25_s1"],
    "staticinit": ["runs_vast/QG25staticinit_s0", "runs_vast/QG25staticinit_s1"],
    "static-nogrid": ["runs_vast/QG25staticnogrid_s0"],
    "QG25-nogrid": ["runs_vast/QG25nogrid_s0"],
    "QG25-ordered": ["runs_vast/QG25ordered_s0"],
}


def cfg_of(run):
    p = os.path.join(run, "model.pt")
    return Cfg(**torch.load(p, map_location="cpu", weights_only=False)["cfg"])


def open_bits(run, G=8):
    z = np.load(os.path.join(run, "tok.npz"))
    ob = z["open_bits"]                                                       # N, L (bit g = group g open)
    return np.unpackbits(ob[..., None], axis=-1, bitorder="little")[..., :G].astype(np.int8), z


def kdist(run):
    cfg = cfg_of(run)
    L, G = cfg.n_layers, cfg.depth_groups
    opn, z = open_bits(run, G)
    k = opn.sum(2)                                                            # N, L
    prefix = []
    for l in range(L):
        q = cfg.static_groups(l, G)
        want = (np.arange(G)[None, :] < k[:, l:l + 1]).astype(np.int8)        # queue order
        prefix.append(float((opn[:, l][:, q] == want).all(1).mean()))
    res = dict(run=run, n_tok=int(k.shape[0]), prefix_share_layer=prefix,
               k_mean_layer=k.mean(0).tolist(), k_sd_layer=k.std(0).tolist(),
               k_hist_layer=[np.bincount(k[:, l], minlength=G + 1).astype(float).__truediv__(k.shape[0]).tolist()
                             for l in range(L)],
               k_hist_all=(np.bincount(k.ravel(), minlength=G + 1) / k.size).tolist(),
               static_like_share=float((k == 2).mean()), static_like_layer=(k == 2).mean(0).tolist(),
               tokens_all_static=float((k == 2).all(1).mean()), k_total_mean=float(k.sum(1).mean()))
    # how much of the spread in k is the token itself (per block): R^2 of the per-token mean of k
    cur = z["cur"].astype(np.int64)
    r2 = []
    for l in range(L):
        kk = k[:, l].astype(np.float64)
        m = (np.bincount(cur, kk, cfg.vocab) / np.maximum(np.bincount(cur, minlength=cfg.vocab), 1))[cur]
        r2.append(float(1 - ((kk - m) ** 2).sum() / max(((kk - kk.mean()) ** 2).sum(), 1e-12)))
    res["r2_token_in_sample_layer"] = r2
    json.dump(res, open(os.path.join(run, "kdist.json"), "w"), indent=1)
    print(json.dumps({k_: v for k_, v in res.items() if "hist_layer" not in k_}, indent=1))
    return res


@torch.no_grad()
def state(run, n_win=64):
    """RMS of the state after the embedding and every block (no clip in nogrid) and the share beyond +-K delta"""
    from .ts_eval import load_run, load_stream, val_starts, batches
    model, cfg = load_run(run)
    val = load_stream(os.path.join("data_ts", "val.bin"), "cuda")
    starts = val_starts(val)[:n_win]
    L = cfg.n_layers
    ss, beyond, cnt = np.zeros(L + 1), np.zeros(L + 1), 0
    for idx in batches(val, starts):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, aux = model(idx, collect=True)
        for i, x in enumerate(aux["xs"]):
            xf = x.float()
            ss[i] += float(xf.pow(2).sum())
            beyond[i] += float((xf.abs() > cfg.q_levels * cfg.qd(i) + 1e-6).float().sum())
        cnt += aux["xs"][0].numel()
    res = dict(run=run, rms=np.sqrt(ss / cnt).tolist(), beyond_K_delta=(beyond / cnt).tolist(),
               delta=[cfg.qd(i) for i in range(L + 1)])
    json.dump(res, open(os.path.join(run, "state_rms.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


def ce_of(run):
    p = os.path.join(run, "ts_eval.json")
    return json.load(open(p))["val_ce"] if os.path.exists(p) else None


def report(out="report/r10_tables.md"):
    rows, ces = [], {}
    for name, runs in RUNS.items():
        v = [c for c in (ce_of(r) for r in runs) if c is not None]
        if not v:
            continue
        ces[name] = v
        sd = f"{np.std(v, ddof=1):.4f}" if len(v) > 1 else "—"
        rows.append(f"| {name} | {np.mean(v):.4f} | {sd} | {len(v)} | " + ", ".join(f"{c:.4f}" for c in v) + " |")
    md = ["# Round 10: no grid and the ordered gate", "", "| run | val CE mean | SD (seeds) | seeds | per seed |",
          "|---|---|---|---|---|"] + rows + [""]
    if "static-nogrid" in ces and "QG25-nogrid" in ces:
        g1 = np.mean(ces["static"]) - np.mean(ces["QG25"])
        g0 = np.mean(ces["static-nogrid"]) - np.mean(ces["QG25-nogrid"])
        md += [f"gap static - QG25: grid {g1:+.4f}, no grid {g0:+.4f}, difference {g0 - g1:+.4f}", ""]
    for r in RUNS["QG25-ordered"]:
        p = os.path.join(r, "kdist.json")
        if os.path.exists(p):
            k = json.load(open(p))
            md += ["## QG25-ordered: k per block", "", "| block | mean k | SD k | k=0 | k=1 | k=2 | k=3 | k>=4 | prefix | R2 token |",
                   "|---|---|---|---|---|---|---|---|---|---|"]
            for l, h in enumerate(k["k_hist_layer"]):
                md.append(f"| {l + 1} | {k['k_mean_layer'][l]:.2f} | {k['k_sd_layer'][l]:.2f} | {h[0]:.2f} | {h[1]:.2f} | "
                          f"{h[2]:.2f} | {h[3]:.2f} | {sum(h[4:]):.2f} | {k['prefix_share_layer'][l]:.4f} | "
                          f"{k['r2_token_in_sample_layer'][l]:.2f} |")
            md += ["", f"k = 2 (the static mask) in {k['static_like_share']:.1%} of (token, block); all 8 blocks at k = 2 "
                       f"for {k['tokens_all_static']:.1%} of tokens", ""]
    md += ["## Gate behaviour (QG25 for reference)", "",
           "| run | val CE | open by block 1-8 | mu peak (step) / end | rho(open, A loss) | within token | R2 token | "
           "open by A-loss quintile (of 64) | token-table subst dCE | drift 15k->20k | p>0.9 of open | p 0.5-0.6 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    a_loss = np.load("runs_ts/A_s0/tok.npz")["loss"].astype(np.float32)
    qb = np.digitize(a_loss, np.quantile(a_loss, [0.2, 0.4, 0.6, 0.8]))
    for r in ["runs_ts/QG25_s0", "runs_ts/QG25_s1"] + RUNS["QG25-nogrid"] + RUNS["QG25-ordered"]:
        if not os.path.exists(os.path.join(r, "adapt.json")):
            continue
        e, ad = json.load(open(os.path.join(r, "ts_eval.json"))), json.load(open(os.path.join(r, "adapt.json")))
        mu = [(x["step"], x["mu"][0]) for x in json.load(open(os.path.join(r, "log.json")))["train_log"] if "mu" in x]
        pk = max(mu, key=lambda t: t[1])
        tot = open_bits(r)[0].sum((1, 2))
        dr = json.load(open(os.path.join(r, "drift.json"))) if os.path.exists(os.path.join(r, "drift.json")) else None
        pd = json.load(open(os.path.join(r, "pdist.json"))) if os.path.exists(os.path.join(r, "pdist.json")) else None
        d, t = ad["difficulty"], ad["token_table"]
        md.append(f"| {r} | {e['val_ce']:.4f} | " + " ".join(f"{v:.2f}" for v in e["qg"]["paid_layer"]) +
                  f" | {pk[1]:.2f} ({pk[0]}) / {mu[-1][1]:.2f} | {d['rho_total']:.2f} | {d['rho_within_token']:.2f} | "
                  f"{t['r2_total']:.2f} | " + " / ".join(f"{tot[qb == i].mean():.1f}" for i in range(5)) +
                  f" | {t['val_ce_sub'] - t['val_ce_base']:+.3f} | " +
                  (f"{dr['flip_15k_20k']:.1%}" if dr else "—") + " | " +
                  (f"{pd['p_gt_09']:.0%} | {pd['p_05_06']:.0%} |" if pd else "— | — |"))
    md.append("")
    for r in RUNS["QG25-nogrid"] + RUNS["static-nogrid"]:
        p = os.path.join(r, "state_rms.json")
        if os.path.exists(p):
            s = json.load(open(p))
            md += [f"{r}: state RMS " + " ".join(f"{v:.2f}" for v in s["rms"]) + "; beyond +-K delta " +
                   " ".join(f"{v:.4f}" for v in s["beyond_K_delta"]), ""]
    open(out, "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


@torch.no_grad()
def drift(run, n_win=64):
    """share of gate decisions that differ between ckpt 10k / 15k / final, on the first n_win analysis windows,
    and how many of the 15k -> 20k flips are decisions with a final p in 0.4-0.6 (needs ckpt_010000/015000.pt)"""
    from .model import Model
    from .r8_map import windows
    W = windows()[:n_win]

    def decisions(path):
        ck = torch.load(path, map_location="cuda", weights_only=False)
        m = Model(Cfg(**ck["cfg"])).cuda().eval()
        m.load_state_dict(ck["state"])
        Z = {}

        def fn(blk, mm, zg):
            Z.setdefault(blk.l, []).append(zg.detach()[:, :-1].float().cpu())
            return mm
        for i in range(0, W.shape[0], 16):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                m(W[i:i + 16], ctl={"hc_gate_fn": fn})
        return torch.sigmoid(torch.stack([torch.cat(Z[l]) for l in sorted(Z)], 2)).numpy()   # B, T, L, G
    P = {s: decisions(os.path.join(run, f"ckpt_{s:06d}.pt")) for s in (10000, 15000)}
    P[20000] = decisions(os.path.join(run, "model.pt"))
    o = {s: P[s] > 0.5 for s in P}
    fl, near = o[15000] != o[20000], (P[20000] > 0.4) & (P[20000] < 0.6)
    res = dict(run=run, flip_10k_15k=float((o[10000] != o[15000]).mean()), flip_15k_20k=float(fl.mean()),
               flip_10k_20k=float((o[10000] != o[20000]).mean()),
               share_of_flips_with_final_p_0_4_0_6=float((fl & near).sum() / max(fl.sum(), 1)),
               share_of_decisions_with_final_p_0_4_0_6=float(near.mean()),
               flip_15k_20k_by_block=[float(fl[:, :, l].mean()) for l in range(fl.shape[2])])
    json.dump(res, open(os.path.join(run, "drift.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "kdist":
        kdist(sys.argv[2])
    elif cmd == "state":
        state(sys.argv[2])
    elif cmd == "drift":
        drift(sys.argv[2])
    elif cmd == "report":
        report()
