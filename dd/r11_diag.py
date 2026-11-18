"""Round 11: where does static beat QG25-ordered?  Per-block budget (QG25-ordered-perblock), random per-token k
(static-randk) and a fixed but unequal k per block (static-unequal), on finished runs (no training).

    python -m dd.r10_diag kdist runs_vast/QG25orderedperblock_s0    # k per (token, block) (also for randk / unequal)
    python -m dd.r10_diag drift runs_vast/QG25orderedperblock_s0    # needs ckpt_010000/015000.pt
    python -m dd.ts_adapt run runs_vast/QG25orderedperblock_s0      # rho / token table
    python -c "from dd.r9_diag import pdist; pdist('QG25orderedperblock_s0', 'runs_vast')"
    python -m dd.r11_diag report                                     # report/r11_tables.md
"""
import json
import os
import sys

import numpy as np

from .r10_diag import open_bits

RUNS = {
    "A": ["runs_ts/A_s0", "runs_vast/A_s1"],
    "static": ["runs_vast/QG25static_s0", "runs_vast/QG25static_s1", "runs_vast/QG25static_s2"],
    "QG25": ["runs_ts/QG25_s0", "runs_ts/QG25_s1"],
    "QG25-ordered": ["runs_vast/QG25ordered_s0"],
    "QG25-ordered-perblock": ["runs_vast/QG25orderedperblock_s0"],
    "static-randk": ["runs_vast/QG25randk_s0"],
    "static-unequal": ["runs_vast/QG25staticunequal_s0"],
}
# pairs (x, ref): delta = x - ref
PAIRS = [("QG25-ordered-perblock", "QG25-ordered"), ("QG25-ordered-perblock", "static"),
         ("static-randk", "static"), ("static-unequal", "static"),
         ("QG25-ordered", "static"), ("static-randk", "QG25-ordered"), ("static-unequal", "QG25-ordered")]


def ce_of(run):
    """full val CE; for random masks (randk) the mean over the 3 mask seeds"""
    p = os.path.join(run, "ts_eval.json")
    if not os.path.exists(p):
        return None
    e = json.load(open(p))
    return e.get("val_ce_mask_mean", e["val_ce"])


def mu_traj(run):
    tl = json.load(open(os.path.join(run, "log.json")))["train_log"]
    return [(x["step"], x["mu"]) for x in tl if "mu" in x]


def report(out="report/r11_tables.md"):
    ces, rows = {}, []
    for name, runs in RUNS.items():
        v = [c for c in (ce_of(r) for r in runs) if c is not None]
        if not v:
            continue
        ces[name] = v
        sd = f"{np.std(v, ddof=1):.4f}" if len(v) > 1 else "—"
        rows.append(f"| {name} | {np.mean(v):.4f} | {sd} | {len(v)} | " + ", ".join(f"{c:.4f}" for c in v) + " |")
    md = ["# Round 11: per-block budget, random k, unequal fixed k", "",
          "| run | val CE mean | SD (seeds) | seeds | per seed |", "|---|---|---|---|---|"] + rows + [""]
    for r in RUNS["static-randk"]:
        p = os.path.join(r, "ts_eval.json")
        if os.path.exists(p):
            e = json.load(open(p))
            md += [f"{r}: mask seeds " + ", ".join(f"{c:.4f}" for c in e["val_ce_mask_seeds"]) +
                   f" (SD {e['val_ce_mask_std']:.4f}); mask seed 0 pass {e['val_ce']:.4f}", ""]
    # 2 SD rule: the reference spread is the seed SD of the run with several seeds (or the largest one known)
    ref_sd = {n: float(np.std(v, ddof=1)) for n, v in ces.items() if len(v) > 1}
    sd_max = max(ref_sd.values()) if ref_sd else float("nan")
    md += ["| comparison | delta | 2 SD (seed spread) | verdict |", "|---|---|---|---|"]
    for x, y in PAIRS:
        if x in ces and y in ces:
            dlt = np.mean(ces[x]) - np.mean(ces[y])
            sd = max(ref_sd.get(x, 0.0), ref_sd.get(y, 0.0)) or sd_max
            ver = "effect not detected" if abs(dlt) < 2 * sd else ("worse" if dlt > 0 else "better")
            md.append(f"| {x} - {y} | {dlt:+.4f} | {2 * sd:.4f} | {ver} |")
    md += ["", f"seed SD known: " + ", ".join(f"{n} {s:.4f}" for n, s in ref_sd.items()) +
           f"; one-seed runs use the larger SD of the pair (max known {sd_max:.4f})", ""]
    # k per block
    for name in ("QG25-ordered", "QG25-ordered-perblock", "static-randk", "static-unequal"):
        for r in RUNS[name]:
            p = os.path.join(r, "kdist.json")
            if not os.path.exists(p):
                continue
            k = json.load(open(p))
            md += [f"## {name}: k per block ({r})", "",
                   "| block | mean k | SD k | k=0 | k=1 | k=2 | k=3 | k>=4 | prefix | R2 token |",
                   "|---|---|---|---|---|---|---|---|---|---|"]
            for l, h in enumerate(k["k_hist_layer"]):
                md.append(f"| {l + 1} | {k['k_mean_layer'][l]:.2f} | {k['k_sd_layer'][l]:.2f} | {h[0]:.2f} | {h[1]:.2f} | "
                          f"{h[2]:.2f} | {h[3]:.2f} | {sum(h[4:]):.2f} | {k['prefix_share_layer'][l]:.4f} | "
                          f"{k['r2_token_in_sample_layer'][l]:.2f} |")
            md += ["", "k over all blocks: " + " ".join(f"{v:.3f}" for v in k["k_hist_all"]) +
                   f"; k = 2 in {k['static_like_share']:.1%} of (token, block)", ""]
    # per-block mu (perblock) vs shared mu (ordered)
    for name in ("QG25-ordered", "QG25-ordered-perblock"):
        for r in RUNS[name]:
            if not os.path.exists(os.path.join(r, "log.json")):
                continue
            tr = mu_traj(r)
            M = np.array([m for _, m in tr])                                      # steps, L
            st = np.array([s for s, _ in tr])
            md += [f"## {name}: mu per block ({r})", "", "| block | peak (step) | at 5k | at 10k | end |", "|---|---|---|---|---|"]
            for l in range(M.shape[1]):
                i = int(M[:, l].argmax())
                at = lambda s: M[np.searchsorted(st, s), l] if st[-1] >= s else float("nan")
                md.append(f"| {l + 1} | {M[i, l]:.2f} ({st[i]}) | {at(5000):.2f} | {at(10000):.2f} | {M[-1, l]:.2f} |")
            tl = json.load(open(os.path.join(r, "log.json")))["train_log"]
            p35 = [x for x in tl if x["step"] == 3500]
            if p35:
                md += ["", "open share by block at step 3500 (train batch): " +
                       " ".join(f"{v:.2f}" for v in p35[0]["q_l0_layer"])]
            md.append("")
    # gate behaviour (as round 10)
    md += ["## Gate behaviour", "",
           "| run | val CE | open by block 1-8 | rho(open, A loss) | within token | R2 token | "
           "open by A-loss quintile (of 64) | token-table subst dCE | drift 10k->15k | drift 15k->20k | p>0.9 of open | p 0.5-0.6 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    a_loss = np.load("runs_ts/A_s0/tok.npz")["loss"].astype(np.float32)
    qb = np.digitize(a_loss, np.quantile(a_loss, [0.2, 0.4, 0.6, 0.8]))
    for r in ["runs_ts/QG25_s0"] + RUNS["QG25-ordered"] + RUNS["QG25-ordered-perblock"]:
        if not os.path.exists(os.path.join(r, "adapt.json")):
            continue
        e, ad = json.load(open(os.path.join(r, "ts_eval.json"))), json.load(open(os.path.join(r, "adapt.json")))
        tot = open_bits(r)[0].sum((1, 2))
        dr = json.load(open(os.path.join(r, "drift.json"))) if os.path.exists(os.path.join(r, "drift.json")) else None
        pd = json.load(open(os.path.join(r, "pdist.json"))) if os.path.exists(os.path.join(r, "pdist.json")) else None
        d, t = ad["difficulty"], ad["token_table"]
        md.append(f"| {r} | {e['val_ce']:.4f} | " + " ".join(f"{v:.2f}" for v in e["qg"]["paid_layer"]) +
                  f" | {d['rho_total']:.2f} | {d['rho_within_token']:.2f} | {t['r2_total']:.2f} | " +
                  " / ".join(f"{tot[qb == i].mean():.1f}" for i in range(5)) +
                  f" | {t['val_ce_sub'] - t['val_ce_base']:+.3f} | " +
                  (f"{dr['flip_10k_15k']:.1%} | {dr['flip_15k_20k']:.1%}" if dr else "— | —") + " | " +
                  (f"{pd['p_gt_09']:.0%} | {pd['p_05_06']:.0%} |" if pd else "— | — |"))
    md.append("")
    # val curve (256 windows) along training
    md += ["## val CE along training (256 windows)", "", "| step | " + " | ".join(RUNS) + " |",
           "|---|" + "---|" * len(RUNS)]
    curves = {}
    for name, runs in RUNS.items():
        cs = []
        for r in runs:
            p = os.path.join(r, "log.json")
            if os.path.exists(p):
                cs.append({x["step"]: x["val_ce"] for x in json.load(open(p))["val_log"]})
        if cs:
            curves[name] = {s: np.mean([c[s] for c in cs if s in c]) for s in cs[0]}
    for s in (2000, 4000, 8000, 12000, 16000, 20000):
        md.append(f"| {s} | " + " | ".join(f"{curves[n][s]:.4f}" if n in curves and s in curves[n] else "—"
                                           for n in RUNS) + " |")
    md.append("")
    open(out, "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report()
