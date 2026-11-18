"""Round 12: size ladder without the grid -- static-nogrid vs QG25-nogrid at 4 sizes (TinyStories, 20k steps, 1 seed).

    python -m dd.ts_adapt run runs_vast/QG25nogrid_d256L8_s0 ...   # rho with A_s0 loss, token table (per QG run)
    python -m dd.r10_diag drift runs_vast/QG25nogrid_d256L8_s0     # decision flips 10k / 15k / 20k
    python -m dd.r12_diag report                                   # report/r12_tables.md + report/r12_ladder.png
Difficulty: A_s0 (d=512/8) loss on the same tokens, as in every earlier round, and the loss of static-nogrid of the
same size (size-matched reference)."""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

SIZES = [  # label, d, blocks, static run, QG25 run
    ("d256/8", 256, 8, "runs_vast/QG25staticnogrid_d256L8_s0", "runs_vast/QG25nogrid_d256L8_s0"),
    ("d512/8", 512, 8, "runs_vast/QG25staticnogrid_s0", "runs_vast/QG25nogrid_s0"),
    ("d512/16", 512, 16, "runs_vast/QG25staticnogrid_d512L16_s0", "runs_vast/QG25nogrid_d512L16_s0"),
    ("d768/12", 768, 12, "runs_vast/QG25staticnogrid_d768L12_s0", "runs_vast/QG25nogrid_d768L12_s0"),
]


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def n_params(run):
    lg = jl(os.path.join(run, "log.json"))
    if lg is None:
        return None, None
    c = lg["result"]["cfg"]
    return lg["result"]["n_params"], lg["result"]["n_params"] - c["vocab"] * c["d"]


def open_total(run, G=8):
    z = np.load(os.path.join(run, "tok.npz"))
    ob = z["open_bits"]
    opn = np.unpackbits(ob[..., None], axis=-1, bitorder="little")[..., :G]
    return opn.sum((1, 2)).astype(np.float64), z["cur"].astype(np.int64), z["loss"].astype(np.float32), opn.shape[1]


def by_tok_resid(x, cur, V=8192):
    m = np.bincount(cur, x, V) / np.maximum(np.bincount(cur, minlength=V), 1)
    return x - m[cur]


def report(out="report/r12_tables.md", png="report/r12_ladder.png"):
    a_loss = np.load("runs_ts/A_s0/tok.npz")["loss"].astype(np.float32)
    qa = np.quantile(a_loss, [0.2, 0.4, 0.6, 0.8])
    rows, gate_rows, pts = [], [], []
    for lab, d, L, rs, rq in SIZES:
        es, eq = jl(os.path.join(rs, "ts_eval.json")), jl(os.path.join(rq, "ts_eval.json"))
        n_tot, n_ne = n_params(rq) if os.path.exists(os.path.join(rq, "log.json")) else n_params(rs)
        cs, cq = (es or {}).get("val_ce"), (eq or {}).get("val_ce")
        ls, lq = jl(os.path.join(rs, "log.json")), jl(os.path.join(rq, "log.json"))
        sps = lambda lg: lg["result"]["train_time"] / lg["result"]["steps"] if lg else float("nan")
        gap = cs - cq if cs is not None and cq is not None else None
        rows.append(f"| {lab} | {n_tot / 1e6:.1f}M | {n_ne / 1e6:.1f}M | " +
                    (f"{cs:.4f}" if cs else "—") + " | " + (f"{cq:.4f}" if cq else "—") + " | " +
                    (f"{gap:+.4f}" if gap is not None else "—") + f" | {sps(ls):.3f} / {sps(lq):.3f} |"
                    if n_tot else f"| {lab} | — | — | — | — | — | — |")
        if gap is not None:
            pts.append((lab, n_tot, n_ne, cs, cq, gap, ls, lq))
        # gate behaviour of QG25-nogrid
        if eq is None or not os.path.exists(os.path.join(rq, "tok.npz")):
            continue
        tot, cur, own, LL = open_total(rq)
        share = tot / (LL * 8)
        ad, dr = jl(os.path.join(rq, "adapt.json")), jl(os.path.join(rq, "drift.json"))
        rho_s = rho_sw = float("nan")
        if es is not None and os.path.exists(os.path.join(rs, "tok.npz")):
            s_loss = np.load(os.path.join(rs, "tok.npz"))["loss"].astype(np.float32)
            rho_s = float(spearmanr(tot, s_loss).statistic)
            rho_sw = float(spearmanr(by_tok_resid(tot, cur), by_tok_resid(s_loss.astype(np.float64), cur)).statistic)
        qb = np.digitize(a_loss, qa)
        mu = [(x["step"], x["mu"][0]) for x in lq["train_log"] if "mu" in x]
        pk = max(mu, key=lambda t: t[1])
        gate_rows.append(
            f"| {lab} | " + " ".join(f"{v:.2f}" for v in eq["qg"]["paid_layer"]) +
            f" | {float(share.mean()):.3f} | {pk[1]:.2f} ({pk[0]}) / {mu[-1][1]:.2f} | " +
            (f"{ad['difficulty']['rho_total']:.2f} | {ad['difficulty']['rho_within_token']:.2f} | "
             if ad else f"{float(spearmanr(tot, a_loss).statistic):.2f} | — | ") +
            f"{rho_s:.2f} | {rho_sw:.2f} | " +
            (f"{ad['token_table']['r2_total']:.2f} | " if ad else "— | ") +
            " / ".join(f"{share[qb == i].mean():.3f}" for i in range(5)) + " | " +
            (f"{dr['flip_10k_15k']:.1%} | {dr['flip_15k_20k']:.1%} | {dr.get('share_of_flips_with_final_p_0_4_0_6', dr.get('share_of_flips_with_final_p_0.4_0.6')):.0%} |"
             if dr else "— | — | — |"))
    md = ["# Round 12: size ladder without the grid (static-nogrid vs QG25-nogrid, 1 seed, 20k steps)", "",
          "| size | params | non-emb | static-nogrid | QG25-nogrid | static − QG25 | s/step static / QG25 |",
          "|---|---|---|---|---|---|---|"] + rows + [""]
    md += ["## QG25-nogrid gate by size", "",
           "| size | open by block | open share | mu peak (step) / end | rho(open, A_s0 loss) | within token | "
           "rho(open, static same size) | within token | R2 token | open share by A-loss quintile | "
           "drift 10k->15k | drift 15k->20k | flips with final p 0.4-0.6 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"] + gate_rows + [""]
    # gap along training (256 windows)
    md += ["## static − QG25 along training (256 val windows)", "", "| step | " + " | ".join(p[0] for p in pts) + " |",
           "|---|" + "---|" * len(pts)]
    for s in (2000, 4000, 8000, 12000, 16000, 20000):
        cells = []
        for p in pts:
            vs = {x["step"]: x["val_ce"] for x in p[6]["val_log"]}
            vq = {x["step"]: x["val_ce"] for x in p[7]["val_log"]}
            cells.append(f"{vs[s] - vq[s]:+.4f}" if s in vs and s in vq else "—")
        md.append(f"| {s} | " + " | ".join(cells) + " |")
    md.append("")
    open(out, "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md))
    if len(pts) >= 2:
        plot(pts, png)


def plot(pts, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    c_static, c_qg = "#2a78d6", "#eb6834"                      # categorical slots 1, 2 (reference palette)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "axes.labelcolor": ink2, "xtick.color": ink2,
                         "ytick.color": ink2, "axes.facecolor": surf, "figure.facecolor": surf})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.array([p[1] for p in pts]) / 1e6
    gap = np.array([p[5] for p in pts])
    a1.axhline(0, color=ink2, lw=1)
    a1.plot(x, gap, color=c_static, lw=2, marker="o", ms=8, mec=surf, mew=2)
    for xi, g, p in zip(x, gap, pts):
        a1.annotate(f"{p[0]}\n{g:+.3f}", (xi, g), textcoords="offset points", xytext=(0, 9),
                    ha="center", va="bottom", fontsize=9, color=ink)
    a1.set_ylim(float(gap.min()) - 0.006, 0.003)
    a1.set_xlim(x.min() / 1.6, x.max() * 1.6)
    a1.set_xscale("log")
    a1.set_xlabel("parameters, M (log)")
    a1.set_ylabel("val CE: static − QG25 (nats)")
    a1.set_title("static − QG25 without the grid (< 0: static better)", color=ink, fontsize=11, loc="left")
    for c, i, lab in ((c_static, 3, "static-nogrid"), (c_qg, 4, "QG25-nogrid")):
        y = np.array([p[i] for p in pts])
        a2.plot(x, y, color=c, lw=2, marker="o", ms=8, mec=surf, mew=2, label=lab)
        a2.annotate(lab, (x[-1], y[-1]), textcoords="offset points", xytext=(8, 0), va="center", fontsize=9, color=ink)
    a2.set_xscale("log")
    a2.set_xlabel("parameters, M (log)")
    a2.set_ylabel("full val CE (nats)")
    a2.set_title("val CE by size", color=ink, fontsize=11, loc="left")
    a2.legend(frameon=False, loc="upper right")
    for a in (a1, a2):
        a.grid(True, color=grid, lw=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print("->", png)


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report()
