"""Round 6, night block: quality vs realisable savings frontier and longer training (TinyStories).
Tables report/r6n_tables.md, figures report/r6n_frontier.png, report/r6n_long.png.
python -m dd.report_r6n [runs_ts] [report]"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .report import INK2  # noqa: F401  (shared rcParams)
from .report_r6ts import TYPE_RU, TYPES
from .ts_flops import savings

FAM = {"QG": ("QG", "#2a78d6"), "QGsplit": ("QG-split", "#1baf7a")}
FRONT = [("QG", 25, ["QG25_s0", "QG25_s1"]), ("QG", 35, ["QG35_s0"]), ("QG", 50, ["QG50_s0"]),
         ("QGsplit", 25, ["QGsplit25_s0"]), ("QGsplit", 35, ["QGsplit35_s0"]), ("QGsplit", 50, ["QGsplit50_s0"])]
LONG = [("A", "A_s0", "A40k_s0"), ("QG25", "QG25_s0", "QG25_40k_s0"), ("QG-split25", "QGsplit25_s0", "QGsplit25_40k_s0")]


def jl(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def load(base, name):
    run = os.path.join(base, name)
    ev, log = jl(os.path.join(run, "ts_eval.json")), jl(os.path.join(run, "log.json"))
    if ev is None or log is None:
        return None
    r = dict(name=name, ev=ev, log=log)
    if "qg" in ev and os.path.exists(os.path.join(run, "tok.npz")):
        r["sav"] = savings(run)
    return r


def fl(xs, fmt="{:.2f}"):
    return " ".join(fmt.format(x) for x in xs)


def interp_step(val_log, target):
    """first (linearly interpolated) step at which the val CE curve reaches `target`; None if never"""
    st = [v["step"] for v in val_log]
    ce = [v["val_ce"] for v in val_log]
    for i in range(1, len(ce)):
        if ce[i] <= target < ce[i - 1]:
            return st[i - 1] + (st[i] - st[i - 1]) * (ce[i - 1] - target) / (ce[i - 1] - ce[i])
    return st[0] if ce and ce[0] <= target else None


def main(base="runs_ts", out="report"):
    A = load(base, "A_s0")
    Q0 = load(base, "Q0_s0")
    a_ce = A["ev"]["val_ce"]
    L = []
    # ------------------------------------------------------------------ frontier
    L += ["### Граница «качество — экономия» (20k шагов, полный val)\n",
          "Экономия — реализуемая доля FLOPs блока и FFN в своей архитектуре (как в раунде 5: внимание всех голов "
          "считается всегда; общий FFN: закрытая группа не читает свои строки w2, токен без открытых групп пропускает "
          "FFN целиком; разрезанный FFN: закрытая группа пропускает свой блок FFN; гейт добавлен). Контекст "
          "внимания — средний фактический по анализируемым токенам.\n",
          "| прогон | цель | val CE | perplexity | ΔCE к A | ΔCE к Q0 | открыто, доля | экономия: блок / FFN | "
          "открыто по блокам | ρ(открыто, потеря A) | AUROC → top-10% A |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    pts = {}
    for fam, tgt, names in FRONT:
        for n in names:
            r = load(base, n)
            if r is None:
                continue
            e, s, g = r["ev"], r["sav"], r["ev"]["qg"]
            pts.setdefault((fam, tgt), []).append((s["realisable_block"], e["val_ce"] - a_ce))
            L.append(f"| {n} | {tgt / 100:.2f} | {e['val_ce']:.4f} | {e['val_ppl']:.3f} | {e['val_ce'] - a_ce:+.4f} | "
                     f"{e['val_ce'] - Q0['ev']['val_ce']:+.4f} | {s['open_share']:.3f} | "
                     f"{100 * s['realisable_block']:.1f}% / {100 * s['realisable_ffn']:.1f}% | {fl(g['paid_layer'])} | "
                     f"{g.get('spearman_open_vs_A_loss', float('nan')):.3f} | {g.get('auroc_open_top10_A', float('nan')):.3f} |")
    L.append(f"\nОпора: A val CE {a_ce:.4f}; Q0 (та же сетка, без гейта) {Q0['ev']['val_ce']:.4f} "
             f"(ΔCE к A {Q0['ev']['val_ce'] - a_ce:+.4f}).\n")
    # adaptivity by type
    rows = [load(base, n) for _, _, names in FRONT for n in names]
    rows = [r for r in rows if r]
    L += ["### Открытые группы по типу предсказываемого токена (из 64) и по квинтилям потери A\n",
          "| прогон | " + " | ".join(TYPE_RU[t] for t in TYPES) + " | квинтили потери A (лёгкие → трудные) |",
          "|" + "---|" * (len(TYPES) + 2)]
    for r in rows:
        g = r["ev"]["qg"]
        L.append(f"| {r['name']} | " + " | ".join(f"{g['by_type'][t]['open_total']:.1f}" for t in TYPES) +
                 f" | {fl(g.get('open_by_A_loss_quintile', []), '{:.1f}')} |")
    L.append("")
    # where the quality is lost
    L += ["### ΔCE к A по типу предсказываемого токена (анализируемые токены)\n",
          "| прогон | " + " | ".join(TYPE_RU[t] for t in TYPES) + " |", "|" + "---|" * (len(TYPES) + 1)]
    for r in [Q0] + rows:
        L.append(f"| {r['name']} | " + " | ".join(f"{r['ev']['type_loss'][t] - A['ev']['type_loss'][t]:+.3f}"
                                                for t in TYPES) + " |")
    L.append("| доля токенов | " + " | ".join(f"{A['ev']['type_share'][t]:.3f}" for t in TYPES) + " |")
    L.append("")
    # checks
    L += ["### Проверки\n",
          "| прогон | max\\|Δx\\| закрытой | train/eval Δлогит | мёртвых / всегда открытых групп (из 64) | "
          "впустую / открытых (max по блокам) | μ max / в конце | max доля на краях сетки (состояние) |",
          "|---|---|---|---|---|---|---|"]
    extra = [load(base, n) for n in ("QG25_40k_s0", "QGsplit25_40k_s0")]
    for r in rows + [x for x in extra if x]:
        g, ev = r["ev"]["qg"], r["ev"]
        mus = [t["mu"][0] for t in r["log"]["train_log"] if "mu" in t]
        es = ev["levels"]["edge_share"]
        i = int(np.argmax(es))
        L.append(f"| {r['name']} | {g['max_abs_change_closed_group']:.0g} | {g['train_eval_max_abs_logit_diff']:.0g} | "
                 f"{int(round(8 * sum(g['never_open_layer'])))} / {int(round(8 * sum(g['always_open_layer'])))} | "
                 f"{max(g['wasted_of_paid_layer']):.3f} | {max(mus):.3f} / {ev['mu_final'][0]:.3f} | "
                 f"{100 * es[i]:.2f}% ({'emb' if i == 0 else f'b{i - 1}'}) |")
    L.append("")
    # ------------------------------------------------------------------ long training
    L += ["### Долгое обучение: 20k против 40k шагов (косинус на всю длину, полный val)\n",
          "| вариант | val CE 20k | ΔCE к A-20k | val CE 40k | ΔCE к A-40k | улучшение 20k → 40k | шаг A той же длины "
          "с таким же val CE (20k / 40k) |", "|---|---|---|---|---|---|---|"]
    A40 = load(base, "A40k_s0")
    for lab, n20, n40 in LONG:
        r20 = [load(base, n) for n in ([n20] + (["QG25_s1"] if n20 == "QG25_s0" else []))]
        r20 = [r for r in r20 if r]
        r40 = load(base, n40)
        if not r20:
            continue
        v20 = np.mean([r["ev"]["val_ce"] for r in r20])
        s20 = "; ".join(f"{r['ev']['val_ce']:.4f}" for r in r20)
        d20 = "; ".join(f"{r['ev']['val_ce'] - a_ce:+.4f}" for r in r20)
        if r40 and A40:
            v40 = r40["ev"]["val_ce"]
            d40 = f"{v40 - A40['ev']['val_ce']:+.4f}"
            # equivalent A steps on the 256-window training-time val curve
            e20 = interp_step(A["log"]["val_log"], np.mean([r["log"]["val_log"][-1]["val_ce"] for r in r20]))
            e40 = interp_step(A40["log"]["val_log"], r40["log"]["val_log"][-1]["val_ce"])
            eq = f"{e20 / 1000:.1f}k / {e40 / 1000:.1f}k" if e20 and e40 else "—"
            L.append(f"| {lab} | {s20} | {d20} | {v40:.4f} | {d40} | {v40 - v20:+.4f} | {eq if lab != 'A' else '—'} |")
        else:
            L.append(f"| {lab} | {s20} | {d20} | — | — | — | — |")
    L.append("")
    open(os.path.join(out, "r6n_tables.md"), "w", encoding="utf-8").write("\n".join(L))
    # ------------------------------------------------------------------ figures
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for fam in ("QG", "QGsplit"):
        lab, col = FAM[fam]
        xs, ys = [], []
        for tgt in (25, 35, 50):
            p = pts.get((fam, tgt), [])
            for x, y in p:
                ax.plot(100 * x, y, "o", color=col, ms=7, mec="white", mew=1.5, zorder=3)
            if p:
                xs.append(100 * np.mean([q[0] for q in p]))
                ys.append(np.mean([q[1] for q in p]))
                ax.annotate(f"{lab}{tgt}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, 6), fontsize=8,
                            color=INK2)
        if xs:
            ax.plot(xs, ys, "-", color=col, lw=1.6, label=lab, zorder=2)
    if Q0:
        ax.plot(0, Q0["ev"]["val_ce"] - a_ce, "s", color="#a39e93", ms=7, mec="white", mew=1.5, label="Q0 (без гейта)")
    ax.axhline(0, color=INK2, lw=1, ls=":")
    ax.axhline(0.02, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("реализуемая экономия FLOPs блока, %")
    ax.set_ylabel("ΔCE к A (val)")
    ax.set_title("качество против экономии, TinyStories, 20k шагов", loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "r6n_frontier.png"), dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    cols = {"A": "#7a7466", "QG25": "#2a78d6", "QG-split25": "#1baf7a"}
    for lab, n20, n40 in LONG:
        for n, ls in ((n20, "--"), (n40, "-")):
            r = jl(os.path.join(base, n, "log.json"))
            if r:
                vl = [v for v in r["val_log"] if v["step"] >= 4000]
                axes[0].plot([v["step"] / 1000 for v in vl], [v["val_ce"] for v in vl], color=cols[lab], ls=ls, lw=1.4,
                             label=f"{lab} {'20k' if n == n20 else '40k'}")
    axes[0].set_title("val CE (256 окон) по ходу обучения", loc="left")
    axes[0].set_xlabel("шаг, тыс.")
    axes[0].legend(fontsize=8)
    a20 = {v["step"]: v["val_ce"] for v in A["log"]["val_log"]}
    a40 = {v["step"]: v["val_ce"] for v in A40["log"]["val_log"]} if A40 else {}
    for lab, n20, n40 in LONG[1:]:
        for n, ref, ls in ((n20, a20, "--"), (n40, a40, "-")):
            r = jl(os.path.join(base, n, "log.json"))
            if r and ref:
                vv = [(v["step"], v["val_ce"] - ref[v["step"]]) for v in r["val_log"] if v["step"] in ref and v["step"] >= 2000]
                axes[1].plot([s / 1000 for s, _ in vv], [d for _, d in vv], color=cols[lab], ls=ls, lw=1.4,
                             label=f"{lab} {'20k' if n == n20 else '40k'}")
    axes[1].axhline(0, color=INK2, lw=1, ls=":")
    axes[1].set_title("отставание от A того же расписания (ΔCE)", loc="left")
    axes[1].set_xlabel("шаг, тыс.")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "r6n_long.png"), dpi=130)
    plt.close(fig)
    # ------------------------------------------------------------------ generations (A reference + QG50, QG-split50)
    G = [(lab_, jl(os.path.join(base, n, "generations.json")))
         for lab_, n in (("A", "A_s0"), ("QG50", "QG50_s0"), ("QG-split50", "QGsplit50_s0"))]
    G = [(lab_, g) for lab_, g in G if g]
    if len(G) > 1:
        Lg = ["# TinyStories: генерации QG50 и QG-split50 (A для сравнения)\n",
              "Те же затравки и настройки, что в части 2: жадно и с температурой 0.8 (seed = номер затравки), "
              "до 120 токенов.\n"]
        for i, g0 in enumerate(G[0][1]):
            Lg.append(f"## Затравка {i + 1}: «{g0['prompt']}»\n")
            for mode, title in (("greedy", "жадно"), ("temp08", "T = 0.8")):
                Lg.append(f"**{title}**\n")
                for lab_, g in G:
                    Lg.append(f"- *{lab_}*: " + g[i][mode].replace("\n", " ⏎ "))
                Lg.append("")
        open(os.path.join(out, "r6n_generations.md"), "w", encoding="utf-8").write("\n".join(Lg))
    print("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:])
