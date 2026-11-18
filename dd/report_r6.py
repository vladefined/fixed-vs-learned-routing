"""Round 6, part 1 tables (report/r6_tables.md) and figures (report/r6_curves.png, report/r6_depth_by_m.png).
python -m dd.report_r6 [runs8] [report]"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .report import INK2  # noqa: F401  (shared rcParams)
from .report5 import ms
from .report_r5 import f, jk, rank_test

V = ["QG25", "QGsplit25", "QGsplitFFN25", "QGsplitHead25"]
LABEL = {"QG25": "QG25 (общий FFN, полная W_O)", "QGsplit25": "QG-split25 (FFN + головы)",
         "QGsplitFFN25": "QG-splitFFN25 (только FFN)", "QGsplitHead25": "QG-splitHead25 (только головы)"}
SHORT = {"QG25": "QG25", "QGsplit25": "QG-split25", "QGsplitFFN25": "QG-splitFFN25", "QGsplitHead25": "QG-splitHead25"}
COL = {"QG25": "#2a78d6", "QGsplit25": "#1baf7a", "QGsplitFFN25": "#eb6834", "QGsplitHead25": "#6250d6"}


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def load(base):
    R = {}
    for d in sorted(os.listdir(base)):
        p = os.path.join(base, d)
        k = d.rsplit("_s", 1)[0]
        if k in V and os.path.exists(os.path.join(p, "eval_r4.json")):
            R.setdefault(k, []).append(dict(name=d, seed=int(d.rsplit("_s", 1)[1]), log=jl(os.path.join(p, "log.json")),
                                            ev=jl(os.path.join(p, "eval5.json")), r4=jl(os.path.join(p, "eval_r4.json"))))
    for k in R:
        R[k].sort(key=lambda r: r["seed"])
    return R


def stage(log, key):
    return next((v["step"] for v in log["val_log"] if v.get(key, 0) > 0.8), None)


def tables(R, out):
    L = ["### Качество (по сидам s0; s1)\n",
         "| вариант | ответ, % | m = 0…3 | m = 4 | m = 5 | m = 6 | val CE−H | этапы m = 1–2 / 3–4 / ≥ 4 (s0; s1) |",
         "|" + "---|" * 8]
    for k in V:
        rs = R.get(k, [])
        if not rs:
            continue
        ev = [r["ev"] for r in rs]
        m03 = lambda e: 100 * np.mean([e["answer_acc_by_m"][str(m)] for m in range(4)])
        L.append(f"| {LABEL[k]} | {'; '.join(f'{100 * e['answer_acc']:.1f}' for e in ev)} | "
                 f"{'; '.join(f'{m03(e):.1f}' for e in ev)} | " +
                 " | ".join("; ".join(f"{100 * e['answer_acc_by_m'][str(m)]:.1f}" for e in ev) for m in (4, 5, 6)) +
                 f" | {'; '.join(f'{e['val_excess_ce']:.4f}' for e in ev)} | " +
                 "; ".join(" / ".join(jk(stage(r["log"], s)) for s in ("ans_m12", "ans_m34", "ans_m46")) for r in rs) + " |")
    L.append("")
    # jump step comparison
    L.append("### Шаг скачка (m ≥ 4 > 80%) и ранговый тест против QG25 (4 прогона: «нет» = худший ранг)\n")
    L.append("| вариант | шаги скачка | p против QG25 (m ≥ 4) | p против QG25 (m = 3–4) |")
    L.append("|---|---|---|---|")
    ref46 = [stage(r["log"], "ans_m46") for r in R.get("QG25", [])]
    ref34 = [stage(r["log"], "ans_m34") for r in R.get("QG25", [])]
    for k in V:
        rs = R.get(k, [])
        if not rs:
            continue
        j46 = [stage(r["log"], "ans_m46") for r in rs]
        j34 = [stage(r["log"], "ans_m34") for r in rs]
        p46 = "—" if k == "QG25" or not ref46 else f"{rank_test(ref46, j46):.2f}"
        p34 = "—" if k == "QG25" or not ref34 else f"{rank_test(ref34, j34):.2f}"
        L.append(f"| {SHORT[k]} | {', '.join(jk(j) for j in j46)} | {p46} | {p34} |")
    L.append("\nС 2 сидами на вариант минимально возможное p рангового теста = 0.33: тест здесь только ориентир.\n")
    # depth vs m + savings + checks
    L.append("### Глубина на позиции ответа, экономия в своей архитектуре, проверки\n")
    L.append("| прогон | открытых групп на ответе, m = 0 … 6 | m=6 / m=0 | оплачено (блоки 0–3) | впустую / открытых | "
             "max\\|Δx\\| закрытой | закрытых троек | токенов без открытых групп | реализуемая: FFN / блок | μ max / в конце | "
             "подстановка по таблице |")
    L.append("|" + "---|" * 11)
    for k in V:
        for r in R.get(k, []):
            g = r["r4"]["qg"]
            s = g["stats"]
            d = s["answer_shifts_by_m"]
            sk, tp = g["skip"], g["table_prediction"]
            mus = [t["mu"][0] for t in r["log"]["train_log"] if "mu" in t]
            L.append(f"| {r['name']} | {f([d[str(m)] for m in range(7)], '{:.1f}')} | {d['6'] / max(d['0'], 1e-9):.2f} | "
                     f"{f(s['paid_layer'])} | {f(s['wasted_of_paid_layer'], '{:.3f}')} | {s['max_abs_change_of_standing_group']:.0g} | "
                     f"{sk['frac_group_standing']:.2f} | {sk['frac_token_none_open']:.2f} | "
                     f"{100 * sk['post']['realisable_ffn']:.0f}% / {100 * sk['post']['realisable_block']:.0f}% | "
                     f"{max(mus):.2f} / {g['mu_final'][0]:.3f} | {100 * tp['base']['acc']:.1f} → {100 * tp['skip']['acc']:.1f} |")
    L.append("")
    open(os.path.join(out, "r6_tables.md"), "w", encoding="utf-8").write("\n".join(L))


def figs(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), sharey=True)
    for ax, (key, title) in zip(axes, (("ans_m34", "m = 3–4"), ("ans_m46", "m ≥ 4"))):
        for k in V:
            for i, r in enumerate(R.get(k, [])):
                vl = r["log"]["val_log"]
                ax.plot([v["step"] / 1000 for v in vl], [v.get(key, np.nan) for v in vl], color=COL[k], lw=1.3,
                        ls="-" if r["seed"] == 0 else "--", label=f"{SHORT[k]} s{r['seed']}")
        ax.axhline(0.8, color=INK2, lw=1, ls=":")
        ax.set_title(f"точность на валидации, {title}", loc="left")
        ax.set_xlabel("шаг, тыс.")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=8, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(os.path.join(out, "r6_curves.png"), dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.2))
    for k in V:
        for r in R.get(k, []):
            d = r["r4"]["qg"]["stats"]["answer_shifts_by_m"]
            ax.plot(range(7), [d[str(m)] for m in range(7)], color=COL[k], marker="o", ms=4, lw=1.3,
                    ls="-" if r["seed"] == 0 else "--", label=f"{SHORT[k]} s{r['seed']}")
    ax.set_title("открытых групп на позиции ответа, из 32", loc="left")
    ax.set_xlabel("m (число перестановок)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "r6_depth_by_m.png"), dpi=130)
    plt.close(fig)


def main(base="runs8", out="report"):
    R = load(base)
    tables(R, out)
    figs(R, out)
    print("written", os.path.join(out, "r6_tables.md"))


if __name__ == "__main__":
    main(*sys.argv[1:])
