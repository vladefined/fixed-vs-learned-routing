"""Tables (report/tables.md) and figures (report/*.png) for the v4 grid + the legacy loophole pilots.
python -m dd.report [runs4] [report]"""
from __future__ import annotations

import json
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"A": "#52514e", "A_norm": "#a3a29c",
       "B_l0": "#9ec5f4", "B_w": "#5598e7", "B_m": "#256abf", "B_s": "#0d366b",
       "Bdd": "#eb6834", "Bpull": "#1baf7a", "C": "#eda100", "E": "#e87ba4", "F": "#008300", "G": "#6250d6"}
LS = {"A": "-", "A_norm": "--"}
LABEL = {"A": "A", "A_norm": "A_norm", "B_l0": "B λ=0", "B_w": "B λ слабый", "B_m": "B λ средний",
         "B_s": "B λ сильный", "Bdd": "B без min (dd)", "Bpull": "B′ подтягивание", "C": "C tanh(u)",
         "E": "E группы", "F": "F на токен", "G": "G группы+внимание"}
ORDER = list(LABEL)
LAMBDA_SET = ["A", "A_norm", "B_l0", "B_w", "B_m", "B_s"]
VARIANT_SET = ["A", "B_m", "Bdd", "Bpull", "C", "E", "F", "G"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 5,
})


def key_of(name):
    k = name.rsplit("_s", 1)[0]
    return k[:-2] if k in ("Bdd_m", "Bpull_m", "C_m", "E_m", "F_m", "G_m") else k


def load(runs_dir):
    out = {}
    for d in sorted(os.listdir(runs_dir)):
        p = os.path.join(runs_dir, d)
        if os.path.exists(os.path.join(p, "eval.json")):
            out[d] = dict(ev=json.load(open(os.path.join(p, "eval.json"))),
                          log=json.load(open(os.path.join(p, "log.json"))), key=key_of(d), path=p)
    return out


def primary(R):
    sel = {r["key"]: r for n, r in R.items() if n.endswith("_s0")}
    return {k: sel[k] for k in ORDER if k in sel}


def sorted_runs(R):
    grid_lams = {R[n]["log"]["result"]["lam"] for n in R if R[n]["key"] in ("B_l0", "B_w", "B_m", "B_s") and n.endswith("_s0")}
    keep = [n for n in R if not (n.startswith("B_lam") and n.endswith("_s0") and R[n]["log"]["result"]["lam"] in grid_lams)]
    return sorted(keep, key=lambda n: (ORDER.index(R[n]["key"]) if R[n]["key"] in ORDER else 99, n))


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}"


def fmt_list(v, f="{:.2f}"):
    return " ".join(f.format(x) for x in v)


# ------------------------------------------------------------------ tables
def tables(R, out):
    L = []
    L.append("### Качество (test; CE − H в нат/токен; точность ответа по числу перестановок m)\n")
    L.append("| run | λ | парам. | val CE−H | test CE−H | NLL ответа | ответ | m=0 | 1 | 2 | 3 | 4 | 5 | 6 | "
             "k=3 | k=4 | согл. | согл.+аттр. | закрытие top1 | мин |")
    L.append("|" + "---|" * 20)
    for n in sorted_runs(R):
        e, lg = R[n]["ev"], R[n]["log"]["result"]
        bm = e.get("answer_acc_by_m", {})
        bk = e.get("answer_acc_by_k", {})
        L.append(f"| {n} | {lg['lam']:g} | {lg['n_params'] / 1e6:.2f}M | {e['val_excess_ce']:.4f} | "
                 f"{e['test_excess_ce']:.4f} | {e['answer_nll']:.3f} | {pct(e['answer_acc'])} | " +
                 " | ".join(pct(bm.get(str(m))) for m in range(7)) + " | " +
                 f"{pct(bk.get('3'))} | {pct(bk.get('4'))} | {pct(e['agree_acc'])} | "
                 f"{pct(e['agree_acc_attractor'])} | {pct(e['close_top1_acc'])} | {lg['train_time'] / 60:.1f} |")
    L.append("")
    L.append("### Использование глубины и «замёрзшие» координаты (по слоям 0→3)\n")
    L.append("| run | итог. глубина | Σmin(dd,1) | Δd по слоям | доля dd<0.01 | доля |Δx|/RMS<0.01 | "
             "dd<0.01, но |Δx| ≥ 1% | статич. нули | мёртвые гейты | ΔCE обнулить dd<0.01 / <0.1 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for n in sorted_runs(R):
        e = R[n]["ev"]
        if not e["budget"]:
            L.append(f"| {n} | — | — | — | — | {fmt_list(e['frozen_rel_layer'])} | — | — | — | — |")
            continue
        fa = e["frozen_agreement"]
        L.append(f"| {n} | {e['final_depth_mean']:.2f} | {e['eff_update_mean']:.2f} | "
                 f"{fmt_list(e['dd_layer_mean'])} | {fmt_list(e['frozen_dd_layer'])} | "
                 f"{fmt_list(e['frozen_rel_layer'])} | {fmt_list([a['dd_only'] for a in fa])} | "
                 f"{fmt_list(e['static_zero_frac_layer'])} | {fmt_list(e['dead_gate_frac_layer'])} | "
                 f"{e['dce_dd_zeroed']['0.01']:+.4f} / {e['dce_dd_zeroed']['0.1']:+.4f} |")
    L.append("")
    L.append("### Гейты: двухвершинность σ(гейта) по слоям (доля <0.05 / 0.05–0.95 / >0.95)\n")
    L.append("| run | " + " | ".join(f"слой {l}" for l in range(4)) + " |")
    L.append("|---|---|---|---|---|")
    for n in sorted_runs(R):
        e = R[n]["ev"]
        if not e["budget"]:
            continue
        h = np.load(os.path.join(R[n]["path"], "eval_arrays.npz"))["gate_hist"]
        cells = []
        for l in range(h.shape[0]):
            f = h[l] / h[l].sum()
            cells.append(f"{f[:3].sum():.2f} / {f[3:48].sum():.2f} / {f[48:].sum():.2f}")
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("### Какие координаты стоят на месте: разложение дисперсии маски «замёрзла» "
             "(по относительному изменению): статич. (та же координата всегда) / лексич. (зависит от слова) / контекст\n")
    L.append("| run | " + " | ".join(f"слой {l}: доля замёрзших; стат/лекс/конт; всегда/никогда" for l in range(4)) + " |")
    L.append("|---|---|---|---|---|")
    for n in sorted_runs(R):
        e = R[n]["ev"]
        cells = [f"{d['p']:.2f}; {d['static']:.2f}/{d['lexical']:.2f}/{d['context']:.2f}; "
                 f"{d.get('always_frozen', 0):.2f}/{d.get('never_frozen', 0):.2f}" for d in e["decomp_rel"]]
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("### Потраченная глубина в зависимости от числа перестановок m\n")
    L.append("Позиция ответа (токен, после которого модель должна назвать имя) и токены событий. "
             "Для вариантов с бюджетом — итоговая глубина (= Σdd); для всех — Σ по слоям RMS(Δx)/RMS(x).\n")
    L.append("| run | величина | m=0 | 1 | 2 | 3 | 4 | 5 | 6 | ρ(глуб., m) на ответе |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for n in sorted_runs(R):
        sp = R[n]["ev"]["spent"]
        rows = []
        if R[n]["ev"]["budget"]:
            rows.append(("глубина @ответ", sp["answer_pos"]["final_depth"]))
            rows.append(("Σmin(dd,1) @ответ", sp["answer_pos"]["eff_update"]))
            rows.append(("глубина @событие №e (столбцы = e)", sp["event_by_index"]["final_depth"]))
        rows.append(("Σ отн. изм. @ответ", sp["answer_pos"]["relchg_sum"]))
        rows.append(("Σ отн. изм. @событие №e (столбцы = e)", sp["event_by_index"]["relchg_sum"]))
        for i, (nm, d) in enumerate(rows):
            rho = f"{sp.get('answer_pos_spearman_m', float('nan')):+.3f}" if i == 0 and R[n]["ev"]["budget"] else ""
            L.append(f"| {n if i == 0 else ''} | {nm} | " +
                     " | ".join(f"{d[str(m)]:.3f}" if str(m) in d else "—" for m in range(7)) + f" | {rho} |")
    L.append("")
    L.append("### Связь со структурой (токены клаузы; частная ранговая ρ при контроле класса слова и позиции; "
             "95% бутстрап-ДИ по предложениям)\n")
    L.append("| run | сырая ρ(дерево) | частная ρ дерево | частная ρ вложенность | внутри предложения ρ | "
             "t(closes) | t(opens) | t(tree_depth) | не-гл./гл. слово | сущ./прил. |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for n in sorted_runs(R):
        e = R[n]["ev"]
        if not e["budget"]:
            continue
        s = e["structure"]
        o = s["ols_final_depth_std"]
        pt, pe, pw = s["partial_tree"], s["partial_emb"], s["partial_tree_within"]
        L.append(f"| {n} | {s['spearman_tree_raw']:+.3f} | {pt['r']:+.3f} [{pt['ci'][0]:+.3f}, {pt['ci'][1]:+.3f}] | "
                 f"{pe['r']:+.3f} [{pe['ci'][0]:+.3f}, {pe['ci'][1]:+.3f}] | "
                 f"{pw['r']:+.3f} [{pw['ci'][0]:+.3f}, {pw['ci'][1]:+.3f}] | {o['closes']['t']:+.1f} | "
                 f"{o['opens']['t']:+.1f} | {o['tree_depth']['t']:+.1f} | {s['head'][0]:.2f}/{s['head'][1]:.2f} | "
                 f"{s['noun_vs_adj'][0]:.2f}/{s['noun_vs_adj'][1]:.2f} |")
    L.append("")
    L.append("### Устройство потока\n")
    L.append("| run | рост CE при удалении блока 0..3 | эксцесс по токену (оси/поворот) | "
             "эксцесс по координате (оси/поворот) | Hoyer (оси/поворот) | ‖x‖ выход |")
    L.append("|---|---|---|---|---|---|")
    for n in sorted_runs(R):
        e = R[n]["ev"]
        k = e["kurt_final"]
        L.append(f"| {n} | {fmt_list(e['block_removal_dce'])} | {k['tok_kurt']:.1f}/{k['tok_kurt_rot']:.1f} | "
                 f"{k['coord_kurt']:.1f}/{k['coord_kurt_rot']:.1f} | {k['hoyer']:.3f}/{k['hoyer_rot']:.3f} | "
                 f"{e['norm_final']:.1f} |")
    L.append("")
    L.append("### Зонд «какой это токен» по группам координат финального состояния\n")
    L.append("| run | весь вектор | группы: (глубина →) точность |")
    L.append("|---|---|---|")
    for n in sorted_runs(R):
        e = R[n]["ev"]
        gd = e.get("probe_group_depth", {})
        cells = ", ".join(f"{g}: {gd[g]:.2f}→{v:.2f}" if g in gd else f"{g}: {v:.2f}"
                          for g, v in e["probe_groups"].items())
        L.append(f"| {n} | {e['probe_full']:.3f} | {cells} |")
    L.append("")
    # ---- seeds: mean and range
    L.append("### Разброс по сидам (среднее [мин, макс])\n")
    L.append("| вариант | сидов | val CE−H | ответ, % | m=5, % | m=6, % |")
    L.append("|---|---|---|---|---|---|")
    for k in ORDER:
        runs = [R[n] for n in R if R[n]["key"] == k]
        if len(runs) < 2:
            continue

        def agg(f, scale=1.0, nd=4):
            v = [scale * f(r["ev"]) for r in runs]
            return f"{np.mean(v):.{nd}f} [{min(v):.{nd}f}, {max(v):.{nd}f}]"
        L.append(f"| {LABEL[k]} | {len(runs)} | {agg(lambda e: e['val_excess_ce'])} | "
                 f"{agg(lambda e: e['answer_acc'], 100, 1)} | {agg(lambda e: e['answer_acc_by_m']['5'], 100, 1)} | "
                 f"{agg(lambda e: e['answer_acc_by_m']['6'], 100, 1)} |")
    L.append("")
    # ---- extra analysis (frozen coordinates: same word vs different words)
    L.append("### Совпадают ли замёрзшие координаты от предложения к предложению (Жаккар множеств "
             "|Δx|/RMS<0.01: одно и то же слово в разных предложениях / разные слова / случайный уровень)\n")
    L.append("| run | " + " | ".join(f"слой {l}" for l in range(4)) + " |")
    L.append("|---|---|---|---|---|")
    for n in sorted_runs(R):
        ex = os.path.join(R[n]["path"], "extra.json")
        if not os.path.exists(ex):
            continue
        j = json.load(open(ex))["jaccard"]
        L.append(f"| {n} | " + " | ".join(
            f"p={x['p']:.2f}: {x['same_word']:.2f} / {x['diff_word']:.2f} / {x['chance']:.2f}" for x in j) + " |")
    L.append("")
    # ---- criteria
    A = [R[n]["ev"] for n in R if R[n]["key"] == "A"]
    a_ce, a_ans = np.mean([e["val_excess_ce"] for e in A]), np.mean([e["answer_acc"] for e in A])
    a_ce_max = max(e["val_excess_ce"] for e in A)
    a_ans_min = min(e["answer_acc"] for e in A)
    L.append(f"### Критерии успеха (A: CE−H {a_ce:.4f}, ответ {100 * a_ans:.1f}%; «не хуже A» = в пределах "
             f"разброса сидов A: CE−H ≤ {a_ce_max:.4f} и ответ ≥ {100 * a_ans_min:.1f}%)\n")
    L.append("| run | 1) качество не хуже A | 2) min по слоям: dd<0.01 / |Δx|<1% (> 0.5?) | 3) частная ρ с деревом [ДИ] |")
    L.append("|---|---|---|---|")
    for n in sorted_runs(R):
        e = R[n]["ev"]
        if not e["budget"]:
            continue
        c1 = (e["val_excess_ce"] <= a_ce_max + 1e-9) and (e["answer_acc"] >= a_ans_min - 1e-9)
        mdd, mrel = min(e["frozen_dd_layer"]), min(e["frozen_rel_layer"])
        pt = e["structure"]["partial_tree"]
        c3 = pt["ci"][0] > 0
        L.append(f"| {n} | {'да' if c1 else 'нет'} ({e['val_excess_ce']:.4f}, {100 * e['answer_acc']:.1f}%) | "
                 f"{mdd:.2f} {'да' if mdd > 0.5 else 'нет'} / {mrel:.2f} {'да' if mrel > 0.5 else 'нет'} | "
                 f"{pt['r']:+.3f} [{pt['ci'][0]:+.3f}, {pt['ci'][1]:+.3f}] {'да' if c3 else 'нет'} |")
    L.append("")
    open(os.path.join(out, "tables.md"), "w", encoding="utf-8").write("\n".join(L))


# ------------------------------------------------------------------ figures
def legend_right(ax):
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)


def fig_answer(R, out):
    P = primary(R)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
    for ax, keys, title in ((axes[0], LAMBDA_SET, "Нормировка и λ"), (axes[1], VARIANT_SET, "Варианты (средний λ)")):
        for k in keys:
            runs = [R[n] for n in R if R[n]["key"] == k]
            if not runs:
                continue
            Y = np.array([[r["ev"]["answer_acc_by_m"][str(m)] for m in range(7)] for r in runs])
            lab = LABEL[k] + (f" ({len(runs)} сида)" if len(runs) > 1 else "")
            ax.plot(range(7), Y.mean(0), LS.get(k, "-"), marker="o", color=COL[k], label=lab)
            if len(runs) > 1:
                ax.fill_between(range(7), Y.min(0), Y.max(0), color=COL[k], alpha=0.18, linewidth=0)
        ax.axhline(0.29, color=INK2, lw=1, ls=":")
        ax.set_title(title, loc="left")
        ax.set_xlabel("число перестановок m")
        legend_right(ax)
    axes[0].set_ylabel("точность ответа (пунктир ≈ случайный выбор)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_answer_by_m.png"), dpi=130)
    plt.close(fig)


def fig_spent(R, out):
    P = primary(R)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for k, r in P.items():
        sp = r["ev"]["spent"]
        if r["ev"]["budget"]:
            d = sp["answer_pos"]["final_depth"]
            axes[0].plot(range(7), [d[str(m)] for m in range(7)], "-o", color=COL[k], label=LABEL[k])
            d = sp["event_by_index"]["final_depth"]
            axes[1].plot(range(1, 7), [d[str(m)] for m in range(1, 7)], "-o", color=COL[k], label=LABEL[k])
        d = sp["answer_pos"]["relchg_sum"]
        axes[2].plot(range(7), [d[str(m)] for m in range(7)], LS.get(k, "-"), marker="o", color=COL[k], label=LABEL[k])
    axes[0].set_title("итоговая глубина на позиции ответа", loc="left")
    axes[1].set_title("итоговая глубина на токенах события №e", loc="left")
    axes[2].set_title("Σ отн. изменения состояния @ответ (лог)", loc="left")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("число перестановок m")
    axes[1].set_xlabel("номер события e в истории")
    legend_right(axes[2])
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_spent_by_m.png"), dpi=130)
    plt.close(fig)


def fig_gates(R, out):
    P = {k: r for k, r in primary(R).items() if r["ev"]["budget"]}
    keys = [k for k in ("B_l0", "B_w", "B_m", "B_s", "Bdd", "Bpull", "E", "G") if k in P]
    if not keys:
        return
    L = P[keys[0]]["ev"]["L"]
    for name, arr, edges, xl in (("gate", "gate_hist", np.linspace(0, 1, 51), "σ(гейта)"),
                                 ("dd", "dd_hist", np.linspace(0, 8, 81), "dd")):
        fig, axes = plt.subplots(len(keys), L, figsize=(3.3 * L, 1.7 * len(keys)), sharex=True)
        for i, k in enumerate(keys):
            h = np.load(os.path.join(P[k]["path"], "eval_arrays.npz"))[arr]
            for l in range(L):
                ax = axes[i, l]
                f = h[l] / max(h[l].sum(), 1)
                ax.bar(edges[:-1], f, width=(edges[1] - edges[0]) * 0.9, align="edge", color=COL[k])
                ax.set_yscale("log")
                ax.set_ylim(1e-5, 1.2)
                if name == "dd":
                    ax.set_xlim(0, l + 1.05)
                if i == 0:
                    ax.set_title(f"блок {l}", loc="left")
                if l == 0:
                    ax.set_ylabel(LABEL[k], fontsize=8)
                if i == len(keys) - 1:
                    ax.set_xlabel(xl)
        fig.suptitle(f"Распределение {xl} по блокам (доля, лог-шкала)", x=0.01, ha="left", fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"fig_hist_{name}.png"), dpi=120)
        plt.close(fig)


def fig_frozen(R, out):
    P = primary(R)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for k, r in P.items():
        e = r["ev"]
        Ls = range(len(e["frozen_rel_layer"]))
        axes[0].plot(Ls, e["frozen_rel_layer"], LS.get(k, "-"), marker="o", color=COL[k], label=LABEL[k])
        if e["budget"]:
            axes[1].plot(Ls, e["frozen_dd_layer"], "-o", color=COL[k], label=LABEL[k])
            axes[2].plot(Ls, e["dd_layer_mean"], "-o", color=COL[k], label=LABEL[k])
    axes[0].set_title("доля |Δx|/RMS(x) < 0.01", loc="left")
    axes[1].set_title("доля dd < 0.01", loc="left")
    axes[2].set_title("средний Δd", loc="left")
    for ax in axes[:2]:
        ax.axhline(0.5, color=INK2, lw=1, ls=":")
        ax.set_ylim(-0.02, 1.02)
    for ax in axes:
        ax.set_xlabel("блок")
    legend_right(axes[2])
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_frozen.png"), dpi=130)
    plt.close(fig)


def fig_decomp(R, out, key="B_l0"):
    P = primary(R)
    keys = [k for k in (key, "B_m", "A") if k in P]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.5 * len(keys), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    cols = {"static": "#2a78d6", "lexical": "#eb6834", "context": "#1baf7a"}
    for ax, k in zip(axes, keys):
        dec = P[k]["ev"]["decomp_rel"]
        x = np.arange(len(dec))
        bottom = np.zeros(len(dec))
        for part, lab in (("static", "статич."), ("lexical", "лексич."), ("context", "контекст")):
            v = np.array([dd[part] for dd in dec])
            ax.bar(x, v, bottom=bottom, color=cols[part], width=0.7, label=lab, edgecolor=SURF, linewidth=2)
            bottom += v
        for i, dd in enumerate(dec):
            ax.text(i, 1.02, f"{dd['p']:.2f}", ha="center", fontsize=8, color=INK2)
        ax.set_title(f"{LABEL[k]}", loc="left")
        ax.set_xticks(x, [f"блок {i}" for i in x])
        ax.set_ylim(0, 1.1)
    axes[0].set_ylabel("доля дисперсии маски «замёрзла»\n(над столбцом — доля замёрзших)")
    legend_right(axes[-1])
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_frozen_decomp.png"), dpi=130)
    plt.close(fig)


def fig_lambda(R, out):
    pts = []
    for n, r in R.items():
        if r["key"] in ("B_l0", "B_w", "B_m", "B_s") or n.startswith("B_lam"):
            if n.endswith("_s0"):
                pts.append((r["log"]["result"]["lam"], r["ev"]))
    if not pts:
        return
    seen, P2 = set(), []
    for lam, e in sorted(pts, key=lambda t: t[0]):
        if lam not in seen:
            seen.add(lam)
            P2.append((lam, e))
    lam = [max(p[0], 1e-5) for p in P2]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    axes[0].plot(lam, [p[1]["final_depth_mean"] for p in P2], "-o", color=COL["B_m"])
    axes[0].set_title("итоговая глубина (из 4)", loc="left")
    axes[1].plot(lam, [min(p[1]["frozen_dd_layer"]) for p in P2], "-o", color=COL["B_m"], label="dd < 0.01")
    axes[1].plot(lam, [min(p[1]["frozen_rel_layer"]) for p in P2], "--o", color=COL["Bdd"], label="|Δx|/RMS < 0.01")
    axes[1].axhline(0.5, color=INK2, lw=1, ls=":")
    axes[1].set_title("min по слоям доли замёрзших", loc="left")
    axes[1].legend(fontsize=8)
    axes[2].plot(lam, [p[1]["val_excess_ce"] for p in P2], "-o", color=COL["B_m"])
    a = [r["ev"]["val_excess_ce"] for r in R.values() if r["key"] == "A"]
    if a:
        axes[2].axhline(np.mean(a), color=COL["A"], lw=1.5, ls="--", label="A")
        axes[2].legend(fontsize=8)
    axes[2].set_title("val CE − H", loc="left")
    axes[3].plot(lam, [p[1]["answer_acc_by_m"]["6"] for p in P2], "-o", color=COL["B_m"], label="m=6")
    axes[3].plot(lam, [p[1]["answer_acc_by_m"]["5"] for p in P2], "--o", color=COL["B_w"], label="m=5")
    axes[3].set_title("точность ответа", loc="left")
    axes[3].legend(fontsize=8)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("λ (λ=0 показан в 1e-5)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_lambda.png"), dpi=130)
    plt.close(fig)


def fig_structure(R, out):
    P = {k: r for k, r in primary(R).items() if r["ev"]["budget"]}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for k, r in P.items():
        s = r["ev"]["structure"]
        bt = s["by_tree_depth"]
        xs = sorted(int(x) for x in bt)
        axes[0].plot(xs, [bt[str(x)] for x in xs], "-o", color=COL[k], label=LABEL[k])
        axes[1].plot(range(len(s["partial_tree_dd_by_layer"])), s["partial_tree_dd_by_layer"], "-o",
                     color=COL[k], label=LABEL[k])
    axes[0].set_title("итоговая глубина vs глубина узла в дереве (клауза)", loc="left")
    axes[0].set_xlabel("глубина в дереве")
    axes[1].set_title("частная ρ(Δd блока, глубина в дереве)", loc="left")
    axes[1].set_xlabel("блок")
    axes[1].axhline(0, color=INK2, lw=1)
    legend_right(axes[1])
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_structure.png"), dpi=130)
    plt.close(fig)


def fig_curves(R, out):
    P = primary(R)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for k, r in P.items():
        vl = r["log"]["val_log"]
        axes[0].plot([v["step"] for v in vl], [v["copy"] for v in vl], LS.get(k, "-"), color=COL[k], label=LABEL[k])
        axes[1].plot([v["step"] for v in vl], [v.get("ans_m56", np.nan) for v in vl], LS.get(k, "-"),
                     color=COL[k], label=LABEL[k])
    axes[0].set_title("точность ответа (все m) по ходу обучения", loc="left")
    axes[1].set_title("точность ответа при m=5–6", loc="left")
    for ax in axes:
        ax.set_xlabel("шаг")
    legend_right(axes[1])
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_curves.png"), dpi=130)
    plt.close(fig)


def fig_flow(R, out):
    P = primary(R)
    fig, ax = plt.subplots(figsize=(8, 4))
    for k, r in P.items():
        e = r["ev"]
        ax.plot(range(len(e["block_removal_dce"])), e["block_removal_dce"], LS.get(k, "-"), marker="o",
                color=COL[k], label=LABEL[k])
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_title("рост val CE при удалении одного блока", loc="left")
    ax.set_xlabel("удалённый блок")
    legend_right(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_flow.png"), dpi=130)
    plt.close(fig)


def fig_legacy(out, pilot="pilot"):
    """Stage-1 finding: in the unnormalised formulation, larger λ shrinks the stream instead of skipping."""
    pts = []
    for lam in ("0.001", "0.01", "0.1", "1.0"):
        p = os.path.join(pilot, f"lam_{lam}", "eval.json")
        if os.path.exists(p):
            e = json.load(open(p))
            pts.append((float(lam), e))
    if not pts:
        return
    lam = [p[0] for p in pts]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    axes[0].plot(lam, [p[1]["final_depth_mean"] for p in pts], "-o", color=COL["B_m"])
    axes[0].set_title("итоговая глубина (из 8)", loc="left")
    axes[0].set_yscale("log")
    axes[1].plot(lam, [p[1]["test_excess_ce"] for p in pts], "-o", color=COL["B_m"])
    axes[1].set_title("test CE − H", loc="left")
    axes[2].plot(lam, [p[1]["dce_dd_zeroed"] for p in pts], "-o", color=COL["Bdd"], label="dd<0.01")
    axes[2].plot(lam, [p[1]["ce_dd_zeroed_0.1"] for p in pts], "--o", color=COL["G"], label="dd<0.1")
    axes[2].set_yscale("symlog", linthresh=1e-3)
    axes[2].set_title("рост CE, если обнулить «малые» Δd", loc="left")
    axes[2].legend(fontsize=8)
    axes[3].plot(lam, [p[1]["norm_layer"][-1] for p in pts], "-o", color=COL["A"])
    axes[3].set_yscale("log")
    axes[3].set_title("‖x‖ на выходе последнего блока", loc="left")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("λ")
    fig.suptitle("Этап 1 (без нормировки): обход бюджета через сжатие потока", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_legacy_loophole.png"), dpi=130)
    plt.close(fig)


def draw_tree(ax, spans, n):
    lab_col = {"NP": "#2a78d6", "VP": "#eb6834", "PP": "#1baf7a", "RC": "#e87ba4", "CP": "#6250d6"}
    spans = [s for s in spans if s[2] in lab_col]
    spans = sorted(spans, key=lambda s: (s[1] - s[0]), reverse=True)
    level = []
    for i, (a, b, lab) in enumerate(spans):
        lv = 0
        for j in range(i):
            a2, b2, _ = spans[j]
            if a2 <= a and b <= b2:
                lv = max(lv, level[j] + 1)
        level.append(lv)
    for (a, b, lab), lv in zip(spans, level):
        ax.plot([a - 0.4, a - 0.4, b + 0.4, b + 0.4], [-lv - 0.6, -lv, -lv, -lv - 0.6], color=lab_col[lab], lw=1.1)
        if lab in ("PP", "RC", "CP"):
            ax.text((a + b) / 2, -lv + 0.08, lab, ha="center", va="bottom", fontsize=6.5, color=lab_col[lab])
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-(max(level) if level else 0) - 1, 1)
    ax.axis("off")


def fig_trajectories(R, out, data="data4", n_sent=3):
    from .evaluate import load_run
    P = {k: r for k, r in primary(R).items() if r["ev"]["budget"] and k in ("B_l0", "B_m", "E", "G")}
    if not P:
        return
    te = np.load(os.path.join(data, "test.npz"))
    trees = pickle.load(open(os.path.join(data, "test_trees.pkl"), "rb"))
    meta = json.load(open(os.path.join(data, "meta.json")))
    tok, sid = te["tok"], te["sid"]
    cands = []
    for s in range(400):
        rc = np.argwhere(sid == s)
        if not len(rc):
            continue
        r, c = rc[0]
        n = int((sid[r] == s).sum())
        emb = te["emb_depth"][r][sid[r] == s]
        nsw = te["nsw"][r, c]
        if 30 <= n <= 56 and emb.max() >= 2 and nsw >= 2:
            cands.append((s, r, c, n))
        if len(cands) >= n_sent:
            break
    for k, r in P.items():
        model, cfg = load_run(r["path"])
        fig = plt.figure(figsize=(17, 4.0 * n_sent))
        gs = fig.add_gridspec(2 * n_sent, 1, height_ratios=[0.9, 1.4] * n_sent, hspace=0.45)
        for i, (s, row, col, n) in enumerate(cands):
            idx = torch.tensor(tok[row:row + 1].astype(np.int64), device="cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, aux = model(idx, collect=True)
            dep = torch.stack(aux["depths"])[:, 0, col:col + n].float().cpu().numpy()  # L, n, G
            words = [meta["vocab"][t] for t in tok[row, col:col + n]]
            draw_tree(fig.add_subplot(gs[2 * i]), trees[s][0], n)
            ax = fig.add_subplot(gs[2 * i + 1])
            G = dep.shape[-1]
            if 1 < G <= 8:
                M = dep[-1].T
                ylabels = [f"группа {h}" for h in range(G)]
                origin = "upper"
            else:
                M = dep.mean(-1)
                ylabels = [f"после блока {l}" for l in range(M.shape[0])]
                origin = "lower"
            im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=cfg.n_layers, origin=origin)
            ax.set_yticks(range(M.shape[0]), ylabels, fontsize=8)
            ax.set_xticks(range(n), words, rotation=65, ha="right", fontsize=8)
            aq = np.where(te["ansq"][row, col:col + n] == 1)[0]
            for j in aq:
                ax.get_xticklabels()[j].set_color("#e34948")
                ax.get_xticklabels()[j].set_fontweight("bold")
            ax.grid(False)
            ax.set_title(f"m = {te['nsw'][row, col]}, k = {te['nk'][row, col]}  (красным — позиция ответа)",
                         loc="left", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01)
        fig.suptitle(f"Траектории глубины — {LABEL[k]}", x=0.01, ha="left", fontweight="bold")
        fig.savefig(os.path.join(out, f"fig_traj_{k}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)
        del model
        torch.cuda.empty_cache()


def main(runs="runs4", out="report"):
    os.makedirs(out, exist_ok=True)
    R = load(runs)
    extra = "runs4p"
    if os.path.isdir(extra):
        for n, r in load(extra).items():
            R.setdefault(n, r)
    print("runs:", list(R))
    tables(R, out)
    for f in (fig_answer, fig_spent, fig_gates, fig_frozen, fig_decomp, fig_lambda, fig_structure, fig_curves,
              fig_flow, fig_trajectories):
        try:
            f(R, out)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            print(f"{f.__name__} failed: {ex!r}")
    fig_legacy(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
