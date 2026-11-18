"""Round 6, part 2 (TinyStories): tables report/r6ts_tables.md, generations report/r6ts_generations.md,
figures report/r6ts_val.png, report/r6ts_adapt.png, report/r6ts_blocks.png.
python -m dd.report_r6ts [runs_ts] [report]"""
from __future__ import annotations

import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .report import INK2  # noqa: F401  (shared rcParams)

TYPES = ["sentence_start", "name", "rare", "punct", "function", "other"]
TYPE_RU = dict(sentence_start="начало предложения", name="имя", rare="редкое слово", punct="пунктуация",
               function="служебное слово", other="прочее")
COL = {"A": "#7a7466", "Q0": "#a39e93", "QG25": "#2a78d6", "QGsplit25": "#1baf7a", "QGsplitFFN25": "#eb6834",
       "QGsplitHead25": "#6250d6"}
SHORT = {"A": "A", "Q0": "Q0", "QG25": "QG25", "QGsplit25": "QG-split25", "QGsplitFFN25": "QG-splitFFN25",
         "QGsplitHead25": "QG-splitHead25"}
ORDER = ["A", "Q0", "QG25", "QGsplit25", "QGsplitFFN25", "QGsplitHead25"]


def jl(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def load(base):
    R = []
    for d in os.listdir(base):
        p = os.path.join(base, d)
        if not os.path.isdir(p) or not os.path.exists(os.path.join(p, "log.json")):
            continue
        k, s = d.rsplit("_s", 1)
        if k not in ORDER:
            continue
        R.append(dict(name=d, key=k, seed=int(s), log=jl(os.path.join(p, "log.json")), ev=jl(os.path.join(p, "ts_eval.json")),
                      gen=jl(os.path.join(p, "generations.json"))))
    R.sort(key=lambda r: (ORDER.index(r["key"]), r["seed"]))
    return R


def lab(r):
    return f"{SHORT[r['key']]} s{r['seed']}"


def fl(xs, fmt="{:.2f}"):
    return " ".join(fmt.format(x) for x in xs)


def tables(R, base, out, speed):
    L = []
    grid = jl(os.path.join(base, "grid.json"))
    A = next((r for r in R if r["key"] == "A" and r["ev"]), None)
    # ---------------- setup
    if R:
        res = R[0]["log"]["result"]
        L += ["### Постановка\n",
              f"- Токенов на прогон: {res['tokens'] / 1e6:.0f}M ({res['steps']} шагов × {res['args']['bs']} × "
              f"{res['args']['ctx']}).",
              f"- Параметров: " + ", ".join(f"{lab(r)} {r['log']['result']['n_params'] / 1e6:.2f}M" for r in R) + "."]
    if grid:
        c = grid["calib"]
        if grid.get("scheme") == "profile":
            es = grid.get("emb_scale", 1.0)
            es_txt = f", эмбеддинг ×{es:.3g}" if es != 1.0 else ""
            L.append(f"- Сетка: свой шаг на каждое состояние, Δ_l = RMS состояния A / 3.2, уровни −{grid['K']}…{grid['K']}"
                     f"{es_txt}: " + ", ".join(f"{'emb' if i == 0 else f'b{i - 1}'} {v:g}"
                                               for i, v in enumerate(grid["delta_layers"])) + ".")
        else:
            L.append(f"- Сетка из состояния A: средний RMS внутренних состояний {c['inner_rms_mean']:.3f}, max q99 "
                     f"{c['inner_q99_max']:.3f} → Δ = {grid['delta']}, уровни −{grid['K']}…{grid['K']}.")
        L.append("- RMS / q99 / q99.9 |x| по состояниям A: " + "; ".join(
            f"{'emb' if i == 0 else f'b{i - 1}'} {a['rms']:.2f}/{a['q99']:.2f}/{a['q999']:.2f}"
            for i, a in enumerate(c["per_state"])) + ".")
    L.append("")
    # ---------------- quality
    L += ["### Качество\n",
          "| прогон | val CE | perplexity | ΔCE против A | ρ(потеря, потеря A) | время обучения, мин |",
          "|---|---|---|---|---|---|"]
    for r in R:
        e = r["ev"]
        if not e:
            continue
        d = e["val_ce"] - A["ev"]["val_ce"] if A else float("nan")
        L.append(f"| {lab(r)} | {e['val_ce']:.4f} | {e['val_ppl']:.3f} | {d:+.4f} | "
                 f"{e.get('spearman_own_vs_A_loss', float('nan')):.3f} | {r['log']['result']['train_time'] / 60:.0f} |")
    L.append("")
    # ---------------- per-type loss
    if A:
        L += ["### Типы предсказываемых токенов: доля и потеря\n",
              "| тип | доля | " + " | ".join(f"CE {lab(r)}" for r in R if r["ev"]) + " |",
              "|" + "---|" * (2 + sum(1 for r in R if r["ev"]))]
        for t in TYPES:
            L.append(f"| {TYPE_RU[t]} | {100 * A['ev']['type_share'][t]:.1f}% | " +
                     " | ".join(f"{r['ev']['type_loss'][t]:.3f}" for r in R if r["ev"]) + " |")
        L.append("")
    # ---------------- levels
    Lq = [r for r in R if r["ev"] and "levels" in r["ev"]]
    if Lq:
        L += ["### Использование сетки (состояния: эмбеддинг, после блоков 0…7)\n",
              "| прогон | доля на краях ±K | доля нуля | уровней на 99% |", "|---|---|---|---|"]
        for r in Lq:
            lv = r["ev"]["levels"]
            L.append(f"| {lab(r)} | {fl([100 * x for x in lv['edge_share']], '{:.2f}')} % | "
                     f"{fl([100 * x for x in lv['zero_share']], '{:.0f}')} % | {fl(lv['levels_99'], '{:d}')} |")
        L.append("")
    # ---------------- QG checks
    Q = [r for r in R if r["ev"] and "qg" in r["ev"]]
    if Q:
        L += ["### QG: проверки\n",
              "| прогон | оплачено по блокам | впустую / открытых | токенов без открытых групп по блокам | "
              "мёртвых / всегда открытых групп (из 64) | max\\|Δx\\| закрытой | train/eval Δлогит | μ max / в конце |",
              "|---|---|---|---|---|---|---|---|"]
        for r in Q:
            g = r["ev"]["qg"]
            mus = [t["mu"][0] for t in r["log"]["train_log"] if "mu" in t]
            dead = int(round(8 * sum(g["never_open_layer"])))
            alw = int(round(8 * sum(g["always_open_layer"])))
            L.append(f"| {lab(r)} | {fl(g['paid_layer'])} | {max(g['wasted_of_paid_layer']):.3f} (max по блокам) | "
                     f"{fl(g['token_none_open_layer'])} | {dead} / {alw} | {g['max_abs_change_closed_group']:.0g} | "
                     f"{g['train_eval_max_abs_logit_diff']:.0g} | {max(mus):.3f} / {r['ev']['mu_final'][0]:.4f} |")
        L.append("")
        # ---------------- realisable savings (round-5 accounting, dd.ts_flops)
        from .ts_flops import savings
        L += ["### Реализуемая экономия (по решениям гейта на анализируемых токенах)\n",
              "| прогон | открыто, доля | токенов без открытых групп (среднее по блокам) | экономия FFN | экономия блока | "
              "экономия блока по блокам | средний контекст внимания |", "|---|---|---|---|---|---|---|"]
        for r in Q:
            sv = savings(os.path.join(base, r["name"]))
            r["sav"] = sv
            L.append(f"| {lab(r)} | {sv['open_share']:.3f} | {sv['token_none_open']:.3f} | {100 * sv['realisable_ffn']:.1f}% | "
                     f"{100 * sv['realisable_block']:.1f}% | {fl([100 * x for x in sv['realisable_block_layer']], '{:.0f}')} % | "
                     f"{sv['ctx']:.0f} |")
        L.append("")
        # ---------------- adaptivity
        L += ["### Адаптивность: открытые группы против трудности токена\n",
              "Позиция t: гейт решает по состоянию токена t, трудность — потеря предсказания токена t+1. "
              "Открытых групп — сумма по 8 блокам × 8 группам (из 64).\n",
              "| прогон | открыто в среднем | ρ(открыто, своя потеря) | ρ(открыто, потеря A) | AUROC «открыто → свой top-10%» | "
              "AUROC «→ top-10% A» | открыто по квинтилям потери A (лёгкие → трудные) |",
              "|---|---|---|---|---|---|---|"]
        for r in Q:
            g = r["ev"]["qg"]
            L.append(f"| {lab(r)} | {g['open_total_mean']:.1f} | {g['spearman_open_vs_own_loss']:.3f} | "
                     f"{g.get('spearman_open_vs_A_loss', float('nan')):.3f} | {g['auroc_open_top10_own']:.3f} | "
                     f"{g.get('auroc_open_top10_A', float('nan')):.3f} | {fl(g.get('open_by_A_loss_quintile', []), '{:.1f}')} |")
        L.append("")
        L += ["### Открытые группы по типу токена (из 64)\n",
              "| тип | " + " | ".join(f"{lab(r)}: предсказываемый / текущий | ρ внутри типа" for r in Q) + " |",
              "|" + "---|" * (1 + 2 * len(Q))]
        for t in TYPES:
            L.append(f"| {TYPE_RU[t]} | " + " | ".join(
                f"{r['ev']['qg']['by_type'][t]['open_total']:.1f} / {r['ev']['qg']['by_type_current'][t]:.1f} | "
                f"{r['ev']['qg'].get('spearman_open_vs_A_loss_within_type', {}).get(t, float('nan')):.3f}" for r in Q) + " |")
        L.append("")
    # ---------------- speed
    if speed:
        L += ["### Скорость генерации (batch 1, CUDA graph, медиана по шагам, мкс)\n",
              "| модель | путь | шаг | шаг: ускорение к A | FFN 8 блоков | FFN: ускорение к A | доля открытых | "
              "пропускаемая доля чтений FFN | Δлогит к полному проходу | совпадение argmax |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        a = next((s for s in speed if not s["qg"]), None)
        aS = a["step_A"]["median_us"] if a else float("nan")
        aF = a["ffn_A"]["median_us"] if a and "ffn_A" in a else float("nan")
        for s in speed:
            nm = os.path.basename(s["run"])
            paths = ["A"] if not s["qg"] else ["dense", "tri_all", "tri_skip"]
            for p in paths:
                st, ff, ex = s[f"step_{p}"]["median_us"], s.get(f"ffn_{p}", {}).get("median_us", float("nan")), s[f"exact_{p}"]
                L.append(f"| {nm} | {p} | {st:.1f} | ×{aS / st:.3f} | {ff:.1f} | ×{aF / ff:.2f} | "
                         f"{100 * s.get('open_share', 1.0):.1f}% | {100 * s.get('ffn_skippable', 0.0):.1f}% | "
                         f"{ex['max_abs_logit_diff']:.1e} | {100 * ex.get('top1_agreement', float('nan')):.1f}% |")
        L.append("")
        for s in speed:
            if s["qg"]:
                fa, fs = s["ffn_tri_all"]["median_us"], s["ffn_tri_skip"]["median_us"]
                L.append(f"- {os.path.basename(s['run'])}: пропуск ускоряет FFN в ×{fa / fs:.2f} (те же ядра без "
                         f"пропуска / с пропуском); по доле пропускаемых чтений ожидалось ×{1 / (1 - s['ffn_skippable']):.2f}. "
                         f"FFN — {100 * s['ffn_share_of_step_bytes']:.0f}% байт весов шага.")
        L.append("")
    open(os.path.join(out, "r6ts_tables.md"), "w", encoding="utf-8").write("\n".join(L))


def generations(R, out):
    G = [r for r in R if r["gen"]]
    if not G:
        return
    L = ["# TinyStories: генерации с одинаковых затравок\n",
         "Жадно и с температурой 0.8 (seed = номер затравки), до 120 токенов или до конца истории.\n"]
    for i, g0 in enumerate(G[0]["gen"]):
        L.append(f"## Затравка {i + 1}: «{g0['prompt']}»\n")
        for mode, title in (("greedy", "жадно"), ("temp08", "T = 0.8")):
            L.append(f"**{title}**\n")
            for r in G:
                txt = r["gen"][i][mode].replace("\n", " ⏎ ")
                L.append(f"- *{lab(r)}*: {txt}")
            L.append("")
    open(os.path.join(out, "r6ts_generations.md"), "w", encoding="utf-8").write("\n".join(L))


def figs(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for r in R:
        vl = r["log"]["val_log"]
        st = [v["step"] / 1000 for v in vl]
        ce = [v["val_ce"] for v in vl]
        for ax, lo in zip(axes, (0, len(st) // 2)):
            ax.plot(st[lo:], ce[lo:], color=COL[r["key"]], lw=1.4, ls="-" if r["seed"] == 0 else "--", label=lab(r))
    axes[0].set_title("val CE по ходу обучения", loc="left")
    axes[1].set_title("вторая половина обучения", loc="left")
    for ax in axes:
        ax.set_xlabel("шаг, тыс.")
    axes[0].set_ylim(top=3.0)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "r6ts_val.png"), dpi=130)
    plt.close(fig)
    Q = [r for r in R if r["ev"] and "qg" in r["ev"]]
    if not Q:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for r in Q:
        g = r["ev"]["qg"]
        if "open_by_A_loss_quintile" in g:
            axes[0].plot(range(1, 6), g["open_by_A_loss_quintile"], color=COL[r["key"]], marker="o", ms=5, lw=1.4,
                         ls="-" if r["seed"] == 0 else "--", label=lab(r))
    axes[0].set_xticks(range(1, 6))
    axes[0].set_xlabel("квинтиль потери A на токене (1 = самые лёгкие)")
    axes[0].set_title("открытых групп из 64", loc="left")
    axes[0].legend(fontsize=8)
    w = 0.8 / len(Q)
    for i, r in enumerate(Q):
        axes[1].bar(np.arange(len(TYPES)) + (i - (len(Q) - 1) / 2) * w,
                    [r["ev"]["qg"]["by_type"][t]["open_total"] for t in TYPES], width=w * 0.9, color=COL[r["key"]],
                    alpha=1.0 if r["seed"] == 0 else 0.55, label=lab(r))
    axes[1].set_xticks(range(len(TYPES)))
    axes[1].set_xticklabels([TYPE_RU[t] for t in TYPES], fontsize=8)
    axes[1].set_title("открытых групп из 64 по типу предсказываемого токена", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "r6ts_adapt.png"), dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.2))
    for r in Q:
        ax.plot(range(8), r["ev"]["qg"]["paid_layer"], color=COL[r["key"]], marker="o", ms=5, lw=1.4,
                ls="-" if r["seed"] == 0 else "--", label=lab(r))
    ax.axhline(0.25, color=INK2, lw=1, ls=":")
    ax.set_xlabel("блок")
    ax.set_title("доля открытых групп по блокам (цель в среднем 0.25)", loc="left")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "r6ts_blocks.png"), dpi=130)
    plt.close(fig)


def main(base="runs_ts", out="report"):
    R = load(base)
    speed = jl(os.path.join(out, "r6_speed.json"))
    tables(R, base, out, speed)
    generations(R, out)
    figs(R, out)
    print("written", os.path.join(out, "r6ts_tables.md"))


if __name__ == "__main__":
    main(*sys.argv[1:])
