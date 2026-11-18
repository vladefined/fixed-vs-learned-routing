"""Round-3 tables (report/r3_tables.md) and figures (report/r3_*.png).  python -m dd.report5 [runs5]"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .report import SURF, INK, INK2, GRID  # noqa: F401  (also applies the shared rcParams)

COL = {"A": "#52514e", "A_norm": "#a3a29c", "A_groupnorm": "#9ec5f4", "E": "#2a78d6", "G": "#6250d6",
       "H": "#eb6834", "I": "#1baf7a", "J": "#e87ba4"}
LS = {"A": "-", "A_norm": "--", "A_groupnorm": "--"}
ORDER = ["A", "A_norm", "A_groupnorm", "E", "G", "H", "I", "J"]
LABEL = {"A": "A", "A_norm": "A_norm (на токен)", "A_groupnorm": "A_groupnorm (по группам)",
         "E": "E бюджет по группам", "G": "G = E + внимание", "H": "H общий бюджет", "I": "I отн. изменение",
         "J": "J = I + шум"}
BUDGET = ["E", "G", "H", "I", "J"]


def load(runs):
    R = {}
    for d in sorted(os.listdir(runs)):
        p = os.path.join(runs, d)
        if os.path.exists(os.path.join(p, "eval5.json")):
            k = d.rsplit("_s", 1)[0]
            R.setdefault(k, []).append(dict(name=d, ev=json.load(open(os.path.join(p, "eval5.json"))),
                                            log=json.load(open(os.path.join(p, "log.json"))), path=p))
    return {k: R[k] for k in ORDER if k in R}


def jump_step(log, thr=0.8):
    for v in log["val_log"]:
        if v.get("ans_m46", 0) > thr:
            return v["step"]
    return None


def ms(vals, f="{:.4f}", scale=1.0):
    v = [scale * x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return "—"
    if len(v) == 1:
        return f.format(v[0])
    return f"{f.format(np.mean(v))} [{f.format(min(v))}, {f.format(max(v))}]"


def tables(R, out):
    L = []
    L.append("### Качество (среднее [мин, макс] по сидам)\n")
    L.append("| вариант | сидов | val CE−H | ответ, % | m=0 | 1 | 2 | 3 | 4 | 5 | 6 | шаг скачка (m≥4 > 80%) | согл., % |")
    L.append("|" + "---|" * 13)
    for k, runs in R.items():
        ev = [r["ev"] for r in runs]
        jumps = [jump_step(r["log"]) for r in runs]
        L.append(f"| {LABEL[k]} | {len(runs)} | {ms([e['val_excess_ce'] for e in ev])} | "
                 f"{ms([e['answer_acc'] for e in ev], '{:.1f}', 100)} | " +
                 " | ".join(ms([e['answer_acc_by_m'].get(str(m)) for e in ev], '{:.1f}', 100) for m in range(7)) +
                 f" | {', '.join(str(j) if j else 'нет' for j in jumps)} | {ms([e['agree_acc'] for e in ev], '{:.1f}', 100)} |")
    L.append("")
    L.append("### Проверки на обходы (по слоям 0→3; среднее по сидам)\n")
    L.append("| вариант | ‖x‖ по слоям (эмб.→выход) | max|x_i|/медиана (по токену) | max RMS_i / медиана RMS_i | max |mean_i|/RMS_i | "
             "стоят: |Δx|/RMS_гр<0.01 | стоят: dd<0.01 | dd<0.01, но |Δx|≥1% | ΔCE обнулить стоящие (отн. / dd) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k, runs in R.items():
        ev = [r["ev"] for r in runs]
        avg = lambda key: np.mean([e[key] for e in ev], 0)
        row = (f"| {LABEL[k]} | {' '.join(f'{v:.1f}' for v in avg('norm_layer'))} | "
               f"{' '.join(f'{v:.1f}' for v in avg('maxabs_over_median_layer'))} | "
               f"{' '.join(f'{v:.1f}' for v in avg('coord_rms_max_over_median_layer'))} | "
               f"{' '.join(f'{v:.2f}' for v in avg('coord_dc_fraction_layer'))} | "
               f"{' '.join(f'{v:.2f}' for v in avg('standing_rel_layer'))} | ")
        if ev[0]["budget"]:
            ddo = np.mean([[a["dd_only"] for a in e["standing_agreement_layer"]] for e in ev], 0)
            row += (f"{' '.join(f'{v:.2f}' for v in avg('standing_dd_layer'))} | {' '.join(f'{v:.2f}' for v in ddo)} | "
                    f"{np.mean([e['dce_zero_rel'] for e in ev]):+.4f} / {np.mean([e['dce_zero_dd'] for e in ev]):+.4f} |")
        else:
            row += "— | — | — |"
        L.append(row)
    L.append("")
    L.append("### Потраченная глубина на позиции ответа по m и AUROC «глубина → ошибка»\n")
    L.append("| вариант | мера | m=0 | 1 | 2 | 3 | 4 | 5 | 6 | AUROC все | AUROC по m (1…6) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, runs in R.items():
        ev = [r["ev"] for r in runs]
        for meas in (["final_depth", "eff_update"] if ev[0]["budget"] else []) + ["relchg_sum"]:
            vals = [ms([e["spent_by_m"][meas].get(str(m)) for e in ev], "{:.2f}") for m in range(7)]
            au = ms([e["auroc_wrong"][meas]["all"] for e in ev], "{:.2f}")
            aum = " ".join(f"{np.nanmean([e['auroc_wrong'][meas]['by_m'][str(m)] for e in ev]):.2f}" for m in range(1, 7))
            L.append(f"| {LABEL[k]} | {meas} | " + " | ".join(vals) + f" | {au} | {aum} |")
    L.append("")
    L.append("### Адаптивные вычисления\n")
    L.append("| вариант | итог. глубина | группы исчерпали бюджет к блоку 0/1/2/3 | последний блок, где группа ещё меняется "
             "(−1 никогда / 0 / 1 / 2 / 3) | посл. блок @ответ по m (0…6) | доля стоящих (токен,блок,группа) | "
             "экономия FFN / attention / блока |")
    L.append("|---|---|---|---|---|---|---|")
    for k, runs in R.items():
        ev = [r["ev"] for r in runs]
        if not ev[0]["budget"]:
            continue
        ex = " / ".join(f"{v:.2f}" for v in np.mean([e["exhausted_by_block"] for e in ev], 0)) if "exhausted_by_block" in ev[0] else "—"
        lm = " / ".join(f"{v:.2f}" for v in np.mean([e["last_move_hist"] for e in ev], 0))
        lmm = " ".join(f"{np.mean([e['last_move_by_m_answer'][str(m)] for e in ev]):.2f}" for m in range(7))
        sk = [e["skip_estimate"] for e in ev]
        L.append(f"| {LABEL[k]} | {ms([e['final_depth_mean'] for e in ev], '{:.2f}')} | {ex} | {lm} | {lmm} | "
                 f"{np.mean([s['frac_standing_rel'] for s in sk]):.2f} | "
                 f"{100*np.mean([s['ffn_saving'] for s in sk]):.0f}% / {100*np.mean([s['attn_saving'] for s in sk]):.0f}% / "
                 f"{100*np.mean([s['block_saving'] for s in sk]):.0f}% |")
    L.append("")
    L.append("### Предсказание гейта до FFN (точность «стоит / двигается», подстановка: точность ответа и val CE)\n")
    L.append("| вариант | точность: таблица / голова | полнота «стоит»: таблица / голова | ответ: настоящие гейты | "
             "пропуск по таблице | пропуск по голове | гейт = таблица | гейт = голова | Δ val CE (пропуск: таблица / голова) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k, runs in R.items():
        ev = [r["ev"] for r in runs if "gate_prediction" in r["ev"]]
        if not ev:
            continue
        gp = [e["gate_prediction"] for e in ev]

        def f(*keys):
            vals = []
            for g in gp:
                for kk in keys:
                    g = g[kk]
                vals.append(g)
            return float(np.mean(vals))
        rec_t = np.mean([np.mean(g["table"]["recall_standing"]) for g in gp])
        rec_h = np.mean([np.mean(g["head"]["recall_standing"]) for g in gp])
        base_ce = f("base", "val_ce")
        L.append(f"| {LABEL[k]} | {f('table', 'acc'):.3f} / {f('head', 'acc'):.3f} | {rec_t:.2f} / {rec_h:.2f} | "
                 f"{100 * f('base', 'acc'):.1f} | {100 * f('table', 'skip', 'acc'):.1f} | "
                 f"{100 * f('head', 'skip', 'acc'):.1f} | {100 * f('table', 'full', 'acc'):.1f} | "
                 f"{100 * f('head', 'full', 'acc'):.1f} | {f('table', 'skip', 'val_ce') - base_ce:+.4f} / "
                 f"{f('head', 'skip', 'val_ce') - base_ce:+.4f} |")
    L.append("")
    L.append("### Представления: зонд «какой токен» (неглубокие 4 / глубокие 4 группы) и совпадение стоящих групп\n")
    L.append("| вариант | весь вектор | неглубокие / глубокие | Жаккар стоящих групп: то же слово / разные слова / случайно (слой 1) |")
    L.append("|---|---|---|---|")
    for k, runs in R.items():
        ev = [r["ev"] for r in runs]
        sd = " / ".join(f"{v:.2f}" for v in np.mean([e["probe_shallow4_deep4"] for e in ev], 0)) if "probe_shallow4_deep4" in ev[0] else "—"
        j = [e["jaccard_standing"][1] for e in ev]
        L.append(f"| {LABEL[k]} | {np.mean([e['probe_full'] for e in ev]):.3f} | {sd} | "
                 f"{np.mean([x['same_word'] for x in j]):.2f} / {np.mean([x['diff_word'] for x in j]):.2f} / "
                 f"{np.mean([x['chance'] for x in j]):.2f} (p={np.mean([x['p'] for x in j]):.2f}) |")
    L.append("")
    open(os.path.join(out, "r3_tables.md"), "w", encoding="utf-8").write("\n".join(L))


def band(ax, k, runs, getter, xs):
    Y = np.array([[getter(r["ev"], x) for x in xs] for r in runs], float)
    ax.plot(xs, np.nanmean(Y, 0), LS.get(k, "-"), marker="o", color=COL[k], label=f"{LABEL[k]} ({len(runs)})")
    if len(runs) > 1:
        ax.fill_between(xs, np.nanmin(Y, 0), np.nanmax(Y, 0), color=COL[k], alpha=0.15, linewidth=0)


def figures(R, out):
    xs = list(range(7))
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.2))
    for k, runs in R.items():
        band(axes[0], k, runs, lambda e, m: e["answer_acc_by_m"].get(str(m), np.nan), xs)
        if runs[0]["ev"]["budget"]:
            band(axes[1], k, runs, lambda e, m: e["spent_by_m"]["final_depth"].get(str(m), np.nan), xs)
            band(axes[2], k, runs, lambda e, m: e["auroc_wrong"]["final_depth"]["by_m"].get(str(m), np.nan), xs)
    axes[0].set_title("точность ответа по m", loc="left")
    axes[1].set_title("итоговая глубина на позиции ответа", loc="left")
    axes[2].set_title("AUROC: глубина → неверный ответ", loc="left")
    axes[2].axhline(0.5, color=INK2, lw=1, ls=":")
    for ax in axes:
        ax.set_xlabel("перестановок до вопроса, m")
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "r3_answer_depth_auroc.png"), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for k, runs in R.items():
        for r in runs:
            vl = r["log"]["val_log"]
            axes[0].plot([v["step"] for v in vl], [v.get("ans_m46", np.nan) for v in vl], LS.get(k, "-"),
                         color=COL[k], lw=1.3, label=LABEL[k] if r is runs[0] else None)
            axes[1].plot([v["step"] for v in vl], [v["val_ce"] for v in vl], LS.get(k, "-"), color=COL[k], lw=1.3,
                         label=LABEL[k] if r is runs[0] else None)
    axes[0].axhline(0.8, color=INK2, lw=1, ls=":")
    axes[0].set_title("точность ответа при m ≥ 4 по ходу обучения (все сиды)", loc="left")
    axes[1].set_title("val CE (600 строк)", loc="left")
    for ax in axes:
        ax.set_xlabel("шаг")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "r3_curves.png"), dpi=130)
    plt.close(fig)

    B = {k: v for k, v in R.items() if v[0]["ev"]["budget"]}
    if B:
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        for k, runs in B.items():
            band(axes[0], k, runs, lambda e, l: e["standing_rel_layer"][l], range(4))
            band(axes[1], k, runs, lambda e, l: e["standing_dd_layer"][l], range(4))
            band(axes[2], k, runs, lambda e, b: e["last_move_hist"][b + 1], range(-1, 4))
        axes[0].set_title("доля стоящих групп: |Δx|/RMS_гр < 0.01", loc="left")
        axes[1].set_title("доля стоящих групп: dd < 0.01", loc="left")
        axes[2].set_title("последний блок, где группа меняется (−1 = никогда)", loc="left")
        for ax in axes[:2]:
            ax.set_xlabel("блок")
            ax.axhline(0.5, color=INK2, lw=1, ls=":")
        axes[2].set_xlabel("блок")
        axes[2].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "r3_standing.png"), dpi=130)
        plt.close(fig)


def main(runs="runs5", out="report"):
    os.makedirs(out, exist_ok=True)
    R = load(runs)
    print({k: len(v) for k, v in R.items()})
    tables(R, out)
    figures(R, out)


if __name__ == "__main__":
    main(*sys.argv[1:])
