"""Variant-K tables (report/k_tables.md) and figures (report/k_*.png).  python -m dd.report_k [runs5] [report]"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .report import GRID, INK2  # noqa: F401  (also applies the shared rcParams)
from .report5 import jump_step, ms

KORD = ["K25", "K50", "K25L", "K25free", "K25norm"]
REF = ["A", "E", "G"]
COL = {"K25": "#2a78d6", "K50": "#6250d6", "K25L": "#1baf7a", "K25free": "#eb6834", "K25norm": "#e87ba4",
       "A": "#52514e", "E": "#9ec5f4", "G": "#a3a29c"}
LABEL = {"K25": "K25 (цель 25%)", "K50": "K50 (цель 50%)", "K25L": "K25L (μ на блок)",
         "K25free": "K25-free (без tanh и нормировки)", "K25norm": "K25-norm (нормировка, без tanh)",
         "A": "A", "E": "E", "G": "G"}
RAMP_END = 16000


def load(runs):
    R = {}
    for d in sorted(os.listdir(runs)):
        p = os.path.join(runs, d)
        k = d.rsplit("_s", 1)[0]
        if k not in KORD + REF or not os.path.exists(os.path.join(p, "eval5.json")):
            continue
        rec = dict(name=d, ev=json.load(open(os.path.join(p, "eval5.json"))),
                   log=json.load(open(os.path.join(p, "log.json"))), path=p)
        if os.path.exists(os.path.join(p, "evalk.json")):
            rec["k"] = json.load(open(os.path.join(p, "evalk.json")))
        R.setdefault(k, []).append(rec)
    return {k: R[k] for k in REF + KORD if k in R}


def stability(log):
    t = log["train_log"]
    st = np.array([x["step"] for x in t])
    gn = np.array([x["gn"] for x in t])
    ce = np.array([x["ce"] for x in t])
    # skip warmup (first 1000 steps).  Spikes: gn > 3x the median of the previous 20 records (records every
    # 100 steps); CE > previous-20 median + 0.15 nat (batch-to-batch CE noise has p99 ~ 0.10)
    w = st >= 1000
    gsp = sum(gn[i] > 3 * np.median(gn[i - 20:i]) for i in range(20, len(gn)) if w[i])
    cdev = np.array([ce[i] - np.median(ce[i - 20:i]) for i in range(20, len(ce)) if w[i]])
    j = int(np.argmax(np.where(w, gn, -1)))
    out = dict(gn_med=float(np.median(gn[w])), gn_p99=float(np.percentile(gn[w], 99)), gn_max=float(gn[j]),
               gn_max_step=int(st[j]), gn_spikes=int(gsp), ce_spikes=int((cdev > 0.15).sum()),
               ce_dev_max=float(cdev.max()))
    if "mu" in t[-1]:
        mu = np.array([np.mean(x["mu"]) for x in t])
        po = np.array([x["p_open_layer"] for x in t])           # steps x L
        after = st >= RAMP_END + 4000
        out.update(mu_max=float(mu.max()), mu_end_ramp=float(mu[st >= RAMP_END][0]), mu_final=float(mu[-1]),
                   mu_after_mean=float(mu[after].mean()),
                   share_drift=float(np.abs(po[after] - po[after][0]).max()),
                   share_final=po[-1].tolist(), target=float(t[-1]["target"]))
    return out


def tables(R, out):
    L = []
    L.append("### Качество (среднее [мин, макс] по сидам; K — жёсткий порог при оценке)\n")
    L.append("| вариант | сидов | val CE−H | ответ, % | m=0…3 | 4 | 5 | 6 | шаг скачка (m≥4 > 80%) | ответ с сэмплированием гейтов, % | ΔCE сэмпл − порог |")
    L.append("|" + "---|" * 11)
    for k, runs in R.items():
        ev = [r["ev"] for r in runs]
        jumps = [jump_step(r["log"]) for r in runs]
        m03 = [np.mean([e["answer_acc_by_m"][str(m)] for m in range(4)]) for e in ev]
        if k in KORD:
            samp = ms([r["k"]["sampled"]["acc"] for r in runs], "{:.1f}", 100)
            dce = ms([r["k"]["sampled"]["val_ce"] - r["k"]["hard"]["val_ce"] for r in runs], "{:+.4f}")
        else:
            samp = dce = "—"
        L.append(f"| {LABEL[k]} | {len(runs)} | {ms([e['val_excess_ce'] for e in ev])} | "
                 f"{ms([e['answer_acc'] for e in ev], '{:.1f}', 100)} | {ms(m03, '{:.1f}', 100)} | " +
                 " | ".join(ms([e['answer_acc_by_m'][str(m)] for e in ev], '{:.1f}', 100) for m in (4, 5, 6)) +
                 f" | {', '.join(str(j) if j else 'нет' for j in jumps)} | {samp} | {dce} |")
    L.append("")

    K = {k: v for k, v in R.items() if k in KORD}
    L.append("### Доля сдвигов, проверки (по сидам: s0; s1)\n")
    L.append("| вариант | цель | доля сдвигов по блокам 0–3 (порог) | всего: порог / сэмпл | дробных гейтов при сэмпле, макс | "
             "max \\|Δx\\| стоящей группы | ‖x‖ выход | max\\|x_i\\|/медиана, макс по слоям |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, runs in K.items():
        h = [r["k"]["stats"]["hard"] for r in runs]
        s = [r["k"]["stats"]["sampled"] for r in runs]
        tgt = stability(runs[0]["log"])["target"]
        L.append(f"| {LABEL[k]} | {tgt:.2f} | " + "; ".join(" ".join(f"{v:.2f}" for v in x["shift_frac_layer"]) for x in h) +
                 " | " + "; ".join(f"{a['shift_frac']:.3f} / {b['shift_frac']:.3f}" for a, b in zip(h, s)) +
                 " | " + "; ".join(f"{max(x['fractional_gate_frac_layer']):.3f}" for x in s) +
                 " | " + "; ".join(f"{x['max_abs_change_of_standing_group']:.1g}" for x in h) +
                 " | " + "; ".join(f"{r['ev']['norm_layer'][-1]:.1f}" for r in runs) +
                 " | " + "; ".join(f"{max(r['ev']['maxabs_over_median_layer']):.0f}" for r in runs) + " |")
    L.append("")

    L.append("### Структура сдвигов\n")
    L.append("| вариант | групп, которые никогда не сдвигаются, по блокам (s0; s1) | всегда сдвигаются | "
             "\\|Δx\\|/RMS сдвинутой группы: медиана [p10, p90] | доля сдвигов с изменением < 0.1 | "
             "сдвинутых групп на позиции ответа, m = 0…6 (s0; s1) |")
    L.append("|---|---|---|---|---|---|")
    for k, runs in K.items():
        h = [r["k"]["stats"]["hard"] for r in runs]
        q = [x["moving_rel_change_quantiles"] for x in h]
        L.append(f"| {LABEL[k]} | " + "; ".join(" ".join(f"{v:.2f}" for v in x["never_shift_frac_layer"]) for x in h) +
                 " | " + "; ".join(" ".join(f"{v:.2f}" for v in x["always_shift_frac_layer"]) for x in h) +
                 " | " + "; ".join(f"{a['0.5']:.2f} [{a['0.1']:.2f}, {a['0.9']:.1f}]" for a in q) +
                 " | " + "; ".join(f"{x['moving_small_frac']['0.1']:.3f}" for x in h) +
                 " | " + "; ".join(" ".join(f"{x['answer_shifts_by_m'][str(m)]:.1f}" for m in range(7)) for x in h) + " |")
    L.append("")

    L.append("### Устойчивость обучения (по сидам: s0; s1)\n")
    L.append("| вариант | норма градиента после 1k шагов: медиана / p99 / макс (шаг) | всплесков gn (>3× медианы 20 пред.) | "
             "всплесков CE (>+0.15 нат) / макс. отклонение | "
             "μ: макс / в конце разгона / среднее после 20k / в конце | макс. дрейф доли сдвигов после 20k (по блокам) |")
    L.append("|---|---|---|---|---|---|")
    for k, runs in R.items():
        S = [stability(r["log"]) for r in runs]
        mu = "; ".join(f"{s['mu_max']:.2f} / {s['mu_end_ramp']:.2f} / {s['mu_after_mean']:.3f} / {s['mu_final']:.3f}"
                       for s in S) if "mu_max" in S[0] else "—"
        dr = "; ".join(f"{s['share_drift']:.3f}" for s in S) if "share_drift" in S[0] else "—"
        L.append(f"| {LABEL[k]} | " + "; ".join(f"{s['gn_med']:.2f} / {s['gn_p99']:.2f} / {s['gn_max']:.2f} ({s['gn_max_step'] // 1000}k)"
                                                  for s in S) +
                 " | " + "; ".join(str(s["gn_spikes"]) for s in S) +
                 " | " + "; ".join(f"{s['ce_spikes']} / {s['ce_dev_max']:.2f}" for s in S) +
                 f" | {mu} | {dr} |")
    L.append("")

    L.append("### Экономия: доля (токен, блок, группа) без сдвига = пропускаемые срезы FFN\n")
    L.append("| вариант | без сдвига, всего | по блокам 0–3 | для сравнения: стоящие в E / G |")
    L.append("|---|---|---|---|")
    eg = {k: np.mean([r["ev"]["skip_estimate"]["frac_standing_dd"] for r in R[k]]) for k in ("E", "G") if k in R}
    for k, runs in K.items():
        h = [r["k"]["stats"]["hard"] for r in runs]
        tot = [1 - x["shift_frac"] for x in h]
        bl = np.mean([[1 - v for v in x["shift_frac_layer"]] for x in h], 0)
        L.append(f"| {LABEL[k]} | {ms(tot, '{:.0f}', 100)}% | {' '.join(f'{100 * v:.0f}%' for v in bl)} | "
                 f"{100 * eg.get('E', np.nan):.0f}% / {100 * eg.get('G', np.nan):.0f}% |")
    L.append("")
    open(os.path.join(out, "k_tables.md"), "w", encoding="utf-8").write("\n".join(L))


def figures(R, out):
    K = {k: v for k, v in R.items() if k in KORD}
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
    for k, runs in K.items():
        for i, r in enumerate(runs):
            v = r["log"]["val_log"]
            ax[0].plot([x["step"] for x in v], [x.get("ans_m46", np.nan) for x in v], color=COL[k],
                       lw=1.6, ls="-" if i == 0 else "--", label=LABEL[k] if i == 0 else None)
    for a in ax:
        a.axvline(RAMP_END, color=INK2, lw=1, ls=":")
    ax[0].axhline(0.8, color=INK2, lw=0.8, ls=":")
    ax[0].set_title("точность при m ≥ 4 (сплошная s0, пунктир s1)")
    ax[0].set_xlabel("шаг")
    ax[0].legend(fontsize=8, loc="upper left")
    for k, runs in K.items():
        for i, r in enumerate(runs):
            t = r["log"]["train_log"]
            st = [x["step"] for x in t]
            ax[1].plot(st, [np.mean(x["mu"]) for x in t], color=COL[k], lw=1.4, ls="-" if i == 0 else "--",
                       label=LABEL[k] if i == 0 else None)
            ax[2].plot(st, [np.mean(x["p_open_layer"]) for x in t], color=COL[k], lw=1.4, ls="-" if i == 0 else "--")
    t = K["K25"][0]["log"]["train_log"]
    ax[2].plot([x["step"] for x in t], [x["target"] for x in t], color=INK2, lw=1, ls=":", label="цель (25%)")
    t = K["K50"][0]["log"]["train_log"]
    ax[2].plot([x["step"] for x in t], [x["target"] for x in t], color=INK2, lw=1, ls="-.", label="цель (50%)")
    ax[1].set_title("μ (среднее по блокам)")
    ax[2].set_title("доля открытых гейтов P(m>0) при обучении")
    for a in ax[1:]:
        a.set_xlabel("шаг")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "k_curves.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
    for k, runs in K.items():
        for i, r in enumerate(runs):
            h = r["k"]["stats"]["hard"]
            ls = "-" if i == 0 else "--"
            ax[0].plot(range(4), h["shift_frac_layer"], marker="o", color=COL[k], ls=ls, label=LABEL[k] if i == 0 else None)
            ax[1].plot(range(7), [h["answer_shifts_by_m"][str(m)] for m in range(7)], marker="o", color=COL[k], ls=ls)
            tl = r["log"]["train_log"]
            ax[2].plot([x["step"] for x in tl], [x["gn"] for x in tl], color=COL[k], lw=0.8, ls=ls, alpha=0.8)
    ax[0].set_title("доля сдвигов по блокам (оценка, порог)")
    ax[0].set_xlabel("блок")
    ax[0].set_xticks(range(4))
    ax[0].legend(fontsize=8)
    ax[1].set_title("сдвинутых групп на позиции ответа (из 32)")
    ax[1].set_xlabel("перестановок до вопроса, m")
    ax[2].set_title("норма градиента до клипа")
    ax[2].set_xlabel("шаг")
    ax[2].set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "k_shifts.png"), dpi=130)
    plt.close(fig)


def main(runs="runs5", out="report"):
    R = load(runs)
    print({k: len(v) for k, v in R.items()})
    tables(R, out)
    figures(R, out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
