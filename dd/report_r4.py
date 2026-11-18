"""Round-4 tables (report/r4_tables.md) and figures (report/r4_*.png).
python -m dd.report_r4 [runs6] [runs5] [report]"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .report import INK2  # noqa: F401  (also applies the shared rcParams)
from .report5 import jump_step, ms

NEW = ["Q0", "QL1w", "QL1s", "QL0", "Ip", "K25STE", "K50STE", "A_groupnorm_free", "E_free", "G_free"]
REF = ["A", "A_groupnorm", "E", "G", "H", "I", "K25", "K50"]
LABEL = {"Q0": "Q0 (без цены)", "QL1w": "Q-L1 цель 0.25", "QL1s": "Q-L1 цель 0.1", "QL0": "Q-L0 цель 0.25",
         "Ip": "I′ (медиана)", "K25STE": "K25-STE", "K50STE": "K50-STE", "A_groupnorm_free": "A_groupnorm + free",
         "E_free": "E + free", "G_free": "G + free", "A": "A", "A_groupnorm": "A_groupnorm", "E": "E", "G": "G",
         "H": "H", "I": "I", "K25": "K25", "K50": "K50"}
COL = {"Q0": "#52514e", "QL1w": "#9ec5f4", "QL1s": "#2a78d6", "QL0": "#eb6834", "Ip": "#1baf7a",
       "K25STE": "#6250d6", "K50STE": "#e87ba4", "A_groupnorm_free": "#a3a29c", "E_free": "#2a78d6",
       "G_free": "#6250d6", "A": "#0b0b0b", "E": "#9ec5f4", "G": "#c9c1f5", "I": "#8fd9bd", "K25": "#c9c1f5",
       "K50": "#f4c3d6", "A_groupnorm": "#d6d5cf", "H": "#f6b89c"}
QV = ["Q0", "QL1w", "QL1s", "QL0"]
FREE = {"A_groupnorm_free": "A_groupnorm", "E_free": "E", "G_free": "G"}


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def load(runs6, runs5):
    R = {}
    for base, keys in ((runs5, REF), (runs6, NEW)):
        for d in sorted(os.listdir(base)):
            p = os.path.join(base, d)
            k = d.rsplit("_s", 1)[0]
            if k not in keys or not os.path.exists(os.path.join(p, "eval5.json")):
                continue
            R.setdefault(k, []).append(dict(name=d, path=p, ev=jl(os.path.join(p, "eval5.json")),
                                            log=jl(os.path.join(p, "log.json")), r4=jl(os.path.join(p, "eval_r4.json")),
                                            ex=jl(os.path.join(p, "r4_extra.json")), tails=jl(os.path.join(p, "tails.json")),
                                            k=jl(os.path.join(p, "evalk.json"))))
    return R


def f(v, fmt="{:.2f}", sep=" "):
    return sep.join(fmt.format(x) for x in v)


def seeds(runs, fn, sep="; "):
    return sep.join(fn(r) for r in runs)


def tables(R, out):
    L = []
    # ------------------------------------------------------------------ quality
    L.append("### Качество (среднее [мин, макс] по сидам)\n")
    L.append("| вариант | сидов | val CE−H | ответ, % | m=0…3 | m=4 | m=5 | m=6 | шаг скачка (m≥4 > 80%) |")
    L.append("|" + "---|" * 9)
    for k in REF + NEW:
        if k not in R:
            continue
        runs = R[k]
        ev = [r["ev"] for r in runs]
        m03 = [np.mean([e["answer_acc_by_m"][str(m)] for m in range(4)]) for e in ev]
        L.append(f"| {LABEL[k]} | {len(runs)} | {ms([e['val_excess_ce'] for e in ev])} | "
                 f"{ms([e['answer_acc'] for e in ev], '{:.1f}', 100)} | {ms(m03, '{:.1f}', 100)} | " +
                 " | ".join(ms([e['answer_acc_by_m'][str(m)] for e in ev], '{:.1f}', 100) for m in (4, 5, 6)) +
                 f" | {', '.join(str(j // 1000) + 'k' if j else 'нет' for j in (jump_step(r['log']) for r in runs))} |")
    L.append("")
    # ------------------------------------------------------------------ stream
    L.append("### Поток: ‖x‖ и max|x_i| / RMS(x) на токене по слоям (эмб. → выход; s0; s1)\n")
    L.append("| вариант | ‖x‖ | max\\|x_i\\|/RMS | доля координат с медианой \\|x_i\\| < 0.1·среднего (блоки 0–3) |")
    L.append("|---|---|---|---|")
    for k in REF + NEW:
        if k not in R or R[k][0]["ex"] is None:
            continue
        runs = [r for r in R[k] if r["ex"]]
        L.append(f"| {LABEL[k]} | {seeds(runs, lambda r: f(r['ex']['norm_layer'], '{:.1f}'))} | "
                 f"{seeds(runs, lambda r: f(r['ex']['maxabs_over_rms_layer'], '{:.1f}'))} | "
                 f"{seeds(runs, lambda r: f(r['ex']['sparse']['sparse_layer']))} |")
    L.append("")
    # ------------------------------------------------------------------ Q
    L.append("### Q: движение по сетке (по сидам s0; s1)\n")
    L.append("| вариант | μ в конце | уровней на координату за блок (0–3) | доля сдвинутых координат | n = 0 / 1 / 2 / ≥3 (среднее по блокам) | "
             "стоящих групп (все 16 коорд. n = 0) | max\\|Δx\\| стоящей коорд. | max\\|\\|Δx\\|/Δ − n\\| |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k in QV:
        runs = R.get(k, [])
        q = [r["r4"]["q"] for r in runs]
        mu = [np.mean(r["log"]["train_log"][-1].get("mu", [0.0])) for r in runs]
        L.append(f"| {LABEL[k]} | {f(mu, '{:.2f}', '; ')} | {'; '.join(f(x['l1_layer']) for x in q)} | "
                 f"{'; '.join(f(x['moved_share_layer']) for x in q)} | "
                 f"{'; '.join(f(np.mean(x['n_hist_layer'], 0)) for x in q)} | "
                 f"{'; '.join(f(x['group_standing_layer']) for x in q)} | "
                 f"{'; '.join(f'{x["max_abs_change_standing_coord"]:.0g}' for x in q)} | "
                 f"{'; '.join(f'{x["max_abs_n_minus_dx_over_delta"]:.0e}' for x in q)} |")
    L.append("")
    L.append("### Q: уровни (слои: эмб., выходы блоков 0–3; по сидам s0; s1)\n")
    L.append("| вариант | доля на краях ±8 | доля на нуле | мёртвых уровней (< 10⁻⁴) | уровней на 99% массы |")
    L.append("|---|---|---|---|---|")
    for k in QV:
        q = [r["r4"]["q"] for r in R.get(k, [])]
        L.append(f"| {LABEL[k]} | {'; '.join(f(x['edge_share_layer']) for x in q)} | "
                 f"{'; '.join(f(x['zero_share_layer']) for x in q)} | "
                 f"{'; '.join(f(x['dead_levels_layer'], '{:d}') for x in q)} | "
                 f"{'; '.join(f(x['levels_used_99_layer'], '{:d}') for x in q)} |")
    L.append("")
    # ------------------------------------------------------------------ depth vs m
    L.append("### Глубина на позиции ответа по m и AUROC «глубина → ошибка» внутри m\n")
    L.append("| вариант | мера | m = 0 … 6 (по сидам s0; s1) | отношение m=6 / m=0 | AUROC внутри m = 4 / 5 / 6 (s0; s1) |")
    L.append("|---|---|---|---|---|")

    def au(d):
        return " / ".join("—" if d.get(str(m)) is None or d.get(str(m)) != d.get(str(m)) else f"{d[str(m)]:.2f}"
                          for m in (4, 5, 6))

    def row(k, meas, getter, aug):
        runs = R.get(k, [])
        if not runs:
            return
        vals = [[getter(r)[str(m)] for m in range(7)] for r in runs]
        L.append(f"| {LABEL[k]} | {meas} | {'; '.join(f(v, '{:.1f}' if max(v) > 9 else '{:.2f}') for v in vals)} | "
                 f"{'; '.join(f'{v[6] / v[0]:.2f}' for v in vals)} | {'; '.join(au(aug(r)) for r in runs)} |")
    for k in QV:
        row(k, "уровней на координату (сумма по блокам)", lambda r: r["r4"]["q"]["answer_depth_by_m"],
            lambda r: r["r4"]["q"]["auroc_wrong"]["by_m"])
    for k in ("K25", "K50"):
        row(k, "сдвинутых групп из 32", lambda r: r["k"]["stats"]["hard"]["answer_shifts_by_m"],
            lambda r: r["ev"]["auroc_wrong"]["final_depth"]["by_m"])
    for k in ("K25STE", "K50STE"):
        row(k, "сдвинутых групп из 32", lambda r: r["r4"]["kste"]["stats"]["answer_shifts_by_m"],
            lambda r: r["r4"]["kste"]["stats"]["auroc_wrong"]["by_m"])
    for k in ("E", "G", "A_groupnorm_free", "E_free", "G_free"):
        if k != "A_groupnorm_free":
            row(k, "итоговая глубина", lambda r: r["ev"]["spent_by_m"]["final_depth"],
                lambda r: r["ev"]["auroc_wrong"]["final_depth"]["by_m"])
    for k in ("I", "Ip"):
        row(k, "сумма отн. изменений (платится)", lambda r: r["ev"]["spent_by_m"]["relchg_sum"],
            lambda r: r["ev"]["auroc_wrong"]["relchg_sum"]["by_m"])
        row(k, "итоговая глубина (dd)", lambda r: r["ev"]["spent_by_m"]["final_depth"],
            lambda r: r["ev"]["auroc_wrong"]["final_depth"]["by_m"])
    L.append("")
    # ------------------------------------------------------------------ table prediction / substitution
    L.append("### Предсказание «стоит» до FFN по таблице «токен → доля стояния» и подстановка\n")
    L.append("| вариант | единица | доля стоящих (блоки 0–3) | предсказано стоящими | точность «стоит» | полнота «стоит» | "
             "ответ: как есть → с подстановкой, % | Δ val CE |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k in QV:
        for r in R.get(k, []):
            sp = r["r4"]["q"]["standing_prediction"]
            for kind, nm in (("coord", "координата"), ("group", "группа")):
                s = sp[kind]
                L.append(f"| {LABEL[k]} {r['name'][-2:]} | {nm} | {f(s['standing_rate'])} | {f(s['predicted_standing_rate'])} | "
                         f"{f(s['precision_standing'])} | {f(s['recall_standing'])} | "
                         f"{100 * sp['base']['acc']:.1f} → {100 * s['skip']['acc']:.1f} | {s['skip']['val_ce'] - sp['base']['val_ce']:+.4f} |")
    for k in ("Ip", "E_free", "G_free", "E", "G", "I"):
        for r in R.get(k, []):
            g = r["ev"].get("gate_prediction")
            if not g:
                continue
            L.append(f"| {LABEL[k]} {r['name'][-2:]} | группа (таблица гейта) | — | — | {g['table']['acc']:.2f} (точн. всего) | "
                     f"{np.mean(g['table']['recall_standing']):.2f} | {100 * g['base']['acc']:.1f} → {100 * g['table']['skip']['acc']:.1f} | "
                     f"{g['table']['skip']['val_ce'] - g['base']['val_ce']:+.4f} |")
    L.append("")
    # ------------------------------------------------------------------ savings
    L.append("### Экономия (оценка для блока, разрезанного по группам; доля троек «токен × блок × группа» без изменения)\n")
    L.append("| вариант | стоящих троек, post hoc | экономия FFN / внимания / блока, post hoc | то же, по таблице до FFN | "
             "пар «токен × блок» без единого сдвига (блоки 0–3) |")
    L.append("|---|---|---|---|---|")
    for k in QV:
        runs = R.get(k, [])
        sk = [r["r4"]["q"]["skip_estimate"] for r in runs]
        L.append(f"| {LABEL[k]} | {'; '.join(f'{s["frac_group_standing"]:.2f}' for s in sk)} | "
                 f"{'; '.join(f(100 * np.array([s['post_hoc'][x] for x in ('ffn', 'attn', 'block')]), '{:.0f}%', ' / ') for s in sk)} | "
                 f"{'; '.join(f(100 * np.array([s['table_predicted'][x] for x in ('ffn', 'attn', 'block')]), '{:.0f}%', ' / ') for s in sk)} | — |")
    for k in ("K25STE", "K50STE", "Ip", "E_free", "G_free"):
        runs = R.get(k, [])
        sk = [r["ev"]["skip_estimate"] for r in runs]
        tok = "; ".join(f(r["r4"]["kste"]["stats"]["token_no_shift_layer"]) for r in runs) if k.startswith("K") else "—"
        L.append(f"| {LABEL[k]} | {'; '.join(f'{s["frac_standing_rel"]:.2f}' for s in sk)} | "
                 f"{'; '.join(f(100 * np.array([s[x] for x in ('ffn_saving', 'attn_saving', 'block_saving')]), '{:.0f}%', ' / ') for s in sk)} | "
                 f"{'решение до FFN' if k.startswith('K') else '—'} | {tok} |")
    L.append("")
    # ------------------------------------------------------------------ I' tails
    L.append("### I′: хвосты и честность цены (блоки 0–3; блок 0 — вход = эмбеддинги)\n")
    L.append("| прогон | RMS / медиана \\|x_i\\| | доля \\|x_i\\| > 3·медианы | оплачено / реально изменилось | стоят по собств. значению | "
             "ΔCE / ответ при обнулении стоящих (собств. значение) | разреженных коорд. | коорд., где >50% токенов < 0.05·RMS | пол s ≥ 0.1·mean активен |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k in ("A", "H", "I", "Ip"):
        for r in R.get(k, []):
            t = r["tails"]
            if t is None:
                continue
            ex = r["ex"]["sparse"] if r["ex"] else {}
            zt = (f"{t['dce_zero_own']:+.5f} / {100 * t['answer_base']:.2f} → {100 * t['answer_zero_own']:.2f}"
                  if "dce_zero_own" in t else "—")
            fl = r["r4"]["sparse"].get("floor_active_layer") if r["r4"] else None
            L.append(f"| {r['name']} | {f(t['tail'], '{:.1f}')} | {f(t['frac_gt3med'])} | {f(t['ratio'])} | {f(t['frozen_own'])} | {zt} | "
                     f"{f(ex.get('sparse_layer', []))} | {f(ex.get('near0_majority_layer', []))} | {f(fl) if fl else '—'} |")
    L.append("")
    # ------------------------------------------------------------------ K-STE trajectory
    L.append("### K-STE: траектория расстановки сдвигов по чекпойнтам\n")
    L.append("Цель: 1.0 до 24k (μ = 0), линейно до итоговой к 48k. «Совпадение» — доля решений (токен × блок × группа), "
             "совпадающих с итоговой моделью / с предыдущим чекпойнтом (600 тестовых строк).\n")
    L.append("| прогон | шаг | доля сдвигов | совп. с итогом | совп. с пред. | сдвигов на ответе, m = 0 / 3 / 4 / 6 | точность m = 4 / 5 / 6 |")
    L.append("|---|---|---|---|---|---|---|")
    for k in ("K25STE", "K50STE"):
        for r in R.get(k, []):
            js = jump_step(r["log"])
            for t in r["r4"]["kste"]["ckpt_traj"]:
                a = t["answer_shifts_by_m"]
                L.append(f"| {r['name']} (скачок: {str(js // 1000) + 'k' if js else 'нет'}) | {t['step'] // 1000}k | "
                         f"{np.mean(t['shift_frac_layer']):.2f} | {t['agree_final']:.3f} | "
                         f"{'—' if t['agree_prev'] is None else format(t['agree_prev'], '.3f')} | "
                         f"{a['0']:.1f} / {a['3']:.1f} / {a['4']:.1f} / {a['6']:.1f} | "
                         f"{f([t['acc'][str(m)] for m in (4, 5, 6)], '{:.2f}', ' / ')} |")
    L.append("")
    L.append("### K-STE: проверки и структура (s0; s1)\n")
    L.append("| вариант | доля сдвигов по блокам | max\\|Δx\\| несдвинутой группы | обучение − оценка, max\\|Δлогит\\| | μ в конце | "
             "групп «никогда» по блокам | групп «всегда» по блокам |")
    L.append("|---|---|---|---|---|---|---|")
    for k in ("K25STE", "K50STE"):
        runs = R.get(k, [])
        ks = [r["r4"]["kste"] for r in runs]
        L.append(f"| {LABEL[k]} | {'; '.join(f(x['stats']['shift_frac_layer']) for x in ks)} | "
                 f"{'; '.join(format(x['stats']['max_abs_change_of_standing_group'], '.0g') for x in ks)} | "
                 f"{'; '.join(format(x['train_eval_max_abs_logit_diff'], '.0g') for x in ks)} | "
                 f"{'; '.join(format(x['mu_final'][0], '.3f') for x in ks)} | "
                 f"{'; '.join(f(x['stats']['never_shift_frac_layer']) for x in ks)} | "
                 f"{'; '.join(f(x['stats']['always_shift_frac_layer']) for x in ks)} |")
    L.append("")
    # ------------------------------------------------------------------ free coordinates
    L.append("### Свободные координаты (2 на группу, 16 из 128)\n")
    L.append("| прогон | RMS свободных по слоям (норм. = 1.00) | max\\|x\\| свободных | R² зонда m на вопросе, слои 1–4: свободные / 16 случайных норм. | "
             "R² «перестановок до сих пор» в истории: своб. / 16 норм. | ответ, %: как есть → обновления свободных обнулены | Δ val CE |")
    L.append("|---|---|---|---|---|---|---|")
    for k in FREE:
        for r in R.get(k, []):
            fr = r["r4"]["free"]
            pr = fr["probe_r2"]
            ab = fr["ablate_free_updates"]
            L.append(f"| {r['name']} | {f(fr['rms_free_layer'])} | {f(fr['max_abs_free_layer'], '{:.1f}')} | "
                     f"{' '.join(format(pr[str(l)]['m_free'], '.2f') + '/' + format(pr[str(l)]['m_norm16'], '.2f') for l in range(1, 5))} | "
                     f"{' '.join(format(pr[str(l)]['sofar_free'], '.2f') + '/' + format(pr[str(l)]['sofar_norm16'], '.2f') for l in range(1, 5))} | "
                     f"{100 * ab['base']['acc']:.1f} → {100 * ab['ablated']['acc']:.1f} | {ab['ablated']['val_ce'] - ab['base']['val_ce']:+.3f} |")
    L.append("")
    L.append("### Свободные координаты: max |corr| с признаками (вопросы: m, число имён, позиция; слой 4) — свободные / нормированные\n")
    L.append("| прогон | m | число имён | позиция | перестановок до сих пор (история, слой 4) |")
    L.append("|---|---|---|---|---|")
    for k in FREE:
        for r in R.get(k, []):
            cq, cs = r["r4"]["free"]["corr_question"]["4"], r["r4"]["free"]["corr_story"]["4"]
            g = lambda c, key: f"{np.max(np.abs(c[key])):.2f} / {c['norm_max_abs'][key]:.2f}"
            L.append(f"| {r['name']} | {g(cq, 'qm')} | {g(cq, 'nk')} | {g(cq, 'pos')} | {g(cs, 'sofar')} |")
    L.append("")
    # ------------------------------------------------------------------ zeroing checks
    L.append("### Обнуление стоящих (eval5: ΔCE при обнулении изменений групп, стоящих по |Δx|/RMS_гр < 0.01 и по dd < 0.01)\n")
    L.append("| вариант | ΔCE (отн. мера) s0; s1 | ΔCE (dd) s0; s1 | доля «dd < 0.01, но изменение ≥ 1%» по блокам |")
    L.append("|---|---|---|---|")
    for k in ("Ip", "K25STE", "K50STE", "E_free", "G_free", "I", "E", "G"):
        runs = R.get(k, [])
        if not runs:
            continue
        L.append(f"| {LABEL[k]} | {'; '.join(format(r['ev']['dce_zero_rel'], '+.5f') for r in runs)} | "
                 f"{'; '.join(format(r['ev']['dce_zero_dd'], '+.5f') for r in runs)} | "
                 f"{'; '.join(f([a['dd_only'] for a in r['ev']['standing_agreement_layer']]) for r in runs)} |")
    L.append("")
    open(os.path.join(out, "r4_tables.md"), "w", encoding="utf-8").write("\n".join(L))


def curves(R, out):
    groups = [("Q (квантованное состояние)", ["A"] + QV), ("I′ и K-STE", ["I", "Ip", "K25STE", "K50STE"]),
              ("свободные координаты", ["A_groupnorm", "A_groupnorm_free", "E_free", "G_free"])]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.2), sharey=True)
    for ax, (title, ks) in zip(axes, groups):
        for k in ks:
            for i, r in enumerate(R.get(k, [])):
                vl = r["log"]["val_log"]
                ax.plot([v["step"] / 1000 for v in vl], [v.get("ans_m46", np.nan) for v in vl], color=COL[k], lw=1.4,
                        ls="-" if k in NEW else "--", label=LABEL[k] if i == 0 else None)
        ax.axhline(0.8, color=INK2, lw=1, ls=":")
        ax.set_title(f"точность при m ≥ 4: {title}", loc="left")
        ax.set_xlabel("шаг, тыс.")
        ax.legend(fontsize=8, loc="lower right")
    for x in (24, 48):
        for ax in axes[:2]:
            ax.axvline(x, color="#d6d5cf", lw=1)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "r4_curves.png"), dpi=130)
    plt.close(fig)


def depth_fig(R, out):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.2))
    xs = list(range(7))
    spec = [(axes[0], "Q: уровней на координату на ответе / m=0", QV, lambda r: r["r4"]["q"]["answer_depth_by_m"]),
            (axes[1], "K: сдвинутых групп на ответе (из 32)", ["K25", "K50", "K25STE", "K50STE"],
             lambda r: (r["k"]["stats"]["hard"] if r["k"] else r["r4"]["kste"]["stats"])["answer_shifts_by_m"]),
            (axes[2], "итоговая глубина на ответе / m=0", ["E", "G", "E_free", "G_free"],
             lambda r: r["ev"]["spent_by_m"]["final_depth"])]
    for ax, title, ks, get in spec:
        for k in ks:
            for i, r in enumerate(R.get(k, [])):
                v = np.array([get(r)[str(m)] for m in xs])
                if ax is not axes[1]:
                    v = v / v[0]
                ax.plot(xs, v, marker="o", ms=4, color=COL[k], ls="-" if k in NEW else "--", lw=1.4,
                        label=LABEL[k] if i == 0 else None)
        ax.set_title(title, loc="left")
        ax.set_xlabel("перестановок до вопроса, m")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "r4_depth_by_m.png"), dpi=130)
    plt.close(fig)


def kste_fig(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for k in ("K25STE", "K50STE"):
        for r in R.get(k, []):
            tr = r["r4"]["kste"]["ckpt_traj"]
            st = [t["step"] / 1000 for t in tr]
            js = jump_step(r["log"])
            ls = "-" if r["name"].endswith("s0") else "--"
            lab = f"{r['name']} (скачок {js // 1000}k)" if js else f"{r['name']} (нет скачка)"
            axes[0].plot(st, [t["agree_final"] for t in tr], ls, marker="o", ms=4, color=COL[k], label=lab)
            axes[1].plot(st, [t["answer_shifts_by_m"]["6"] - t["answer_shifts_by_m"]["0"] for t in tr], ls, marker="o",
                         ms=4, color=COL[k], label=lab)
            if js:
                for ax in axes:
                    ax.axvline(js / 1000, color=COL[k], lw=1, ls=":")
    axes[0].set_title("совпадение решений о сдвиге с итоговой моделью", loc="left")
    axes[1].set_title("сдвигов на ответе: m = 6 минус m = 0", loc="left")
    for ax in axes:
        ax.axvspan(24, 48, color="#efeeea", zorder=0)
        ax.set_xlabel("шаг, тыс. (серая полоса — снижение цели)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "r4_kste_traj.png"), dpi=130)
    plt.close(fig)


def q_levels_fig(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    lv = np.arange(-8, 9)
    w = 0.2
    for j, (ax, layer, title) in enumerate(((axes[0], 2, "вход блока 2"), (axes[1], 4, "выход последнего блока"))):
        for i, k in enumerate(QV):
            occ = np.mean([r["r4"]["q"]["level_occupancy"][layer] for r in R.get(k, [])], 0)
            ax.bar(lv + (i - 1.5) * w, occ, width=w, color=COL[k], label=LABEL[k])
        ax.set_title(f"заселённость уровней: {title}", loc="left")
        ax.set_xlabel("уровень")
        ax.set_xticks(lv)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "r4_q_levels.png"), dpi=130)
    plt.close(fig)


def main(runs6="runs6", runs5="runs5", out="report"):
    os.makedirs(out, exist_ok=True)
    R = load(runs6, runs5)
    print({k: len(v) for k, v in R.items()})
    tables(R, out)
    curves(R, out)
    depth_fig(R, out)
    kste_fig(R, out)
    q_levels_fig(R, out)


if __name__ == "__main__":
    main(*sys.argv[1:])
