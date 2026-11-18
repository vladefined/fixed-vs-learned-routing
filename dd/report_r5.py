"""Round-5 tables (report/r5_tables.md) and figures (report/r5_*.png).
python -m dd.report_r5 [runs7] [runs6] [runs5] [report] [runs7x]"""
from __future__ import annotations

import itertools
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .report import INK2  # noqa: F401  (also applies the shared rcParams)
from .report5 import jump_step, ms
from .evaluate5 import flop_model
from .model import Cfg

NEW = ["QL0start", "QG25", "QG50", "QGsplit25", "QGnogate25"]
REF = ["A", "Q0", "QL0", "K25STE"]
QG = ["QG25", "QG50", "QGsplit25", "QGnogate25"]
LABEL = {"A": "A", "Q0": "Q0 (без цены)", "QL0": "Q-L0 (цена с 24k)", "QL0start": "Q-L0-start (цена с 0)",
         "K25STE": "K25-STE", "QG25": "QG25", "QG50": "QG50", "QGsplit25": "QG-split25", "QGnogate25": "QG-nogate25"}
COL = {"A": "#0b0b0b", "Q0": "#a39e93", "QL0": "#f6b89c", "QL0start": "#eb6834", "K25STE": "#c9c1f5",
       "QG25": "#2a78d6", "QG50": "#9ec5f4", "QGsplit25": "#1baf7a", "QGnogate25": "#6250d6"}


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def load(runs7, runs6, runs5):
    R = {}
    for base, keys in ((runs5, ["A"]), (runs6, ["Q0", "QL0", "K25STE"]), (runs7, NEW + ["A", "Q0"])):
        for d in sorted(os.listdir(base)):
            p = os.path.join(base, d)
            k = d.rsplit("_s", 1)[0]
            if k not in keys or not os.path.exists(os.path.join(p, "log.json")):
                continue
            R.setdefault(k, []).append(dict(name=d, path=p, seed=int(d.rsplit("_s", 1)[1]),
                                            ev=jl(os.path.join(p, "eval5.json")), log=jl(os.path.join(p, "log.json")),
                                            r4=jl(os.path.join(p, "eval_r4.json"))))
    for k in R:
        R[k].sort(key=lambda r: r["seed"])
    return R


def f(v, fmt="{:.2f}", sep=" "):
    return sep.join(fmt.format(x) for x in v)


def jk(j):
    return f"{j // 1000}k" if j else "нет"


def rank_test(a, b):
    """Exact two-sided permutation test on the difference of mean ranks; no jump = 80k+ (ties at the top)."""
    x = np.array([v if v else 1e9 for v in a] + [v if v else 1e9 for v in b], float)
    from scipy.stats import rankdata
    r = rankdata(x)
    na = len(a)
    obs = abs(r[:na].mean() - r[na:].mean())
    cnt = tot = 0
    for c in itertools.combinations(range(len(x)), na):
        m = np.zeros(len(x), bool)
        m[list(c)] = True
        tot += 1
        cnt += abs(r[m].mean() - r[~m].mean()) >= obs - 1e-9
    return cnt / tot


def kste_realisable(r, ctx):
    """K-STE (gate after attention, not split): standing group skips its w2 rows; token with no shift skips the FFN."""
    cfg = Cfg(**r["log"]["result"]["cfg"])
    st = r["r4"]["kste"]["stats"]
    fm = flop_model(cfg, ctx)
    tot = fm["ffn"] + fm["attn"]
    G, gs, hid, d = cfg.depth_groups, cfg.d // cfg.depth_groups, cfg.ffn_hidden, cfg.d
    fs, fn = 1 - st["shift_frac"], float(np.mean(st["token_no_shift_layer"]))
    ffn = fn * fm["ffn"] + max(fs * G - fn * G, 0) * 2 * hid * gs - 2 * d * G
    comp = fs * (fm["ffn"] + G * fm["attn_skip_per_group"]) / tot
    return comp, ffn / tot


def tables(R, out):
    L = []
    # ------------------------------------------------------------------ quality
    L.append("### Качество (среднее [мин, макс] по сидам)\n")
    L.append("| вариант | сидов | val CE−H | ответ, % | m=0…3 | m=4 | m=5 | m=6 | шаг скачка (m≥4 > 80%) по сидам |")
    L.append("|" + "---|" * 9)
    for k in REF + NEW:
        runs = [r for r in R.get(k, []) if r["ev"]]
        if not runs:
            continue
        ev = [r["ev"] for r in runs]
        m03 = [np.mean([e["answer_acc_by_m"][str(m)] for m in range(4)]) for e in ev]
        L.append(f"| {LABEL[k]} | {len(runs)} | {ms([e['val_excess_ce'] for e in ev])} | "
                 f"{ms([e['answer_acc'] for e in ev], '{:.1f}', 100)} | {ms(m03, '{:.1f}', 100)} | " +
                 " | ".join(ms([e['answer_acc_by_m'][str(m)] for e in ev], '{:.1f}', 100) for m in (4, 5, 6)) +
                 f" | {', '.join(f's{r['seed']} ' + jk(jump_step(r['log'])) for r in runs)} |")
    L.append("")
    # ------------------------------------------------------------------ jump distribution A vs Q0
    ja = [jump_step(r["log"]) for r in R.get("A", [])]
    jq = [jump_step(r["log"]) for r in R.get("Q0", [])]
    L.append("### Шаг скачка: A против Q0\n")
    L.append("| вариант | сидов | шаги скачка (отсортированы) | медиана | среднее (без «нет») |")
    L.append("|---|---|---|---|---|")
    for k, js in (("A", ja), ("Q0", jq)):
        v = [j for j in js if j]
        L.append(f"| {LABEL[k]} | {len(js)} | {', '.join(jk(j) for j in sorted(js, key=lambda j: j or 1e9))} | "
                 f"{jk(int(np.median([j or 10 ** 9 for j in js])))} | {np.mean(v) / 1000:.1f}k |")
    if ja and jq:
        L.append(f"\nТочный перестановочный тест по рангам (двусторонний, «нет скачка» = худший ранг): p = {rank_test(ja, jq):.3f}.\n")
    # ------------------------------------------------------------------ learning stages (first val step > 0.8)
    def stage(log, key):
        for v in log["val_log"]:
            if v.get(key, 0) > 0.8:
                return v["step"]
        return None
    STG = (("ans_m12", "m = 1–2"), ("ans_m34", "m = 3–4"), ("ans_m46", "m ≥ 4"))
    L.append("### Этапы обучения: первый шаг, где точность на валидации > 80%\n")
    L.append("| вариант | " + " | ".join(n for _, n in STG) + " |")
    L.append("|---|" + "---|" * len(STG))
    for k in ("A", "Q0", "QL0", "QL0start") + tuple(n for n in NEW if n.startswith("QG")):
        runs = R.get(k, [])
        if runs:
            L.append(f"| {LABEL[k]} | " + " | ".join(", ".join(jk(stage(r["log"], key)) for r in runs)
                                                     for key, _ in STG) + " |")
    if ja and jq:
        ps = [rank_test([stage(r["log"], key) for r in R["A"]], [stage(r["log"], key) for r in R["Q0"]])
              for key, _ in STG]
        L.append("\nA против Q0, тот же тест по рангам: " +
                 ", ".join(f"{n}: p = {p:.3f}" for (_, n), p in zip(STG, ps)) + ".\n")
    # ------------------------------------------------------------------ cost trajectory of Q-L0-start
    L.append("### Q-L0-start: цена и точность по ходу обучения\n")
    L.append("| прогон | шаг | цель | μ | доля сдвинутых координат (блоки 0–3) | точность m≥4 (val) |")
    L.append("|---|---|---|---|---|---|")
    for r in R.get("QL0start", []):
        tl = {t["step"]: t for t in r["log"]["train_log"]}
        vl = {v["step"]: v for v in r["log"]["val_log"]}
        for s in (2000, 5000, 8000, 16000, 24000, 32000, 40000, 48000, 64000, 80000):
            t = tl.get(s) or tl.get(s - 1) or tl.get(min(tl, key=lambda q: abs(q - s)))
            v = vl.get(s)
            L.append(f"| {r['name']} | {s // 1000}k | {t.get('target', float('nan')):.3f} | {np.mean(t.get('mu', [0])):.3f} | "
                     f"{f(t['q_l0_layer'])} | {'—' if v is None else format(v['ans_m46'], '.2f')} |")
    L.append("")
    # ------------------------------------------------------------------ Q family checks (QL0start vs QL0, Q0)
    L.append("### Q-L0-start против Q-L0: сетка и стояние (по сидам)\n")
    L.append("| прогон | μ в конце | доля сдвинутых координат (0–3) | стоящих групп (0–3) | на краях ±8 (эмб., 0–3) | "
             "max\\|Δx\\| стоящей | ответ: как есть → подстановка «группа» по таблице, % |")
    L.append("|---|---|---|---|---|---|---|")
    for k in ("QL0", "QL0start"):
        for r in R.get(k, []):
            if not r["r4"]:
                continue
            q = r["r4"]["q"]
            mu = np.mean(r["log"]["train_log"][-1].get("mu", [0.0]))
            sp = q["standing_prediction"]
            L.append(f"| {r['name']} | {mu:.2f} | {f(q['moved_share_layer'])} | {f(q['group_standing_layer'])} | "
                     f"{f(q['edge_share_layer'])} | {q['max_abs_change_standing_coord']:.0g} | "
                     f"{100 * sp['base']['acc']:.1f} → {100 * sp['group']['skip']['acc']:.1f} |")
    L.append("")
    # ------------------------------------------------------------------ depth vs m
    L.append("### Глубина на позиции ответа по m и AUROC «глубина → ошибка»\n")
    L.append("| прогон | мера | m = 0 … 6 | m=6 / m=0 | AUROC всего | ошибок (m = 4/5/6) |")
    L.append("|---|---|---|---|---|---|")

    def row(r, meas, d, au, nw):
        v = [d[str(m)] for m in range(7)]
        L.append(f"| {r['name']} | {meas} | {f(v, '{:.1f}' if max(v) > 9 else '{:.2f}')} | {v[6] / max(v[0], 1e-9):.2f} | "
                 f"{au:.2f} | {nw.get('4', 0)}/{nw.get('5', 0)}/{nw.get('6', 0)} |")
    for k in ("Q0", "QL0", "QL0start"):
        for r in R.get(k, []):
            if r["r4"]:
                q = r["r4"]["q"]
                row(r, "уровней на коорд. (сумма по блокам)", q["answer_depth_by_m"], q["auroc_wrong"]["all"], q["n_wrong_by_m"])
    for r in R.get("K25STE", []):
        s = r["r4"]["kste"]["stats"]
        row(r, "сдвинутых групп из 32", s["answer_shifts_by_m"], s["auroc_wrong"]["all"], s["n_wrong_by_m"])
    for k in QG:
        for r in R.get(k, []):
            if r["r4"]:
                s = r["r4"]["qg"]["stats"]
                row(r, "открытых групп из 32" if "nogate" not in k else "сдвинутых групп из 32",
                    s["answer_shifts_by_m"], s["auroc_wrong"]["all"], s["n_wrong_by_m"])
                if "nogate" not in k:
                    row(r, "из них реально изменившихся", s["answer_real_by_m"], float("nan"), {})
    L.append("")
    # ------------------------------------------------------------------ QG checks
    L.append("### QG: проверки (по сидам)\n")
    L.append("| прогон | μ в конце | оплачено (доля открытых групп, блоки 0–3) | реально изменилось | впустую оплачено / открытых | "
             "max\\|Δx\\| закрытой группы | обучение − оценка, max\\|Δлогит\\| | групп «никогда» по блокам | групп «всегда» по блокам | "
             "токенов без открытых групп | на краях ±8 (эмб., 0–3) | мёртвых уровней |")
    L.append("|" + "---|" * 12)
    for k in QG:
        for r in R.get(k, []):
            if not r["r4"]:
                continue
            g = r["r4"]["qg"]
            s = g["stats"]
            L.append(f"| {r['name']} | {g['mu_final'][0]:.3f} | {f(s['paid_layer'])} | {f(s['real_layer'])} | "
                     f"{f(s['wasted_of_paid_layer'], '{:.3f}')} | {s['max_abs_change_of_standing_group']:.0g} | "
                     f"{g['train_eval_max_abs_logit_diff']:.0g} | {f(s['never_open_frac_layer'])} | {f(s['always_open_frac_layer'])} | "
                     f"{f(s['token_none_open_layer'])} | {f(s['edge_share_layer'])} | {f(s['dead_levels_layer'], '{:d}')} |")
    L.append("")
    # ------------------------------------------------------------------ table prediction
    L.append("### QG: предсказание решения по таблице «токен → решение» (блок × группа) и подстановка\n")
    L.append("| прогон | точность таблицы (блоки 0–3) | доля закрытых | предсказано закрытыми | точность «закрыта» | полнота «закрыта» | "
             "ответ: как есть → с подстановкой, % | Δ val CE |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k in QG:
        for r in R.get(k, []):
            if not r["r4"]:
                continue
            p = r["r4"]["qg"]["table_prediction"]
            L.append(f"| {r['name']} | {f(p['acc_layer'])} | {f(p['standing_rate'])} | {f(p['predicted_standing_rate'])} | "
                     f"{f(p['precision_standing'])} | {f(p['recall_standing'])} | {100 * p['base']['acc']:.1f} → {100 * p['skip']['acc']:.1f} | "
                     f"{p['skip']['val_ce'] - p['base']['val_ce']:+.4f} |")
    L.append("")
    # ------------------------------------------------------------------ savings
    L.append("### Экономия вычислений блока (доля FLOPs блока; ctx ≈ средняя длина контекста)\n")
    L.append("«Сопоставимая» — оценка раунда 4: стоящая группа пропускает свой срез FFN и свою голову (запрос, оценки, выход); "
             "предполагает, что решение известно до внимания. «Реализуемая» — с учётом того, что гейт читает rmsnorm(x + a), "
             "т. е. внимание всех голов считается всегда: QG пропускает строки w2 закрытых групп и весь FFN, если у токена "
             "закрыты все группы; QG-split пропускает блок FFN закрытой группы; стоимость гейта вычтена. "
             "Q-L0 / QG-nogate: решение известно только после FFN, реализуемая экономия без таблицы = 0.\n")
    L.append("| прогон | закрытых троек (токен × блок × группа) | токенов без открытых групп | сопоставимая (блок) | реализуемая: FFN / блок | "
             "реализуемая по таблице: блок |")
    L.append("|---|---|---|---|---|---|")
    for k in QG:
        for r in R.get(k, []):
            if not r["r4"]:
                continue
            sk = r["r4"]["qg"]["skip"]
            L.append(f"| {r['name']} | {sk['frac_group_standing']:.2f} | {sk['frac_token_none_open']:.2f} | "
                     f"{100 * sk['post']['comparable_block']:.0f}% | {100 * sk['post']['realisable_ffn']:.0f}% / "
                     f"{100 * sk['post']['realisable_block']:.0f}% | " +
                     ("— |" if k.startswith("QGnogate") else
                      f"{100 * sk['table']['realisable_block']:.0f}% (с потерей качества) |"))
    for k in ("QL0", "QL0start"):
        for r in R.get(k, []):
            if not r["r4"]:
                continue
            sk = r["r4"]["q"]["skip_estimate"]
            L.append(f"| {r['name']} | {sk['frac_group_standing']:.2f} | — | {100 * sk['post_hoc']['block']:.0f}% | 0% (после FFN) | "
                     f"{100 * sk['table_predicted']['block']:.0f}% (с потерей качества) |")
    for r in R.get("K25STE", []):
        ctx = r["ev"]["skip_estimate"]["ctx"]
        comp, real = kste_realisable(r, ctx)
        st = r["r4"]["kste"]["stats"]
        L.append(f"| {r['name']} | {1 - st['shift_frac']:.2f} | {np.mean(st['token_no_shift_layer']):.2f} | {100 * comp:.0f}% | "
                 f"— / {100 * real:.0f}% | — |")
    L.append("")
    # ------------------------------------------------------------------ QG trajectory
    L.append("### QG: расстановка решений по чекпойнтам\n")
    L.append("| прогон | шаг | оплачено (среднее по блокам) | реально изменилось | совп. с итогом | совп. с пред. | открытых на ответе, m = 0 / 3 / 6 |")
    L.append("|---|---|---|---|---|---|---|")
    for k in QG:
        for r in R.get(k, []):
            if not r["r4"]:
                continue
            js = jump_step(r["log"])
            for t in r["r4"]["qg"]["ckpt_traj"]:
                a = t["answer_shifts_by_m"]
                L.append(f"| {r['name']} (скачок: {jk(js)}) | {t['step'] // 1000}k | {np.mean(t['paid_layer']):.2f} | "
                         f"{np.mean(t['real_layer']):.2f} | {t['agree_final']:.3f} | "
                         f"{'—' if t['agree_prev'] is None else format(t['agree_prev'], '.3f')} | {a['0']:.1f} / {a['3']:.1f} / {a['6']:.1f} |")
    L.append("")
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "r5_tables.md"), "w", encoding="utf-8").write("\n".join(L))


def curves(R, out):
    groups = [("A и Q0 (5 сидов), Q-L0-start", ["A", "Q0", "QL0start"]),
              ("QG и Q-L0", ["QL0", "QG25", "QG50", "QGsplit25", "QGnogate25"])]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True)
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
    fig.tight_layout()
    fig.savefig(os.path.join(out, "r5_curves.png"), dpi=130)
    plt.close(fig)


def cost_fig(R, out):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.2))
    for k in ["QL0start", "QL0"] + QG:
        for i, r in enumerate(R.get(k, [])):
            tl = r["log"]["train_log"]
            st = [t["step"] / 1000 for t in tl]
            y = np.convolve([np.mean(t["q_l0_layer"]) for t in tl], np.ones(10) / 10, mode="valid")
            st = st[4:4 + len(y)]
            ax.plot(st, y, color=COL[k], lw=1.2, label=LABEL[k] if i == 0 else None)
    ax.set_title("оплачиваемая доля (Q: сдвинутые координаты; QG: открытые группы)", loc="left")
    ax.set_xlabel("шаг, тыс.")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "r5_cost.png"), dpi=130)
    plt.close(fig)


def depth_fig(R, out):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.2))
    for k in ["K25STE"] + QG:
        for i, r in enumerate(R.get(k, [])):
            if not r["r4"]:
                continue
            s = r["r4"]["kste" if k == "K25STE" else "qg"]["stats"]["answer_shifts_by_m"]
            ax.plot(range(7), [s[str(m)] for m in range(7)], color=COL[k], marker="o", ms=4, lw=1.4,
                    ls="--" if k == "K25STE" else "-", label=LABEL[k] if i == 0 else None)
    ax.set_title("открытых (сдвинутых) групп на позиции ответа, из 32", loc="left")
    ax.set_xlabel("m (число перестановок)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "r5_depth_by_m.png"), dpi=130)
    plt.close(fig)


def ext(R, runs7x, out):
    """Runs continued from 80k to 120k (run_r5x.py): appended to r5_tables.md + r5_ext.png."""
    X = []
    for d in sorted(os.listdir(runs7x)):
        p = os.path.join(runs7x, d)
        k = d.rsplit("_s", 1)[0]
        base = next((r for r in R.get(k, []) if r["name"] == d), None)
        if base is None or not os.path.exists(os.path.join(p, "eval_r4.json")):
            continue
        lg = jl(os.path.join(p, "log.json"))
        X.append(dict(k=k, name=d, base=base, ev=jl(os.path.join(p, "eval5.json")), r4=jl(os.path.join(p, "eval_r4.json")),
                      vl=base["log"]["val_log"] + lg["val_log"], tl=lg["train_log"]))
    if not X:
        return

    def stage(vl, key):
        return next((v["step"] for v in vl if v.get(key, 0) > 0.8), None)
    L = ["", "### Продление до 120k (QG50_s1, QG-split25_s0)\n",
         "С чекпойнта 80k: веса и μ, поток данных с того же места, LR на конечном уровне косинуса 80k (0.1·lr) после "
         "500 шагов повторного разогрева (состояние AdamW в чекпойнтах не хранится), цель цены прежняя.\n",
         "| прогон | ответ, % (80k → 120k) | m = 4 / 5 / 6, % | val CE−H | этапы m = 1–2 / 3–4 / ≥ 4 | μ max за продление | "
         "оплачено (блоки 0–3) | открытых групп на ответе, m = 0 … 6 | m=6 / m=0 | реализуемая: FFN / блок | подстановка по таблице |",
         "|" + "---|" * 11]
    for x in X:
        b, e, s = x["base"], x["ev"], x["r4"]["qg"]
        mus = [t["mu"][0] for t in x["tl"] if "mu" in t]
        d = s["stats"]["answer_shifts_by_m"]
        sk, tp = s["skip"]["post"], s["table_prediction"]
        acc = lambda ev: " / ".join(f"{100 * ev['answer_acc_by_m'][str(m)]:.1f}" for m in (4, 5, 6))
        L.append(f"| {x['name']} | {100 * b['ev']['answer_acc']:.1f} → {100 * e['answer_acc']:.1f} | "
                 f"{acc(b['ev'])} → {acc(e)} | {b['ev']['val_excess_ce']:.4f} → {e['val_excess_ce']:.4f} | "
                 f"{' / '.join(jk(stage(x['vl'], k)) for k in ('ans_m12', 'ans_m34', 'ans_m46'))} | {max(mus):.3f} | "
                 f"{f(s['stats']['paid_layer'])} | {f([d[str(m)] for m in range(7)], '{:.1f}')} | "
                 f"{d['6'] / max(d['0'], 1e-9):.2f} | {100 * sk['realisable_ffn']:.0f}% / {100 * sk['realisable_block']:.0f}% | "
                 f"{100 * tp['base']['acc']:.1f} → {100 * tp['skip']['acc']:.1f} |")
    L.append("")
    with open(os.path.join(out, "r5_tables.md"), "a", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.2))
    for x in X:
        st = [v["step"] / 1000 for v in x["vl"]]
        for key, ls, lab in (("ans_m34", ":", "m = 3–4"), ("ans_m56", "-", "m = 5–6")):
            ax.plot(st, [v.get(key, np.nan) for v in x["vl"]], color=COL[x["k"]], lw=1.4, ls=ls,
                    label=f"{x['name']}, {lab}")
    ax.axvline(80, color=INK2, lw=1, ls="--")
    ax.axhline(0.8, color=INK2, lw=1, ls=":")
    ax.set_title("продление 80k → 120k: точность на валидации", loc="left")
    ax.set_xlabel("шаг, тыс.")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "r5_ext.png"), dpi=130)
    plt.close(fig)


def main(runs7="runs7", runs6="runs6", runs5="runs5", out="report", runs7x="runs7x"):
    R = load(runs7, runs6, runs5)
    tables(R, out)
    if os.path.isdir(runs7x):
        ext(R, runs7x, out)
    curves(R, out)
    cost_fig(R, out)
    depth_fig(R, out)
    print("written", os.path.join(out, "r5_tables.md"))


if __name__ == "__main__":
    main(*sys.argv[1:])
