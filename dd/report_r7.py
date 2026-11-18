"""Round 7 (TinyStories, 20k steps): gate controls, finer grid, cost of the cut.
Tables report/r7_tables.md, figures report/r7_val.png (controls), report/r7_grid.png (grid).
python -m dd.report_r7 [runs_ts] [report]"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .evaluate5 import flop_model
from .model import Cfg, Model
from .report import INK2  # noqa: F401  (shared rcParams)
from .ts_flops import savings

CTRL = [("QG25tokengate", "QG25-tokengate", "#7a4bd6"), ("QG25static", "QG25-static", "#eda100"),
        ("QG25position", "QG25-position", "#e87ba4"), ("QG25random", "QG25-random", "#eb6834")]
SEEDS = range(5)
VAST = "runs_vast"                                            # seeds trained on vast.ai (same code and data)
GRID = [("Q0_s0", "Q0, 81 уровень (шаг Δ)", "#a3a29c"), ("Q0_161_s0", "Q0, 161 уровень (Δ/2)", "#5598e7"),
        ("Q0_321_s0", "Q0, 321 уровень (Δ/4)", "#0d366b")]


def jl(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def load(base, name):
    for b in (base, VAST):                                    # local run first, then a vast.ai one
        run = os.path.join(b, name)
        ev, log = jl(os.path.join(run, "ts_eval.json")), jl(os.path.join(run, "log.json"))
        if ev is not None and log is not None:
            ce = ev.get("val_ce_mask_mean", ev["val_ce"])    # random-mask controls: mean over 3 mask seeds
            return dict(name=name, run=run, ev=ev, log=log, ce=ce, cfg=Cfg(**log["result"]["cfg"]),
                        where="vast" if b == VAST else "дома")
    return None


def seeds(base, prefix):
    return [r for r in (load(base, f"{prefix}_s{s}") for s in SEEDS) if r]


def stats(rs):
    """mean, SD over training seeds (ddof 1; nan for one seed), n"""
    ce = np.array([r["ce"] for r in rs])
    return float(ce.mean()), float(ce.std(ddof=1)) if len(ce) > 1 else float("nan"), len(ce)


def pooled_sd(s1, s2):
    """SD over training seeds pooled over two groups; when one group has a single seed, the other group's SD"""
    (_, a, n), (_, b, m) = s1, s2
    if n < 2 and m < 2:
        return float("nan")
    if n < 2 or m < 2:
        return float(b if n < 2 else a)
    return float(np.sqrt(((n - 1) * a ** 2 + (m - 1) * b ** 2) / (n + m - 2)))


def verdict(d, thr, better, worse):
    """d = CE(first) - CE(second) > 0: the second is better.  The rule: |d| < 2 SD -> not detected."""
    if np.isnan(thr):
        return "один сид в обеих — разброс не оценить"
    if abs(d) < thr:
        return "эффект не обнаружен"
    return better if d > 0 else worse


def eff_params(cfg):
    m = Model(cfg)
    tot = sum(p.numel() for p in m.parameters())
    masked = sum(int((b.o_mask == 0).sum()) for b in m.blocks if b.o_mask is not None) + \
        sum(int((b.w2_mask == 0).sum()) for b in m.blocks if b.w2_mask is not None)
    return tot - masked


def curve(r):
    return {v["step"]: v["val_ce"] for v in r["log"]["val_log"]}


def main(base="runs_ts", out="report"):
    torch.set_grad_enabled(False)
    A, Q0 = load(base, "A_s0"), load(base, "Q0_s0")
    QG = seeds(base, "QG25")
    a_ce = A["ce"]
    qs = stats(QG)
    qg_ce = qs[0]
    gap = qg_ce - a_ce
    L = ["# Раунд 7: контроли гейта, сетка, цена разрезания (TinyStories, 20k шагов, полный val)\n"]
    # ------------------------------------------------------------------ 1. gate controls
    desc = {"QG25tokengate": "обученный гейт, на входе только эмбеддинг токена",
            "QG25static": "2 из 8 групп на блок, одни и те же", "QG25position": "случайная, p по позиции в истории",
            "QG25random": "случайная, p = 0.25"}
    ctrl, cst = {}, {}
    for n, lab, _ in CTRL:
        rs = seeds(base, n)
        if rs:
            ctrl[n], cst[n] = rs, stats(rs)
    one = [lab for n, lab, _ in CTRL if n in cst and cst[n][2] == 1]
    L += ["## 1. Контроли гейта\n",
          f"Разрыв QG25 − A = {gap:+.4f} нат (QG25 — среднее {qs[2]} сидов обучения). Сравнение — по среднему по "
          "сидам обучения; SD — стандартное отклонение по сидам обучения (ddof = 1). «% разрыва» = "
          "(прогон − QG25) / (QG25 − A). У случайных масок (random, position) val CE одного прогона — среднее по 3 "
          "сидам маски при оценке. Критерий: если |разность| < 2 × SD (SD по сидам, объединённое по двум сравниваемым "
          "прогонам; если у одного из них один сид — SD второго), пишем «эффект не обнаружен», а не «эффекта нет».\n",
          "| прогон | маска | сидов обучения | val CE: среднее ± SD | по сидам | ΔCE к A | прогон − QG25 | "
          "% разрыва QG25 − A | 2 × SD | вывод |",
          "|---|---|---|---|---|---|---|---|---|---|",
          f"| QG25 | обученный гейт (состояние и контекст) | {qs[2]} | {qs[0]:.4f} ± {qs[1]:.4f} | "
          f"{' / '.join(f'{r['ce']:.4f}' for r in QG)} | {qs[0] - a_ce:+.4f} | — | — | — | — |"]
    for n, lab, _ in CTRL:
        if n not in cst:
            L.append(f"| {lab} | {desc[n]} | 0 | — | — | — | — | — | — | — |")
            continue
        m, sd, k = cst[n]
        d = m - qg_ce
        thr = 2 * pooled_sd(qs, cst[n])
        per = " / ".join(f"{r['ce']:.4f}{'ᵛ' if r['where'] == 'vast' else ''}" for r in ctrl[n])
        L.append(f"| {lab} | {desc[n]} | {k} | {m:.4f}" + (f" ± {sd:.4f}" if k > 1 else "") + f" | {per} | "
                 f"{m - a_ce:+.4f} | {d:+.4f} | {100 * d / gap:+.0f}% | {'—' if np.isnan(thr) else f'{thr:.4f}'} | "
                 f"{verdict(d, thr, 'QG25 лучше', 'QG25 хуже')} |")
    L += [f"| A | — | 1 | {a_ce:.4f} | — | — | {a_ce - qg_ce:+.4f} | −100% | — | — |",
          f"| Q0 | все группы открыты, без гейта | 1 | {Q0['ce']:.4f} | — | {Q0['ce'] - a_ce:+.4f} | "
          f"{Q0['ce'] - qg_ce:+.4f} | {100 * (Q0['ce'] - qg_ce) / gap:+.0f}% | — | — |",
          "", "ᵛ — сид обучен на vast.ai (тот же код, те же токены; проверка A_s0 на vast: полный val +0.0006 к "
              "домашнему)." + (f" Один сид: {', '.join(one)}." if one else ""), ""]
    # decomposition of the gate's gain: random -> static -> tokengate -> QG25
    steps = [("QG25random", "QG25static", "(а) предсказуемость маски: random − static", "static лучше", "static хуже"),
             ("QG25static", "QG25tokengate", "(б) сам токен: static − tokengate", "tokengate лучше", "tokengate хуже"),
             ("QG25tokengate", None, "(в) контекст: tokengate − QG25", "QG25 лучше", "QG25 хуже")]
    if "QG25random" in cst:
        W = cst["QG25random"][0] - qg_ce
        L += [f"Выигрыш обученного гейта над случайной маской W = random − QG25 = {W:+.4f} нат "
              f"({100 * W / gap:+.0f}% разрыва QG25 − A). Разложение по цепочке random → static → tokengate → QG25 "
              "(слагаемые в сумме дают W):\n",
              "| часть | разность, нат | доля W | % разрыва QG25 − A | 2 × SD | вывод |", "|---|---|---|---|---|---|"]
        for x, y, txt, bt, wt in steps:
            sx = cst.get(x)
            sy = qs if y is None else cst.get(y)
            if sx is None or sy is None:
                L.append(f"| {txt} | — | — | — | — | нет прогона |")
                continue
            d = sx[0] - sy[0]
            thr = 2 * pooled_sd(sx, sy)
            L.append(f"| {txt} | {d:+.4f} | {100 * d / W:.0f}% | {100 * d / gap:+.0f}% | "
                     f"{'—' if np.isnan(thr) else f'{thr:.4f}'} | {verdict(d, thr, bt, wt)} |")
        if "QG25position" in cst:
            d = cst["QG25random"][0] - cst["QG25position"][0]
            thr = 2 * pooled_sd(cst["QG25random"], cst["QG25position"])
            L.append(f"| для сравнения — позиция в истории: random − position | {d:+.4f} | {100 * d / W:.0f}% | "
                     f"{100 * d / gap:+.0f}% | {thr:.4f} | {verdict(d, thr, 'position лучше', 'position хуже')} |")
        L.append("")
    ms = [(f"{lab} {r['name'].rsplit('_', 1)[1]}", r["ev"]["val_ce_mask_seeds"]) for n, lab, _ in CTRL
          if n in ctrl for r in ctrl[n] if "val_ce_mask_seeds" in r["ev"]]
    if ms:
        L.append("- разброс от сида маски при оценке (размах по 3 сидам маски): " + "; ".join(
            f"{lab} {max(v) - min(v):.4f}" for lab, v in ms))
    L += [f"- SD по сидам обучения: QG25 {qs[1]:.4f}; " + "; ".join(
        f"{lab} {cst[n][1]:.4f}" for n, lab, _ in CTRL if n in cst and cst[n][2] > 1), ""]
    # checks of the new QG runs
    L += ["### Проверки новых прогонов с квантованным состоянием\n",
          "| прогон | max\\|Δx\\| закрытой | train/eval Δлогит | открыто по блокам | групп с долей < 1% / > 99% (из 64) | "
          "max доля на краях сетки |", "|---|---|---|---|---|---|"]
    extra = [load(base, n) for n in ("QGsplit25wide_s0",)]
    for r in [r for rs in ctrl.values() for r in rs] + [x for x in extra if x]:
        g, es = r["ev"]["qg"], r["ev"]["levels"]["edge_share"]
        i = int(np.argmax(es))
        L.append(f"| {r['name']} | {g['max_abs_change_closed_group']:.0g} | {g['train_eval_max_abs_logit_diff']:.2g} | "
                 f"{' '.join(f'{x:.2f}' for x in g['paid_layer'])} | {int(round(8 * sum(g['never_open_layer'])))} / "
                 f"{int(round(8 * sum(g['always_open_layer'])))} | {100 * es[i]:.2f}% ({'emb' if i == 0 else f'b{i - 1}'}) |")
    L.append("")
    # ------------------------------------------------------------------ 2. grid
    a_cur = curve(A)
    L += ["## 2. Сетка: тот же диапазон ±12.5 RMS_A, шаг в 2 и 4 раза мельче\n",
          "Отставание по ходу обучения — по кривой val из 256 окон, против A на том же шаге.\n",
          "| прогон | уровней | val CE | ΔCE к A | отставание: 6k / 10k / 14k / 20k | рост 10k → 20k | "
          "max доля на краях (состояние) |", "|---|---|---|---|---|---|---|"]
    grid = {}
    for n, lab, _ in GRID:
        r = load(base, n)
        if r is None:
            L.append(f"| {lab} | — | — | — | — | — | — |")
            continue
        grid[n] = r
        c = curve(r)
        gp = {s: c[s] - a_cur[s] for s in (6000, 10000, 14000, 20000) if s in c}
        es = r["ev"]["levels"]["edge_share"]
        i = int(np.argmax(es))
        L.append(f"| {lab} | {2 * r['cfg'].q_levels + 1} | {r['ce']:.4f} | {r['ce'] - a_ce:+.4f} | "
                 f"{' / '.join(f'{gp[s]:+.4f}' for s in sorted(gp))} | {gp.get(20000, np.nan) - gp.get(10000, np.nan):+.4f} | "
                 f"{100 * es[i]:.2f}% ({'emb' if i == 0 else f'b{i - 1}'}) |")
    L.append("")
    # ------------------------------------------------------------------ 3. cost of the cut
    S = {n: load(base, n) for n in ("Asplit_s0", "Q0split_s0", "QGsplit25_s0", "QGsplit25wide_s0")}
    L += ["## 3. Цена разрезания\n",
          "Действующие параметры — без нулей масок W_O и w2. FLOPs блока на токен — с пропуском закрытых групп по "
          "учёту раунда 5 (внимание считается всегда), в долях от A; средний контекст внимания — как в 10.8.\n",
          "| прогон | разрезан | сетка | гейт | FFN, скрытых | действующих параметров | val CE | ΔCE к A | "
          "FLOPs блока к A |", "|---|---|---|---|---|---|---|---|---|"]
    fA = None
    rows = [("A", A), ("A-split", S["Asplit_s0"]), ("Q0", Q0), ("Q0-split", S["Q0split_s0"]), ("QG25 s0", QG[0]),
            ("QG-split25", S["QGsplit25_s0"]), ("QG-split25-wide", S["QGsplit25wide_s0"])]
    ctx = savings(QG[0]["run"])["ctx"]
    for lab, r in rows:
        if r is None:
            L.append(f"| {lab} | | | | | | — | — | — |")
            continue
        cfg = r["cfg"]
        fm = flop_model(cfg, ctx)
        tot = fm["ffn"] + fm["attn"]
        fA = tot if lab == "A" else fA
        skip = savings(r["run"])["realisable_block"] if cfg.qg else 0.0
        L.append(f"| {lab} | {'да' if cfg.qg_split else 'нет'} | {'да' if cfg.quant else 'нет'} | "
                 f"{'да' if cfg.qg else 'нет'} | {cfg.ffn_hidden} | {eff_params(cfg) / 1e6:.2f}M | {r['ce']:.4f} | "
                 f"{r['ce'] - a_ce:+.4f} | {tot * (1 - skip) / fA:.2f} |")
    L.append("")
    if S["Asplit_s0"] and S["Q0split_s0"] and S["QGsplit25_s0"]:
        sp, q0s, qs = S["Asplit_s0"]["ce"], S["Q0split_s0"]["ce"], S["QGsplit25_s0"]["ce"]
        tot_s = qs - a_ce
        cut, grid_d, gate_d = sp - a_ce, q0s - sp, qs - q0s
        grid_x, gate_x = Q0["ce"] - a_ce, qg_ce - Q0["ce"]
        L += [f"Разложение отставания QG-split25 от A ({tot_s:+.4f}) — прямое, по цепочке A → A-split → Q0-split → "
              "QG-split25:\n",
              f"- разрезание (A-split − A): {cut:+.4f} ({100 * cut / tot_s:.0f}%)",
              f"- сетка (Q0-split − A-split): {grid_d:+.4f} ({100 * grid_d / tot_s:.0f}%)",
              f"- гейт (QG-split25 − Q0-split): {gate_d:+.4f} ({100 * gate_d / tot_s:.0f}%)", "",
              "Дополнительно — те же части без разрезания и их расхождение с прямыми (оценка взаимодействия):\n",
              f"- сетка без разрезания (Q0 − A): {grid_x:+.4f}; расхождение (Q0-split − A-split) − (Q0 − A): "
              f"{grid_d - grid_x:+.4f}",
              f"- гейт без разрезания (QG25 − Q0, среднее двух сидов QG25): {gate_x:+.4f}; расхождение с прямым: "
              f"{gate_d - gate_x:+.4f}",
              f"- разложение с сеткой из Q0 − A: гейт = QG-split25 − A-split − (Q0 − A) = {qs - sp - grid_x:+.4f} "
              f"против прямого {gate_d:+.4f}",
              f"- разрезание при сетке и гейте (QG-split25 − QG25): {qs - qg_ce:+.4f}; при сетке без гейта "
              f"(Q0-split − Q0): {q0s - Q0['ce']:+.4f}; без сетки и гейта (A-split − A): {cut:+.4f}"]
        if S["QGsplit25wide_s0"]:
            wd = S["QGsplit25wide_s0"]["ce"]
            L.append(f"- вернуть параметры (QG-split25 − QG-split25-wide): {qs - wd:+.4f}; QG-split25-wide − A: "
                     f"{wd - a_ce:+.4f}; wide − QG25: {wd - qg_ce:+.4f}")
        L.append("")
    open(os.path.join(out, "r7_tables.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))
    # ------------------------------------------------------------------ figures
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    def mean_curve(rs):
        cs = [curve(r) for r in rs]
        return {s: float(np.mean([c[s] for c in cs])) for s in cs[0] if all(s in c for c in cs)}

    q_mean = mean_curve(QG)
    q_lab = f"QG25 ({len(QG)} сида)"
    series = [("A", a_cur, "#52514e"), (q_lab, q_mean, "#2a78d6")] + \
        [(f"{lab} ({len(ctrl[n])} сид.)", mean_curve(ctrl[n]), col) for n, lab, col in CTRL if n in ctrl]
    for lab, c, col in series:
        st = [s for s in sorted(c) if s >= 3000]
        ax[0].plot([s / 1000 for s in st], [c[s] for s in st], color=col, lw=1.6, label=lab)
        if lab != q_lab:
            st = [s for s in st if s in q_mean]
            ax[1].plot([s / 1000 for s in st], [c[s] - q_mean[s] for s in st], color=col, lw=1.6, label=lab)
    ax[0].set_title("val CE (256 окон)", loc="left")
    ax[0].set_xlabel("шаг, тыс.")
    ax[0].legend(fontsize=8)
    ax[1].axhline(0, color=INK2, lw=1, ls=":")
    ax[1].set_title("разность с QG25 (ΔCE; выше нуля — хуже QG25)", loc="left")
    ax[1].set_xlabel("шаг, тыс.")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "r7_val.png"), dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.4))
    for n, lab, col in GRID:
        if n in grid:
            c = curve(grid[n])
            st = [s for s in sorted(c) if s >= 2000 and s in a_cur]
            ax.plot([s / 1000 for s in st], [c[s] - a_cur[s] for s in st], color=col, lw=1.6, label=lab)
    ax.axhline(0, color=INK2, lw=1, ls=":")
    ax.set_title("отставание Q0 от A по ходу обучения (ΔCE, 256 окон)", loc="left")
    ax.set_xlabel("шаг, тыс.")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "r7_grid.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main(*sys.argv[1:])
