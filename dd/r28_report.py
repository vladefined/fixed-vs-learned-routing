"""Round 28 report, stage 0: report/r28_tables.md (what keeps the experts of r21 alike: perlayer s0, hash s0)."""
import json
import os

import numpy as np

from .r28_stage0 import CKPT, OUT

E = 8
BL = ("5", "6", "7", "8")


def jl(name):
    p = os.path.join(OUT, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def wmean(vals, w):
    vals, w = np.asarray(vals, float), np.asarray(w, float)
    ok = ~np.isnan(vals)
    return float((vals[ok] * w[ok]).sum() / max(1e-12, w[ok].sum()))


def fmt(v, se=None, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:+.{nd}f}" + (f" ± {2 * se:.{nd}f}" if se is not None else "")


def swap_part(ck, L):
    s = jl(f"swap_{ck}.json")
    if not s:
        return
    load = s["load"]
    L.append(f"\n### Подмена экспертов: {ck}\n")
    L.append("Рост CE на токенах эксперта i (строка), если их пропустить через эксперта k (столбец) того же блока; "
             "± 2 SE по окнам. «все» — во всех MoE-блоках сразу (токены эксперта i своего блока → эксперт k своего "
             "блока; CE на токенах, бывших у i хотя бы в одном блоке). Проверка: прогон без подмены из кэша даёт рост "
             f"{max(abs(r[0]) for r in s['resume_check']):.1e}.\n")
    summ = []
    for key, blk in s["blocks"].items():
        lab = "все" if key == "all" else BL[int(key)]
        L.append(f"\n**Блок {lab}**\n")
        L.append("| i \\ k | доля токенов | " + " | ".join(str(k) for k in range(E)) + " | FFN A-mix | FFN dense-cont | ноль |")
        L.append("|---|---|" + "---|" * (E + 3))
        offd, am, dn, ze = [], [], [], []
        offd_se, dn_se, ze_se = [], [], []
        w = []
        for i in range(E):
            r = blk[str(i)]
            ld = load[int(key)][i] if key != "all" else float("nan")
            cells = []
            for k in range(E):
                if k == i:
                    cells.append("·")
                else:
                    cells.append(fmt(*r[str(k)]))
            L.append(f"| {i} | {ld:.3f} | " + " | ".join(cells) + f" | {fmt(*r['amix'])} | {fmt(*r['dense'])} | {fmt(*r['zero'])} |")
            ww = ld if key != "all" else 1.0
            offd.append(np.nanmean([r[str(k)][0] for k in range(E) if k != i]))
            offd_se.append(np.nanmean([r[str(k)][1] for k in range(E) if k != i]))   # pairs of a row share tokens
            am.append(r["amix"][0])
            dn.append(r["dense"][0])
            dn_se.append(r["dense"][1])
            ze.append(r["zero"][0])
            ze_se.append(r["zero"][1])
            w.append(ww)
        ww_ = np.asarray(w, float) / np.sum(w)
        wse = lambda se: float(np.sqrt(np.nansum((ww_ * np.asarray(se)) ** 2)))          # noqa: E731
        summ.append((lab, wmean(offd, w), wmean(am, w), wmean(dn, w), wmean(ze, w),
                     float(np.nanmin([blk[str(i)][str(k)][0] for i in range(E) for k in range(E) if k != i])),
                     float(np.nanmax([blk[str(i)][str(k)][0] for i in range(E) for k in range(E) if k != i])),
                     wse(offd_se), wse(dn_se), wse(ze_se)))
    L.append(f"\n**Сводка {ck}** (среднее по строкам с весом доли токенов; «все» — без весов)\n")
    L.append("Специализация = (чужой − свой) / (ноль − свой); свой — сама модель, её рост 0, так что это чужой / ноль; то "
             "же для FFN dense-cont вместо чужого. ± 2 SE отношения — дельта-метод без учёта корреляции числителя и "
             "знаменателя (с запасом).\n")
    L.append("| блок | чужой эксперт (среднее) | мин … макс по парам | FFN A-mix | FFN dense-cont | ноль | специализация: "
             "чужой / ноль | dense-cont / ноль |")
    L.append("|---|---|---|---|---|---|---|---|")
    for lab, o, a, d, z, mn, mx, so, sd, sz in summ:
        ro, rd = o / z, d / z
        so_r = abs(ro) * np.sqrt((so / o) ** 2 + (sz / z) ** 2)
        sd_r = abs(rd) * np.sqrt((sd / d) ** 2 + (sz / z) ** 2)
        L.append(f"| {lab} | {o:+.4f} | {mn:+.4f} … {mx:+.4f} | {a:+.4f} | {d:+.4f} | {z:+.4f} | {ro:.3f} ± {2 * so_r:.3f} | "
                 f"{rd:.3f} ± {2 * sd_r:.3f} |")
    return summ


def fit_part(ck, L):
    f = jl(f"fit_{ck}.json")
    s = jl(f"swap_{ck}.json")
    if not f or not s:
        return
    load = s["load"]
    L.append(f"\n### Подмена с подбором выхода: {ck}\n")
    L.append("Gate и up эксперта k заморожены, down подобран гребневой регрессией на токенах эксперта i (калибровка "
             "2M токенов train, каждое 5-е окно отложено для λ) под выход эксперта i; рост CE на val. Снято = "
             "(подмена − подмена с подбором) / подмена, по парам i ≠ k.\n")
    L.append("| блок | чужой как есть | чужой + подбор down | снято разницы | свой + подбор (проверка) | gate/up A-mix + подбор | "
             "случайные gate/up + подбор | объяснено выхода i: свой / чужие / A-mix / случайные |")
    L.append("|---|---|---|---|---|---|---|---|")
    for key, blk in f["blocks"].items():
        j = int(key)
        w = load[j]
        sw = s["blocks"][key]
        raw, fitd, own, am, rnd, ex_own, ex_f, ex_a, ex_r = [], [], [], [], [], [], [], [], []
        for i in range(E):
            r = blk[str(i)]
            raw.append(np.nanmean([sw[str(i)][str(k)][0] for k in range(E) if k != i]))
            fitd.append(np.nanmean([r[str(k)]["rise"][0] for k in range(E) if k != i]))
            own.append(r[str(i)]["rise"][0])
            am.append(r["amix"]["rise"][0])
            rnd.append(r["rand"]["rise"][0])
            ex_own.append(r[str(i)]["explained"])
            ex_f.append(np.mean([r[str(k)]["explained"] for k in range(E) if k != i]))
            ex_a.append(r["amix"]["explained"])
            ex_r.append(r["rand"]["explained"])
        a, b = wmean(raw, w), wmean(fitd, w)
        L.append(f"| {BL[j]} | {a:+.4f} | {b:+.4f} | {(a - b) / a if a > 0 else float('nan'):.0%} | {wmean(own, w):+.4f} | "
                 f"{wmean(am, w):+.4f} | {wmean(rnd, w):+.4f} | {wmean(ex_own, w):.2f} / {wmean(ex_f, w):.2f} / "
                 f"{wmean(ex_a, w):.2f} / {wmean(ex_r, w):.2f} |")


def div_part(ck, L):
    d = jl(f"div_{ck}.json")
    if not d:
        return
    L.append(f"\n### Расхождение весов: {ck}\n")
    L.append("Вес эксперта — w1, w3, w2 подряд. Все эксперты стартовали копиями FFN A-mix, поэтому сравниваются и сами "
             "веса, и сдвиги от A-mix (эксперт − A-mix). Относительная норма — к норме FFN A-mix. Среднее по парам i ≠ k.\n")
    L.append("| блок | косинус весов | ‖разность‖ / ‖A-mix‖ | ‖сдвиг‖ / ‖A-mix‖ (среднее) | косинус сдвигов между экспертами "
             "(мин … макс) | косинус сдвига со сдвигом dense-cont | ‖сдвиг dense-cont‖ / ‖A-mix‖ |")
    L.append("|---|---|---|---|---|---|---|")
    for key in sorted(d, key=int):
        r = d[key]
        off = [(a, b) for a in range(E) for b in range(E) if a != b]
        cw = np.mean([r["cos_w"][a][b] for a, b in off])
        rd = np.mean([r["rel_diff"][a][b] for a, b in off])
        cu = [r["cos_upd"][a][b] for a, b in off]
        L.append(f"| {BL[int(key)]} | {cw:.4f} | {rd:.3f} | {np.mean(r['upd_rel']):.3f} | {np.mean(cu):.3f} ({min(cu):.3f} … "
                 f"{max(cu):.3f}) | {np.mean(r['cos_upd_dense']):.3f} | {r['dense_upd_rel']:.3f} |")


def grad_part(ck, L):
    g = jl(f"grad_{ck}.json")
    if not g:
        return
    sh = np.asarray(g["reader_topic_share"])
    L.append(f"\n### Чужой градиент: {ck}\n")
    L.append("Градиент весов эксперта от потери на позициях, которые он обработал (свой), и от потерь на других "
             "позициях (чужой: всё, что доходит через K и V внимания следующих блоков); сумма по 256 окнам val, fp32. "
             "Тема читающего токена — его эксперт в блоке 5. Доли тем — ‖чужой градиент от читающих темы g‖ / сумма таких "
             "норм по g. Сосредоточенность: доля двух главных тем и эффективное число тем (exp энтропии долей); для "
             "сравнения — доли токенов по темам: " + " ".join(f"{x:.2f}" for x in sh) +
             f" (эффективное число {np.exp(-(sh[sh > 0] * np.log(sh[sh > 0])).sum()):.1f}).\n")
    L.append("| блок | ‖чужой‖ / ‖полный‖ (среднее, мин … макс) | косинус своего и чужого | доля двух главных тем | "
             "эффективное число тем | доля своей темы (блок 5) | сумма по темам / чужой |")
    L.append("|---|---|---|---|---|---|---|")
    for key in ("0", "1", "2", "3"):
        r = g[key]
        fr = [f / max(1e-30, u) for f, u in zip(r["foreign"], r["full"])]
        bt = np.asarray(r["by_topic"])
        shares = bt / np.maximum(bt.sum(1, keepdims=True), 1e-30)
        top2 = np.sort(shares, 1)[:, -2:].sum(1)
        effn = np.exp(-(np.where(shares > 0, shares * np.log(np.maximum(shares, 1e-30)), 0)).sum(1))
        own_t = np.mean([shares[i, i] for i in range(E)]) if key == "0" else float("nan")
        L.append(f"| {BL[int(key)]} | {np.mean(fr):.2f} ({min(fr):.2f} … {max(fr):.2f}) | {np.mean(r['cos_own_foreign']):+.2f} | "
                 f"{np.mean(top2):.2f} | {np.mean(effn):.1f} | " + (f"{own_t:.2f}" if key == "0" else "—") +
                 (f" | {np.mean(r['sum_topics_vs_foreign']):.2f} |" if np.mean(fr) > 1e-3 else " | — (чужого нет: дальше нет внимания) |"))
    for key in ("0", "1", "2"):
        bt = np.asarray(g[key]["by_topic"])
        shares = bt / np.maximum(bt.sum(1, keepdims=True), 1e-30)
        L.append(f"\nБлок {BL[int(key)]}: доли тем читающих (строка — эксперт, столбец — тема читающего)\n")
        L.append("| эксперт \\ тема | " + " | ".join(str(t) for t in range(E)) + " |")
        L.append("|---|" + "---|" * E)
        for i in range(E):
            L.append(f"| {i} | " + " | ".join(f"{x:.2f}" for x in shares[i]) + " |")


# ------------------------------------------------------------------------------------------------ stage 0b (OLMoE)
RANKS = (9, 12, 16, 24, 32, 48, 64)


def report0b(L):
    sw = jl("olmoe_swap.json")
    rk = jl("olmoe_rank.json")
    ft = jl("olmoe_fit.json")
    gr = jl("olmoe_grad.json")
    small = {ck: jl(f"swap_{ck}.json") for ck in CKPT}
    L.append("\n# Этап 0б: то же на OLMoE-1B-7B (обвязка r26, тот же val, без обучения)\n")
    L.append("Оговорка: в OLMoE у токена 8 экспертов из 64 и выход слоя — их сумма с весами маршрутизатора; подмена "
             "меняет одного из восьми (около 1/8 выхода, с его весом), «ноль» убирает только его. В маленькой модели "
             "эксперт один и весь FFN, «ноль» убирает весь FFN. Поэтому подмена / ноль в OLMoE и в маленькой модели — "
             "меры одного смысла, но разного масштаба.\n")
    if sw:
        L.append("\n## Подмена одного из 8 (слот случайный), 256 окон val\n")
        L.append("| слой | случайный невыбранный | 9-й | самый далёкий | ноль этого слота | оставшиеся 7 (перенормированы) | "
                 "подмена / ноль: случайный / 9-й / далёкий |")
        L.append("|---|---|---|---|---|---|---|")
        for i in range(16):
            r = {v: sw.get(f"{i}/{v}") for v in ("random", "ninth", "far", "zero", "rest7")}
            if not all(r.values()):
                continue
            z = r["zero"][0]
            L.append(f"| {i} | " + " | ".join(fmt(*r[v], nd=4) for v in ("random", "ninth", "far", "zero", "rest7")) +
                     f" | {r['random'][0] / z:.2f} / {r['ninth'][0] / z:.2f} / {r['far'][0] / z:.2f} |")
    if rk:
        L.append(f"\n## Подмена по рангу маршрутизатора (96 окон val; проверка без подмены: {rk.get('check', ['—'])[0]})\n")
        for slot, nm in (("top", "заменяется эксперт с наибольшим весом"), ("bottom", "заменяется эксперт с наименьшим весом")):
            for norm in ("plain", "norm"):
                L.append(f"\n**{nm}; {'выход как есть' if norm == 'plain' else 'норма выхода подгоняется под заменённого'}** — "
                         "рост CE / рост при нуле этого слота\n")
                L.append("| слой | ноль этого слота | " + " | ".join(f"ранг {r}" for r in RANKS) + " | случайный из 56 |")
                L.append("|---|---|" + "---|" * (len(RANKS) + 1))
                for i in range(16):
                    z = rk.get(f"{i}/{slot}/plain/zero")
                    vals = [rk.get(f"{i}/{slot}/{norm}/{r}") for r in list(RANKS) + ["random"]]
                    if not z or not all(vals):
                        continue
                    L.append(f"| {i} | {z[0]:+.4f} | " + " | ".join(f"{v[0] / z[0]:.2f}" for v in vals) + " |")
        plot_rank(rk)
        L.append("\n![ранг → рост CE](r28_olmoe_rank.png)")
        ts = jl("olmoe_tokshare.json") or {}
        L.append("\n**Абсолютные числа: главный эксперт токена (наибольший вес)** — рост CE, нат на токен ± 2 SE (96 окон); "
                 "доля токенов, у которых потеря меняется больше чем на 0.1 нат (в скобках: рост / падение больше 0.1)\n")
        L.append("| слой | ноль главного эксперта | подмена на ранг 9 | доля токенов |Δ| > 0.1: ноль | "
                 "доля токенов |Δ| > 0.1: ранг 9 |")
        L.append("|---|---|---|---|---|")
        for i in range(16):
            z, r9 = rk.get(f"{i}/top/plain/zero"), rk.get(f"{i}/top/plain/9")
            if not z or not r9:
                continue
            tz, tr = ts.get(f"{i}/zero"), ts.get(f"{i}/rank9")
            f = (lambda t: f"{t['any01']:.3f} ({t['up01']:.3f} / {t['down01']:.3f})" if t else "считается")  # noqa: E731
            L.append(f"| {i} | {fmt(*z, nd=4)} | {fmt(*r9, nd=4)} | {f(tz)} | {f(tr)} |")
    if ft:
        L.append("\n## Подмена с подбором выхода (слои 0, 4, 8, 12, 15; 16 экспертов на слой; рост CE на токенах, где "
                 "эксперт выбран)\n")
        L.append("Партнёр «ближайший» — чаще всех выбирается маршрутизатором вместе с экспертом; «случайный» — любой "
                 "другой. Подбор: gate и up партнёра заморожены, down подобран гребневой регрессией на калибровке r26 "
                 "под выход эксперта. Снято = (как есть − с подбором) / как есть.\n")
        L.append("| слой | ближайший: как есть / с подбором / снято | случайный: как есть / с подбором / снято | свой + подбор "
                 "(проверка) | случайные gate/up + подбор | ноль | объяснено выхода: ближайший / случайный / свой / "
                 "случайные gate/up |")
        L.append("|---|---|---|---|---|---|---|")
        for i in (0, 4, 8, 12, 15):
            rows = [v for k, v in ft.items() if k.startswith(f"{i}/")]
            if not rows:
                continue
            m = lambda key: float(np.mean([r[key][0] for r in rows]))                     # noqa: E731
            ex = lambda key: float(np.mean([r[key] for r in rows]))                      # noqa: E731
            cells = []
            for nm in ("near", "random"):
                a, b = m(f"raw_{nm}"), m(f"fit_{nm}")
                cells.append(f"{a:+.4f} / {b:+.4f} / {(a - b) / a:.0%}")
            L.append(f"| {i} | " + " | ".join(cells) + f" | {m('fit_own'):+.4f} | {m('fit_randin'):+.4f} | {m('zero'):+.4f} | "
                     f"{ex('explained_near'):.2f} / {ex('explained_random'):.2f} / {ex('explained_own'):.2f} / "
                     f"{ex('explained_randin'):.2f} |")
    tk = jl("olmoe_topk.json")
    if tk and "dense" in tk:
        L.append("\n## Меньше экспертов на токен (полный val r26, все слои сразу)\n")
        L.append("Каждый слой оставляет из своих 8 экспертов первые k по весу (k = 7…4) или те, у кого вес ≥ t (первый "
                 "остаётся всегда). «как есть» — веса оставшихся не меняются; «перенорм.» — масштабируются так, чтобы сумма "
                 "была как у исходных 8 (сам OLMoE топ-8 не нормирует). Разница к полным 8 — парная по окнам, ± 2 SE.\n")
        L.append("| вариант | экспертов на токен: среднее (мин–макс по слоям) | val CE | Δ к 8 экспертам ± 2 SE |")
        L.append("|---|---|---|---|")
        d0 = tk["dense"]
        n0, dn = np.asarray(d0["num"], float), np.asarray(d0["den"], float)
        for name, r in tk.items():
            nm = np.asarray(r["num"], float)
            if nm.shape != n0.shape:
                continue
            dd = nm - n0
            rr = dd.sum() / dn.sum()
            se = float(np.sqrt(((dd - rr * dn) ** 2).sum() / (len(dn) - 1) / len(dn)) / dn.mean())
            lab = name.replace("_raw", ", как есть").replace("_renorm", ", перенорм.").replace("thr", "вес ≥ ")
            lab = "8 (без изменений)" if name == "dense" else lab.replace("k", "топ-") if name.startswith("k") else lab
            epl = r["experts_per_layer"]
            L.append(f"| {lab} | {r['experts_mean']:.2f} ({min(epl):.2f}–{max(epl):.2f}) | {r['ce']:.4f} | "
                     f"{'—' if name == 'dense' else fmt(float(rr), se, nd=4)} |")
    if gr:
        L.append("\n## Чужой градиент по выходу эксперта (окна 256 токенов)\n")
        L.append("Градиент по выходу эксперта = вес эксперта × градиент по выходу MoE-слоя, поэтому доля «свой / чужой» "
                 "одна для всех экспертов токена. Доля чужого = √Σ|чужой|² / √Σ|полный|².\n")
        L.append("| слой OLMoE | доля чужого | косинус своего и чужого |")
        L.append("|---|---|---|")
        for i, l in enumerate(gr["layers"]):
            L.append(f"| {i} | {l['share_energy']:.2f} | {l['cos']:+.3f} |")
    # side by side
    L.append("\n## Рядом: маленькая модель (этап 0) и OLMoE (этап 0б)\n")
    L.append("Маленькая модель: 8 блоков, эксперты в блоках 5–8, один эксперт на токен = весь FFN. OLMoE: 16 слоёв, 8 из "
             "64 на токен. Сравнивать по числу слоёв после эксперта (через сколько блоков внимания его выход читают другие "
             "позиции) или по относительной глубине.\n")
    L.append("| модель / блок | слоёв после | подмена / ноль | снято подбором down | доля чужого градиента (по выходу) |")
    L.append("|---|---|---|---|---|")
    for ck in CKPT:
        s0 = small.get(ck)
        og = jl(f"ograd_{ck}_w256.json")
        fi = jl(f"fit_{ck}.json")
        if not s0:
            continue
        for j in range(4):
            blk = s0["blocks"][str(j)]
            w = s0["load"][j]
            fo = np.array([np.nanmean([blk[str(i)][str(k)][0] for k in range(E) if k != i]) for i in range(E)])
            ze = np.array([blk[str(i)]["zero"][0] for i in range(E)])
            spec = wmean(fo, w) / wmean(ze, w)
            share = "—"
            if fi:
                fb = fi["blocks"][str(j)]
                raw = wmean(fo, w)
                fit = wmean([np.nanmean([fb[str(i)][str(k)]["rise"][0] for k in range(E) if k != i]) for i in range(E)], w)
                share = f"{(raw - fit) / raw:.0%}"
            gs = f"{og['blocks'][j]['share_energy']:.2f}" if og else "—"
            L.append(f"| {ck}, блок {BL[j]} | {3 - j} | {spec:.2f} (чужой эксперт = весь FFN) | {share} | {gs} |")
    if sw and gr:
        for i in range(16):
            r, z = sw.get(f"{i}/random"), sw.get(f"{i}/zero")
            if not r or not z:
                continue
            share = "—"
            if ft:
                rows = [v for k, v in ft.items() if k.startswith(f"{i}/")]
                if rows:
                    a = float(np.mean([x["raw_random"][0] for x in rows]))
                    b = float(np.mean([x["fit_random"][0] for x in rows]))
                    share = f"{(a - b) / a:.0%}"
            L.append(f"| OLMoE, слой {i} | {15 - i} | {r[0] / z[0]:.2f} (один из 8) | {share} | {gr['layers'][i]['share_energy']:.2f} |")


def plot_rank(rk):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surf, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=110, sharey=True)
    fig.patch.set_facecolor(surf)
    for ax, norm, title in zip(axs, ("plain", "norm"), ("выход как есть", "норма выхода как у заменённого")):
        ax.set_facecolor(surf)
        for c, i in zip(cols, (0, 4, 8, 12, 15)):
            z = rk.get(f"{i}/top/plain/zero")
            ys = [rk.get(f"{i}/top/{norm}/{r}") for r in RANKS]
            if not z or not all(ys):
                continue
            ax.plot(RANKS, [y[0] / z[0] for y in ys], "-o", color=c, lw=2, ms=6, mec=surf, mew=1.5, label=f"слой {i}")
        ax.axhline(1.0, color=ink2, lw=1, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks(RANKS)
        ax.set_xticklabels([str(r) for r in RANKS])
        ax.set_title(f"OLMoE: вместо эксперта с наибольшим весом — ранг r; {title}", color=ink, fontsize=10, loc="left")
        ax.set_xlabel("ранг заменителя по маршрутизатору", color=ink2)
        ax.grid(True, color=grid, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=ink2)
    axs[0].set_ylabel("рост CE / рост при нуле этого эксперта", color=ink2)
    axs[0].legend(frameon=False, fontsize=9, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(os.path.join("report", "r28_olmoe_rank.png"), facecolor=surf)


def report0():
    L = ["# Раунд 28, этап 0: что мешает экспертам различаться (r21: perlayer s0, hash s0, без обучения)\n"]
    L.append("Модель A смеси r21 (d 512, 8 блоков, эксперты в блоках 5–8, 8 экспертов, каждый стартовал копией FFN A-mix "
             "своего блока; 10k шагов). 1024 окна val по 512 токенов для подмены; калибровка — 2M токенов train.\n")
    summs = {}
    for ck in CKPT:
        L.append(f"\n## {ck}\n")
        summs[ck] = swap_part(ck, L)
        fit_part(ck, L)
        div_part(ck, L)
        grad_part(ck, L)
    L.append("\n## Блок 8 полностью (последний MoE-блок, после него нет внимания)\n")
    L.append("Рост CE на токенах эксперта i, среднее по строкам с весом доли токенов, ± 2 SE.\n")
    L.append("| модель | чужой эксперт (мин … макс по парам) | FFN A-mix | FFN dense-cont | ноль | чужой / ноль | "
             "dense-cont / ноль |")
    L.append("|---|---|---|---|---|---|---|")
    for ck, sm in summs.items():
        for lab, o, a, d, z, mn, mx, so, sd, sz in (sm or []):
            if lab == "8":
                L.append(f"| {ck} | {o:+.4f} ± {2 * so:.4f} ({mn:+.4f} … {mx:+.4f}) | {a:+.4f} | {d:+.4f} ± {2 * sd:.4f} | "
                         f"{z:+.4f} ± {2 * sz:.4f} | {o / z:.3f} | {d / z:.3f} |")
    report0b(L)
    os.makedirs("report", exist_ok=True)
    open(os.path.join("report", "r28_tables.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
