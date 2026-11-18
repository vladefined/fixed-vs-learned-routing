"""Round 21 (the user's block «r20»): tables and plots -> report/r21_tables.md, report/r21_drift.png, report/r21_loads.png.
Reads runs_vast/r21_books/{probe,books,amix_analysis}.json, runs_vast/r21_gate/gate.json, runs_vast/p300_*/log.json,
runs_vast/mix_<v>[_s1]/{log,analysis}.json.
The 2 x SD rule (as in the earlier rounds): SD over training seeds, pooled over the two compared variants; when one of
them has a single seed, the other's SD; |difference| < 2 SD -> "эффект не обнаружен"; both single-seed -> no verdict."""
import json
import os

import numpy as np

OUT = "runs_vast"
SRC = ("wiki", "code", "ts")
SRC_RU = ("Википедия", "код", "TinyStories")
TYPES = ("пробелы", "пунктуация", "числа", "служебные", "имена", "начало слова", "продолжение слова")
V = [("dense", "dense-cont"), ("learned", "MoE-learned"), ("hash", "MoE-hash"), ("rvqsem", "MoE-rvq-semantic"),
     ("rvqrand", "MoE-rvq-random"), ("rvqshared", "MoE-rvq-shared"), ("perlayer", "MoE-perlayer"),
     ("frozen", "MoE-rvq-semantic-frozen"), ("rvqsem_b0", "MoE-rvq-semantic, β = 0"),
     ("rvqsemw", "MoE-rvq-semantic-weighted")]
NAME = dict(V)
PAIRS = [  # (first, second, question): difference = CE(second) - CE(first)
    ("dense", "hash", "(а) хеш против плотной"), ("dense", "rvqsem", "(а) путь (RVQ) против плотной"),
    ("dense", "perlayer", "(а) per-layer против плотной"), ("dense", "learned", "(а) обучаемый против плотной"),
    ("hash", "rvqsem", "(а) хеш → путь"), ("rvqsem", "perlayer", "(а) путь → per-layer"),
    ("hash", "perlayer", "(а) хеш → per-layer"), ("rvqrand", "perlayer", "(а) путь со случайными таблицами → per-layer"),
    ("hash", "rvqrand", "(а) хеш → путь со случайными таблицами"),
    ("perlayer", "learned", "(а) per-layer → обучаемый"), ("hash", "learned", "(а) хеш → обучаемый"),
    ("rvqrand", "rvqsem", "(б) смысловые таблицы против случайных"),
    ("rvqsem", "rvqshared", "общий эксперт"), ("rvqsem", "frozen", "заморозка начала"),
    ("rvqsem", "rvqsem_b0", "без штрафа близости (β = 0)"), ("rvqsem", "dense", "плотная против пути"),
    ("rvqrand", "rvqsemw", "(б) взвешенные смысловые таблицы против случайных"),
    ("rvqsem", "rvqsemw", "(б) взвешенные смысловые против вырожденных смысловых")]
DPAIRS = [  # drift @10k: (variant, key, variant, key, question); difference = second - first, in % of tokens
    ("rvqsem", "codes_any_vs0", "perlayer", "plc_any_vs0", "(г) свои коды per-layer против кодов RVQ варианта 4"),
    ("perlayer", "codes_any_vs0", "perlayer", "plc_any_vs0", "(г) per-layer: свои коды против кодов RVQ в той же модели"),
    ("dense", "codes_any_vs0", "perlayer", "plc_any_vs0", "(г) свои коды per-layer против кодов RVQ у dense-cont"),
    ("rvqsem", "eid_any_vs0", "perlayer", "eid_any_vs0", "(г) свои эксперты: per-layer против варианта 4"),
    ("rvqrand", "eid_any_vs0", "perlayer", "eid_any_vs0", "(г) свои эксперты: per-layer против rvq-random"),
    ("dense", "codes_any_vs0", "rvqsem", "codes_any_vs0", "коды RVQ: вариант 4 (β 1e-4) против dense-cont"),
    ("rvqsem_b0", "codes_any_vs0", "rvqsem", "codes_any_vs0", "коды RVQ: штраф β 1e-4 против β = 0"),
    ("rvqsem", "codes_any_vs0", "frozen", "codes_any_vs0", "коды RVQ: заморозка блоков 1..N"),
    ("learned", "eid_any_vs0", "hash", "eid_any_vs0", "свои эксперты: hash против learned")]


def jl(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def run(v, seed=0):
    d = os.path.join(OUT, f"mix_{v}" + (f"_s{seed}" if seed else ""))
    a, lg = jl(os.path.join(d, "analysis.json")), jl(os.path.join(d, "log.json"))
    if not (a or lg):
        return None
    if a:
        ce, src, full = a["val_ce"], [a["val_ce_src"][s] for s in SRC], True
    else:                                                     # analysis missing: 256-window val of the run
        ce, src, full = lg["result"]["val_ce"], lg["result"]["val_ce_src"], False
    return dict(ce=ce, src=src, full=full, a=a, log=lg, res=lg["result"] if lg else None)


def seeds(v):
    return [r for r in (run(v, s) for s in (0, 1)) if r]


def stats(rs, k=None):
    x = np.array([r["ce"] if k is None else r["src"][k] for r in rs])
    return float(x.mean()), (float(x.std(ddof=1)) if len(x) > 1 else float("nan")), len(x)


def pooled_sd(s1, s2):
    (_, a, n), (_, b, m) = s1, s2
    if n < 2 and m < 2:
        return float("nan")
    if n < 2 or m < 2:
        return float(b if n < 2 else a)
    return float(np.sqrt(((n - 1) * a ** 2 + (m - 1) * b ** 2) / (n + m - 2)))


def drift_at(v, key, step=10000):
    out = []
    for r in seeds(v):
        if r["log"]:
            dr = {d["step"]: d for d in r["log"]["drift"]}
            if step in dr and key in dr[step]:
                out.append(100 * dr[step][key])
    x = np.array(out)
    return (float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else float("nan"), len(x)) if len(x) else None


def verdict_drift(d, thr):
    if np.isnan(thr):
        return "один сид у обоих — вывода нет"
    if abs(d) < thr:
        return "эффект не обнаружен"
    return "у второго больше" if d > 0 else "у второго меньше"


def verdict(d, thr):
    """d = CE(second) - CE(first)"""
    if np.isnan(thr):
        return "один сид у обоих — разброс не оценить, вывода нет"
    if abs(d) < thr:
        return "эффект не обнаружен"
    return "второй лучше" if d < 0 else "второй хуже"


def f4(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def report(out="report/r21_tables.md"):
    L = ["# Блок r20 (в коде r21): эксперты без обучаемого гейта", ""]
    probe = jl(os.path.join(OUT, "r21_books", "probe.json"))
    books = jl(os.path.join(OUT, "r21_books", "books.json"))
    amix = jl(os.path.join(OUT, "r21_books", "amix_analysis.json"))
    gate = jl(os.path.join(OUT, "r21_gate", "gate.json"))
    if probe:
        L += [f"## Проба: где принимать решение ({probe['windows']} окон val, {probe['tokens']} токенов; "
              f"k-means на 64 кода по RMS-нормированному состоянию)", "",
              "| после блока | NMI(код, источник) | NMI(код, тип токена) | использовано кодов |", "|---|---|---|---|"]
        for l, b in sorted(probe["blocks"].items(), key=lambda t: int(t[0])):
            mark = " ← N" if int(l) == probe["N"] else ""
            L.append(f"| {l}{mark} | {b['nmi_source']:.3f} | {b['nmi_type']:.3f} | {b['used']} |")
        L += ["", f"Правило: {probe['rule']}. N = {probe['N']}, MoE-блоки {probe['N'] + 1}..8 (M = {probe['M']}). "
              f"NMI(тип, источник) = {probe['nmi_type_vs_source']:.3f}.", ""]
    if books:
        L += [f"## Книги (A-mix, {books['book_tokens'] / 1e6:.2f}M токенов train, не обновляются)", "",
              "| ступень RVQ → блок | отн. ошибка после ступени | кодов | смысловая таблица: MaxVio / энтропия, бит "
              "| случайная таблица: MaxVio | размеры сверхкластеров (кодов) |", "|---|---|---|---|---|---|"]
        for s in books["stages"]:
            L.append(f"| {s['stage']} → {books['N'] + s['stage']} | {s['rel_err_after']:.3f} | {s['used']} | "
                     f"{s['sem']['maxvio']:.2f} / {s['sem']['entropy_bits']:.2f} | {s['rand']['maxvio']:.3f} | "
                     f"{' '.join(map(str, s['sem_sizes']))} |")
        L += ["", f"Штраф близости на старте |u − Σ книга|² = {books['penalty_at_init']:.1f} на токен "
              f"(|u|² = 512).", "",
              "| блок | per-layer: отн. ошибка книги | per-layer: MaxVio / энтропия | хеш: MaxVio |", "|---|---|---|---|"]
        for p, h in zip(books["perlayer"], books["hash"]):
            L.append(f"| {p['block']} | {p['rel_err']:.3f} | {p['sem']['maxvio']:.2f} / {p['sem']['entropy_bits']:.2f} | "
                     f"{h['maxvio']:.3f} |")
        L.append("")
    if gate:
        L += ["## Пробы на 300 шагов и решение по β", "",
              f"β основных прогонов = {gate['beta']:g} ({gate['beta_why']}). Проверки пройдены: "
              f"{'да' if gate['ok'] else 'НЕТ'}.", "",
              "| проба | val CE @300 | − dense | CE на шаге 0 − A-mix | с/шаг |", "|---|---|---|---|---|"]
        for n, c in gate["checks"].items():
            if "val_ce" in c:
                L.append(f"| {n} | {c['val_ce']:.4f} | {c.get('vs_dense', float('nan')):+.4f} | {c['init_diff']:+.1e} | "
                         f"{c['s_per_step']:.3f} |")
        L.append("")
    # main table
    R = {v: seeds(v) for v, _ in V}
    st = {v: stats(rs) for v, rs in R.items() if rs}
    sts = {v: [stats(rs, k) for k in range(3)] for v, rs in R.items() if rs}
    L += ["## Качество: val CE (весь val, 8789 окон по 512; по источнику предсказываемого токена)", "",
          "| вариант | сидов | val CE (по сидам) | среднее ± SD | " + " | ".join(SRC_RU) + " | − dense-cont | с/шаг |",
          "|---|---|---|---|---|---|---|---|---|"]
    if amix:
        L.append(f"| A-mix (20k, старт этапа 2) | 1 | {amix['val_ce']:.4f} | | " +
                 " | ".join(f"{amix['val_ce_src'][s]:.4f}" for s in SRC) + " | | |")
    dn = st.get("dense")
    for v, nm in V:
        if v not in st:
            continue
        m, sd, n = st[v]
        per = " / ".join(f"{r['ce']:.4f}" + ("" if r["full"] else "*") for r in R[v])
        srcs = " | ".join(f"{sts[v][k][0]:.4f}" for k in range(3))
        sps = np.mean([r["res"].get("s_per_step_steady", r["res"]["s_per_step"]) for r in R[v] if r["res"]])
        L.append(f"| {nm} | {n} | {per} | {m:.4f}" + (f" ± {sd:.4f}" if n > 1 else "") + f" | {srcs} | " +
                 (f"{m - dn[0]:+.4f}" if dn and v != "dense" else "") + f" | {sps:.3f} |")
    L += ["", "\\* нет анализа: val CE на 256 окнах прогона.", ""]
    # differences with the 2 SD rule
    L += ["## Разницы и правило 2 × SD", "",
          "Разность = CE(второй) − CE(первый); < 0 — второй лучше. 2 × SD: SD по сидам, объединённое по паре "
          "(если у одного варианта один сид — SD другого).", "",
          "| вопрос | первый → второй | разность (всего) | 2 × SD | вывод | " +
          " | ".join(f"{s}: разность / вывод" for s in SRC_RU) + " |", "|---|---|---|---|---|---|---|---|"]
    for a, b, q in PAIRS:
        if a not in st or b not in st:
            continue
        d = st[b][0] - st[a][0]
        thr = 2 * pooled_sd(st[a], st[b])
        cells = []
        for k in range(3):
            dk = sts[b][k][0] - sts[a][k][0]
            tk = 2 * pooled_sd(sts[a][k], sts[b][k])
            vk = verdict(dk, tk)
            cells.append(f"{dk:+.4f} / {'нет данных о разбросе' if np.isnan(tk) else vk} (2SD {f4(tk)})")
        L.append(f"| {q} | {NAME[a]} → {NAME[b]} | {d:+.4f} | {f4(thr)} | {verdict(d, thr)} | " + " | ".join(cells) + " |")
    two = [v for v in st if st[v][2] > 1]
    if two:
        L += ["", "SD по сидам: " + "; ".join(f"{NAME[v]} {st[v][1]:.4f}" for v in two) +
              f". Объединённое по всем вариантам с двумя сидами: "
              f"{np.sqrt(np.mean([st[v][1] ** 2 for v in two])):.4f} (справочно, в правиле не используется)."]
    L.append("")
    # loads, specialisation, routes (seed 0)
    L += ["## Нагрузка, специализация, маршруты (сид 0, 1024 окна val)", "",
          "| вариант | блок | нагрузка экспертов, % | MaxVio | энтропия, бит | NMI(эксперт, источник) | "
          "NMI(эксперт, тип) | сосед тот же | LRU-2 попадания (перемешано) |", "|---|---|---|---|---|---|---|---|---|"]
    for v, nm in V:
        r0 = run(v)
        if not r0 or not r0["a"] or "blocks" not in r0["a"]:
            continue
        for b in r0["a"]["blocks"]:
            L.append(f"| {nm} | {b['block']} | {' '.join(f'{100 * x:.0f}' for x in b['load'])} | {b['maxvio']:.2f} | "
                     f"{b['entropy_bits']:.2f} | {b['nmi_source']:.3f} | {b['nmi_type']:.3f} | {b['neighbour_same']:.2f} | "
                     f"{b['lru2_hit']:.2f} ({b['lru2_hit_shuffled']:.2f}) |")
    L += ["", "| вариант | путей использовано (из возможных) | на 50% / 90% / 99% токенов | энтропия путей, бит | "
          "NMI(путь, источник) |", "|---|---|---|---|---|"]
    for v, nm in V:
        r0 = run(v)
        if r0 and r0["a"] and "paths" in r0["a"]:
            p = r0["a"]["paths"]
            L.append(f"| {nm} | {p['distinct']} ({p['possible']}) | {p['for_50']} / {p['for_90']} / {p['for_99']} | "
                     f"{p['entropy_bits']:.2f} | {p['nmi_source']:.3f} |")
    if amix:
        L += ["", "Коды RVQ на A-mix (до этапа 2): " + "; ".join(
            f"ступень {c['stage']}: NMI с источником {c['nmi_source']:.3f}, с типом {c['nmi_type']:.3f}"
            for c in amix["codes"])]
    L.append("")
    # source mix per expert + ablation
    L += ["## Чем заняты эксперты: доля источников и выключение эксперта (сид 0)", "",
          "Выключение: выход эксперта = 0, рост CE по источникам на 512 окнах val (маршруты прочих блоков не меняются "
          "для вариантов с кодами; у learned / per-layer могут смениться).", ""]
    for v, nm in V:
        r0 = run(v)
        if not r0 or not r0["a"] or "ablation" not in r0["a"]:
            continue
        A = r0["a"]
        L += [f"### {nm}", "", "| блок | эксперт | нагрузка, % | источники (В / К / TS), % | главный тип | "
              "ΔCE В / К / TS при выключении |", "|---|---|---|---|---|---|"]
        for b, ab in zip(A["blocks"], A["ablation"]["blocks"]):
            for e in range(len(b["load"])):
                sc = np.array(b["src_by_expert"][e], float)
                tc = np.array(b["type_by_expert"][e], float)
                srcp = " / ".join(f"{100 * x:.0f}" for x in (sc / max(1, sc.sum())))
                tt = TYPES[int(tc.argmax())] + f" {100 * tc.max() / max(1, tc.sum()):.0f}%" if tc.sum() else "—"
                L.append(f"| {b['block']} | {e} | {100 * b['load'][e]:.1f} | {srcp} | {tt} | " +
                         " / ".join(f"{x:+.3f}" for x in ab[e]["d_ce_src"]) + " |")
        L.append("")
    r0 = run("rvqsem")
    if r0 and r0["a"] and "blocks" in r0["a"]:
        L += ["## Самые редкие эксперты MoE-rvq-semantic", "", "| блок | эксперт | нагрузка, % | источники В / К / TS | "
              "типы | частые токены (число) |", "|---|---|---|---|---|---|"]
        for b in r0["a"]["blocks"]:
            rr = b["rarest"]
            sc = np.array(rr["src"], float)
            tc = np.array(rr["type"], float)
            tps = ", ".join(f"{TYPES[i]} {100 * tc[i] / max(1, tc.sum()):.0f}%" for i in np.argsort(-tc)[:3] if tc[i] > 0)
            toks = ", ".join(f"`{t!r}` {c}" for t, c in rr["top_tokens"][:10])
            L.append(f"| {b['block']} | {rr['expert']} | {100 * rr['load']:.2f} | " +
                     " / ".join(f"{100 * x / max(1, sc.sum()):.0f}%" for x in sc) + f" | {tps} | {toks} |")
        L.append("")
    # drift
    L += ["## Дрейф маршрутов на этапе 2 (64 окна val, доля токенов, у которых изменилось хоть что-то с шага 0)", "",
          "Коды RVQ — по книгам после блока N для любого варианта (у dense и learned — справочно, на маршрут они не "
          "влияют); свои эксперты — эксперт хотя бы одного MoE-блока; per-layer — свой код хотя бы одного блока.", "",
          "| вариант | коды RVQ 1k / 5k / 10k | свои эксперты 1k / 5k / 10k | коды per-layer 1k / 5k / 10k | "
          "коды RVQ по ступеням @10k |", "|---|---|---|---|---|"]
    curves = {}
    for v, nm in V:
        r0 = run(v)
        if not r0 or not r0["log"]:
            continue
        dr = {d["step"]: d for d in r0["log"]["drift"]}
        curves[v] = r0["log"]["drift"]

        def g(key, s):
            return f"{100 * dr[s][key]:.1f}" if s in dr and key in dr[s] else "—"
        steps = (1000, 5000, 10000)
        last = dr.get(10000, {})
        L.append(f"| {nm} | " + " / ".join(g("codes_any_vs0", s) for s in steps) + " | " +
                 " / ".join(g("eid_any_vs0", s) for s in steps) + " | " + " / ".join(g("plc_any_vs0", s) for s in steps) +
                 " | " + (" ".join(f"{100 * x:.1f}" for x in last["codes_per_vs0"]) if "codes_per_vs0" in last else "—") + " |")
    L.append("")
    # drift over seeds and the 2 SD rule
    L += ["### Дрейф @10k по сидам и правило 2 × SD", "",
          "| вариант | сидов | коды RVQ, % | свои эксперты, % | коды per-layer, % |", "|---|---|---|---|---|"]
    for v, nm in V:
        cells = []
        n = 0
        for key in ("codes_any_vs0", "eid_any_vs0", "plc_any_vs0"):
            z = drift_at(v, key)
            n = max(n, z[2] if z else 0)
            cells.append("—" if not z else f"{z[0]:.1f}" + (f" ± {z[1]:.1f}" if z[2] > 1 else ""))
        if n:
            L.append(f"| {nm} | {n} | " + " | ".join(cells) + " |")
    L += ["", "| вопрос | первый → второй | разность, п.п. | 2 × SD | вывод |", "|---|---|---|---|---|"]
    for a, ka, b, kb, q in DPAIRS:
        za, zb = drift_at(a, ka), drift_at(b, kb)
        if not (za and zb):
            continue
        d, thr = zb[0] - za[0], 2 * pooled_sd(za, zb)
        lab = {"codes_any_vs0": "коды RVQ", "eid_any_vs0": "эксперты", "plc_any_vs0": "коды per-layer"}
        L.append(f"| {q} | {NAME[a]} ({lab[ka]}) → {NAME[b]} ({lab[kb]}) | {d:+.1f} | {f4(thr) if np.isnan(thr) else f'{thr:.1f}'} | "
                 f"{verdict_drift(d, thr)} |")
    L.append("")
    os.makedirs("report", exist_ok=True)
    open(out, "w", encoding="utf-8").write("\n".join(L))
    plot_drift(curves)
    plot_loads()
    print("\n".join(L))


def plot_drift(curves, path="report/r21_drift.png"):
    if not curves:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    pal = ["#2a6fdb", "#e0663a", "#1f9e77", "#8a5cd1", "#c9a227", "#d6457a", "#5b8c3a", "#6b7280", "#00a3b4", "#9c6b30"]
    for i, (v, nm) in enumerate(V):
        if v not in curves:
            continue
        s = [d["step"] for d in curves[v]]
        axes[0].plot(s, [100 * d["codes_any_vs0"] for d in curves[v]], lw=2, color=pal[i], label=nm)
        key = "plc_any_vs0" if "plc_any_vs0" in curves[v][0] else "eid_any_vs0"
        if key in curves[v][0]:
            axes[1].plot(s, [100 * d[key] for d in curves[v]], lw=2, color=pal[i],
                         label=nm + (" (коды per-layer)" if key == "plc_any_vs0" else ""))
    axes[0].set_title("Коды RVQ после блока N")
    axes[1].set_title("Собственная маршрутизация варианта")
    for ax in axes:
        ax.set_xlabel("шаг этапа 2")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("% токенов, изменившихся с шага 0")
    axes[1].legend(fontsize=7, frameon=False)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_loads(path="report/r21_loads.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [(v, nm, run(v)) for v, nm in V if v not in ("dense",)]
    rows = [(v, nm, r) for v, nm, r in rows if r and r["a"] and "blocks" in r["a"]]
    if not rows:
        return
    M = len(rows[0][2]["a"]["blocks"])
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 1.25 * len(rows) + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (v, nm, r) in zip(axes, rows):
        mat = np.array([b["load"] for b in r["a"]["blocks"]]) * 100
        ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=50)
        for j in range(mat.shape[0]):
            for e in range(mat.shape[1]):
                ax.text(e, j, f"{mat[j, e]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if mat[j, e] > 30 else "#1f2937")
        ax.set_yticks(range(M))
        ax.set_yticklabels([str(b["block"]) for b in r["a"]["blocks"]], fontsize=7)
        ax.set_ylabel(nm, rotation=0, ha="right", va="center", fontsize=8)
    axes[-1].set_xticks(range(8))
    axes[-1].set_xlabel("эксперт (числа — % токенов; строки — MoE-блоки)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    report()
