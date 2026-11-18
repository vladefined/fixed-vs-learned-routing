"""Round 28, stages 1-2 report: report/r28_stage12.md (+ appended to report/r28_tables.md by r28_report), plots
report/r28_spec.png. Variants of dd.r28_train, runs in runs_home/r28s1 (smoke: runs_home/r28t, --smoke)."""
import argparse
import json
import math
import os

import numpy as np

from .r28_train import RU, VARIANTS

BL = ("5", "6", "7", "8")
SRC = ("Википедия", "код", "сказки")
OUT = "runs_home/r28s1"


def load(out, name):
    p = os.path.join(out, name, "r28.json")
    return json.load(open(p)) if os.path.exists(p) else None


def ratio_se(num, den):
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    r = num.sum() / den.sum()
    res = num - r * den
    n = len(den)
    return float(r), float(math.sqrt((res ** 2).sum() / (n - 1) / n) / den.mean())


def wins(out, name):
    p = os.path.join(out, name, "val_full_windows.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z["S"], z["C"]


def diff(a, b):
    if a is None or b is None or a[0].shape != b[0].shape:
        return None
    return ratio_se(a[0].sum(1) - b[0].sum(1), a[1].sum(1))


def pm(d, nd=4):
    return f"{d[0]:+.{nd}f} ± {2 * d[1]:.{nd}f}" if d else "—"


def spec_se(rows, load, key="dense"):
    """weighted (token shares) foreign / zero and key / zero with delta-method SEs (pairs of a row share tokens, so the
    row SE is the mean of its pair SEs; numerator / denominator correlation ignored, as in stage 0: conservative)"""
    E = len(rows)
    w = np.asarray(load, float)
    fo = np.array([np.nanmean([rows[str(i)][str(k)][0] for k in range(E) if k != i]) for i in range(E)])
    fo_se = np.array([np.nanmean([rows[str(i)][str(k)][1] for k in range(E) if k != i]) for i in range(E)])
    ze = np.array([rows[str(i)]["zero"][0] for i in range(E)])
    ze_se = np.array([rows[str(i)]["zero"][1] for i in range(E)])
    ke = np.array([rows[str(i)][key][0] for i in range(E)])
    ke_se = np.array([rows[str(i)][key][1] for i in range(E)])
    ok = ~np.isnan(fo) & ~np.isnan(ze)
    ww = np.where(ok, w, 0.0)
    ww = ww / ww.sum()
    m = lambda v: float(np.nansum(ww * v))                                          # noqa: E731
    se = lambda v: float(np.sqrt(np.nansum((ww * v) ** 2)))                         # noqa: E731
    o, z, k_ = m(fo), m(ze), m(ke)
    so, sz, sk = se(fo_se), se(ze_se), se(ke_se)
    r_o, r_k = o / z, k_ / z
    return dict(foreign=o, zero=z, key=k_, spec=r_o, spec_se=abs(r_o) * math.sqrt((so / o) ** 2 + (sz / z) ** 2),
                kspec=r_k, kspec_se=abs(r_k) * math.sqrt((sk / k_) ** 2 + (sz / z) ** 2),
                fo_ratio_key=o / k_, fo_ratio_key_se=abs(o / k_) * math.sqrt((so / o) ** 2 + (sk / k_) ** 2))


def report1(out=OUT, smoke=False):
    names = {v: f"r28t_{v}" if smoke else f"r28_{v}" for v in VARIANTS}
    R = {v: load(out, n) for v, n in names.items()}
    R = {v: r for v, r in R.items() if r}
    seed2 = None
    for v in VARIANTS:
        r = load(out, f"r28_{v}_s1")
        if r:
            seed2 = (v, r)
    ref_dense = wins(os.path.join(out, "refs"), "") if False else None
    rd = os.path.join(out, "refs", "dense_val_full_windows.npz")
    rp = os.path.join(out, "refs", "perlayer_r21_val_full_windows.npz")
    ref_dense = (np.load(rd)["S"], np.load(rd)["C"]) if os.path.exists(rd) else None
    ref_pl = (np.load(rp)["S"], np.load(rp)["C"]) if os.path.exists(rp) else None
    W = {v: wins(out, names[v]) for v in R}
    L = ["# Раунд 28, этапы 1–2: что мешает экспертам различаться (обучение, r21 perlayer, дома)\n"]
    L.append("Все варианты — как perlayer r21: старт с A-mix, 10k шагов, те же окна и план lr, тот же маршрут по книгам "
             "слоёв. В б–д меняется только градиент, который доходит до записей экспертов в K / V внимания верхних "
             "блоков от чужих позиций (вес пары по темам — экспертам блока 5); прямой проход тот же. е — общий down на "
             "блок; ж — общий FFN A-mix из 1056 нейронов + добавки по 352 у каждого эксперта (down = 0 на старте).\n")
    # --- step 0
    L.append("## Шаг 0\n")
    L.append("| вариант | val CE на шаге 0 (256 окон) | A-mix | разница |")
    L.append("|---|---|---|---|")
    for v, r in R.items():
        L.append(f"| {RU[v]} | {r['ce0']:.5f} | {r['ce_amix']:.5f} | {r['ce0'] - r['ce_amix']:+.5f} |")
    # --- quality
    L.append("\n## Качество (полный val)\n")
    L.append("| вариант | val CE | " + " / ".join(SRC) + " | Δ к а) (наш perlayer) ± 2 SE | Δ к dense-cont ± 2 SE | "
             "Δ к perlayer r21 ± 2 SE |")
    L.append("|---|---|---|---|---|---|")
    base_w = W.get("base")
    for v, r in R.items():
        vf = (r.get("stage2") or {}).get("val_full") or r.get("val_full")
        if not vf:
            continue
        L.append(f"| {RU[v]} | {vf['ce']:.4f} | " + " / ".join(f"{x:.4f}" for x in vf["src"]) + f" | "
                 f"{pm(diff(W[v], base_w)) if v != 'base' else '—'} | {pm(diff(W[v], ref_dense))} | {pm(diff(W[v], ref_pl))} |")
    if seed2:
        v, r = seed2
        vf = (r.get("stage2") or {}).get("val_full") or r.get("val_full")
        w2 = wins(out, f"r28_{v}_s1")
        if vf:
            L.append(f"| {RU[v]}, сид 1 | {vf['ce']:.4f} | " + " / ".join(f"{x:.4f}" for x in vf["src"]) + f" | "
                     f"{pm(diff(w2, base_w))} | {pm(diff(w2, ref_dense))} | {pm(diff(w2, ref_pl))} |")
            L.append(f"\nСид 1 против сида 0 того же варианта: {pm(diff(w2, W.get(v)))}.")
    # --- specialisation over training
    L.append("\n## Специализация по ходу обучения (сохранения 2k / 5k / 10k; 256 окон val, 512 окон калибровки)\n")
    L.append("Специализация = (чужой − свой) / (ноль − свой) = чужой / ноль (свой — сама модель); dense-cont / ноль — то же "
             "с FFN dense-cont вместо чужого эксперта; «снято подбором» — доля разницы «свой против чужого», которую "
             "снимает подбор down под выход своего эксперта (2 чужих на эксперта); косинус сдвигов — между экспертами "
             "одного блока (сдвиг = веса − веса на шаге 0). ± 2 SE у «чужой / ноль» на этих 256 окнах (дельта-метод, как "
             "в этапе 0) — 0.005–0.023: в блоках 5–7 у а–д около 0.01–0.02, в блоке 8 около 0.015; у е и ж меньше.\n")
    L.append("| вариант | блок | чужой / ноль: 2k / 5k / 10k | dense-cont / ноль: 2k / 5k / 10k | снято подбором: 2k / 5k / "
             "10k | косинус сдвигов: 2k / 5k / 10k |" + (" без добавок (ж): рост CE 2k / 5k / 10k |" if "add" in R else ""))
    L.append("|---|---|---|---|---|---|" + ("---|" if "add" in R else ""))
    for v, r in R.items():
        ms = r.get("measure", {})
        steps = sorted(ms, key=int)
        for j in range(4):
            g = lambda key: " / ".join(f"{ms[s]['summary'][str(j)].get(key, float('nan')):.3f}" for s in steps)  # noqa: E731
            row = f"| {RU[v]} | {BL[j]} | {g('spec_foreign')} | {g('spec_dense')} | {g('refit_share')} | {g('cos_upd')} |"
            if "add" in R:
                row += (f" {g('addoff')} |" if v == "add" else " — |")
            L.append(row)
    # --- foreign gradient
    L.append("\n## Доля чужого градиента экспертов (начало / конец; естественная и с весами варианта)\n")
    L.append("| вариант | естественная, начало: блоки 5 / 6 / 7 | с весами варианта, начало | естественная, конец | "
             "с весами варианта, конец |")
    L.append("|---|---|---|---|---|")
    for v, r in R.items():
        f = lambda x: " / ".join(f"{y:.2f}" for y in x[:3]) if x else "—"  # noqa: E731
        gs, ge = r.get("grad_start", {}), r.get("grad_end", {})
        L.append(f"| {RU[v]} | {f(gs.get('natural'))} | {f(gs.get('trained'))} | {f(ge.get('natural'))} | {f(ge.get('trained'))} |")
    # --- stage 2 full measurement
    s2 = {v: r["stage2"]["measure"] for v, r in R.items() if r.get("stage2")}
    if seed2 and seed2[1].get("stage2"):
        s2[seed2[0] + "#s1"] = seed2[1]["stage2"]["measure"]
    lab = lambda v: RU[v.split("#")[0]] + (", сид 1" if "#s1" in v else "")             # noqa: E731
    if s2:
        L.append("\n## Этап 2: специализация на 10k с ± 2 SE (1024 окна val; калибровка 1024 окна)\n")
        L.append("Среднее по экспертам с весом доли токенов; ± 2 SE отношения — дельта-метод без учёта корреляции "
                 "числителя и знаменателя (с запасом), как в этапе 0. Разница двух вариантов значима, если она больше "
                 "√(a² + b²) из их ± 2 SE. У ж «чужой» меняет только добавку (общая часть та же), поэтому его чужой / "
                 "ноль не сравним с остальными; для ж сравнимая мера — подмена добавок / отключение добавок.\n")
        L.append("| вариант | блок | чужой эксперт | ноль | чужой / ноль ± 2 SE | dense-cont / ноль ± 2 SE | снято подбором | "
                 "косинус сдвигов | ‖сдвиг‖ / ‖A-mix‖ | ж: отключение добавок | ж: подмена добавок / отключение ± 2 SE |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for v, m in s2.items():
            for j in range(4):
                rows = m["spec"][str(j)]
                q = spec_se(rows, m["load"][j], "dense")
                sm = m["summary"][str(j)]
                addc = "— | —"
                if "addoff" in rows["0"]:
                    qa = spec_se(rows, m["load"][j], "addoff")
                    addc = f"{qa['key']:+.4f} | {qa['fo_ratio_key']:.3f} ± {2 * qa['fo_ratio_key_se']:.3f}"
                L.append(f"| {lab(v)} | {BL[j]} | {q['foreign']:+.4f} | {q['zero']:+.4f} | {q['spec']:.3f} ± "
                         f"{2 * q['spec_se']:.3f} | {q['kspec']:.3f} ± {2 * q['kspec_se']:.3f} | {sm['refit_share']:.3f} | "
                         f"{sm['cos_upd']:.3f} | {sm['upd_rel']:.3f} | {addc} |")
        L.append("\n## Этап 2: полные матрицы на 10k (1024 окна val, 1024 окна калибровки, все пары)\n")
        L.append("| вариант | чужой / ноль: блоки 5 / 6 / 7 / 8 | dense-cont / ноль | снято подбором | все блоки сразу: "
                 "чужой / ноль |")
        L.append("|---|---|---|---|---|")
        for v, m in s2.items():
            sm = m["summary"]
            g = lambda key: " / ".join(f"{sm[str(j)].get(key, float('nan')):.3f}" for j in range(4))  # noqa: E731
            sa = m.get("spec_all")
            if sa:
                fo = np.nanmean([np.nanmean([sa[str(i)][str(k)][0] for k in range(8) if k != i]) for i in range(8)])
                ze = np.nanmean([sa[str(i)]["zero"][0] for i in range(8)])
                alls = f"{fo / ze:.3f}"
            else:
                alls = "—"
            L.append(f"| {lab(v)} | {g('spec_foreign')} | {g('spec_dense')} | {g('refit_share')} | {alls} |")
    plot(R, out)
    L.append("\n![специализация по ходу обучения](r28_spec.png)")
    txt = "\n".join(L) + "\n"
    open(os.path.join("report", "r28_stage12" + ("_smoke" if smoke else "") + ".md"), "w", encoding="utf-8").write(txt)
    print(txt)
    return txt


def plot(R, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surf, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    cols = dict(base="#52514e", add="#2a78d6", hier="#eb6834", own="#1baf7a", shuf="#e87ba4", flat="#eda100", sdown="#6250d6")
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.3), dpi=110, sharey=True)
    fig.patch.set_facecolor(surf)
    panels = [(0, "spec_foreign", "блок 5: чужой / ноль"), (1, "spec_foreign", "блок 6: чужой / ноль"),
              (2, "spec_foreign", "блок 7: чужой / ноль"), (0, "spec_dense", "блок 5: dense-cont / ноль")]
    for ax, (j, key, title) in zip(axs, panels):
        ax.set_facecolor(surf)
        for v, r in R.items():
            ms = r.get("measure", {})
            steps = sorted(ms, key=int)
            if not steps:
                continue
            ax.plot([int(s) for s in steps], [ms[s]["summary"][str(j)][key] for s in steps], "-o", color=cols[v],
                    lw=2, ms=6, mec=surf, mew=1.5, label=RU[v])
        ax.set_title(title, color=ink, fontsize=11, loc="left")
        ax.set_xlabel("шаг", color=ink2)
        ax.grid(True, color=grid, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=ink2)
    axs[0].set_ylabel("специализация", color=ink2)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=ink)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(os.path.join("report", "r28_spec.png"), facecolor=surf)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    report1("runs_home/r28t" if a.smoke else OUT, a.smoke)


if __name__ == "__main__":
    main()
