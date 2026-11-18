"""Round 21 follow-up (user request): is the code drift of the RVQ variants a residual-collapse effect of the closeness
penalty? Per-stage code changes every 1k steps of stage 2, and the norm of the residual at the input of every RVQ stage
relative to the mean distance between the points of that stage's book, with the penalty (beta 1e-4) and without (0).

  python -m dd.r21_resid endpoints   # A-mix (step 0) and the final models of finished runs on the 64 drift windows
                                     #   -> runs_home/r21_resid/endpoints.json (needs model_bf16.pt of the runs)
  python -m dd.r21_resid plbooks     # the same margins / sizes for the per-layer books (blocks N+1..L) on A-mix and
                                     #   on the trained MoE-perlayer -> runs_home/r21_resid/plbooks.json
  python -m dd.r21_resid report      # + drift logs of runs_vast/mix_* (the *_rl reruns log residuals every 1k steps)
                                     #   -> report/r21_resid.md, report/r21_resid.png
Hypothesis (user): changes grow towards the late stages, and the late-stage residual shrinks much more with the penalty
than without it."""
import argparse
import json
import os

import numpy as np
import torch

from .r21_moe import Data, book_spacing, drift_rec, load_moe, rvq_encode, rvq_residuals, snapshot

OUT = os.path.join("runs_home", "r21_resid")
BOOKS = os.path.join("runs_vast", "r21_books", "books.pt")
AMIX = os.path.join("runs_vast", "Amix_s0")
END_RUNS = ["mix_rvqsem", "mix_rvqsem_s1", "mix_rvqsem_b0", "mix_rvqshared", "mix_rvqrand", "mix_frozen",
            "mix_dense", "mix_dense_s1", "mix_hash", "mix_perlayer", "mix_learned"]
TRAJ = [("mix_rvqsem", "β 1e-4, сид 0"), ("mix_rvqsem_s1", "β 1e-4, сид 1"), ("mix_rvqsem_b0", "β 0, сид 0"),
        ("mix_rvqsem_rl", "β 1e-4, сид 0 (повтор с логом остатков)"),
        ("mix_rvqsem_b0_rl", "β 0, сид 0 (повтор с логом остатков)"), ("mix_dense", "dense-cont (β 0)")]


def jl(p):
    try:
        return json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return None


@torch.no_grad()
def endpoints_cmd():
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    books = torch.load(BOOKS, map_location="cpu", weights_only=False)
    D = Data("data_mix", dev)
    offs = D.starts(64)                                       # the drift windows of the training runs
    os.makedirs(OUT, exist_ok=True)
    res = jl(os.path.join(OUT, "endpoints.json")) or {}
    m0 = load_moe(AMIX, books, dev)
    s0 = snapshot(m0, D, offs)
    del m0
    res["Amix_s0"] = drift_rec(0, s0, s0, s0)
    for n in END_RUNS:
        d = os.path.join("runs_vast", n)
        if not os.path.exists(os.path.join(d, "model_bf16.pt")):
            print(f"{n}: no model_bf16.pt, skipped", flush=True)
            continue
        m = load_moe(d, None, dev, fname="model_bf16.pt")
        s = snapshot(m, D, offs)
        del m
        s = {k: v for k, v in s.items() if k in ("codes", "rn", "mg")}          # A-mix has no experts to compare
        r = drift_rec(10000, s, s0, s0)
        lg = jl(os.path.join(d, "log.json"))
        if lg:                                                 # cross-check against the code changes logged in training
            last = [x for x in lg["drift"] if x["step"] == 10000]
            r["logged_codes_per_vs0"] = last[0]["codes_per_vs0"] if last else None
        res[n] = r
        print(f"{n}: resid/|.| " + " ".join(f"{x:.2f}" for x in r["resid_norm"]) + " | changes " +
              " ".join(f"{100 * x:.1f}" for x in r["codes_per_vs0"]) + " (logged " +
              (" ".join(f"{100 * x:.1f}" for x in r["logged_codes_per_vs0"]) if r.get("logged_codes_per_vs0") else "—")
              + ")", flush=True)
        json.dump(res, open(os.path.join(OUT, "endpoints.json"), "w"), indent=1)
    json.dump(res, open(os.path.join(OUT, "endpoints.json"), "w"), indent=1)


@torch.no_grad()
def plbooks_cmd():
    """per-layer books: the code of block l is chosen by the nearest point to the RMS-normalised input of block l;
    margin = distance to the 2nd nearest point - distance to the chosen one, in units of the mean distance between the
    book's points; size = |state| / mean distance; quantisation error = distance to the chosen point / mean distance"""
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    books = torch.load(BOOKS, map_location="cpu", weights_only=False)
    pl = books["pl"].float().to(dev)
    rvq = books["rvq"].float().to(dev)
    sp, sp_rvq = book_spacing(books["pl"]), book_spacing(books["rvq"])
    D = Data("data_mix", dev)
    offs = D.starts(64)
    res = dict(spacing=sp, models={})
    for name, m in (("Amix_s0", lambda: load_moe(AMIX, books, dev)),
                    ("mix_perlayer", lambda: load_moe(os.path.join("runs_vast", "mix_perlayer"), None, dev,
                                                      fname="model_bf16.pt"))):
        model = m()
        acc = [dict(mg=[], q=[], nrm=[], code=[]) for _ in range(pl.shape[0])]
        racc = dict(rn=[], mg=[])
        for i in range(0, offs.numel(), 32):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, aux = model(D.win(offs[i:i + 32]), {"states": True})
            u = aux["s_in"][0][:, :-1].reshape(-1, aux["s_in"][0].shape[-1]).float()
            rc, _ = rvq_encode(u, rvq)
            rn, rmg = rvq_residuals(u, rc, rvq)
            racc["rn"].append(rn)
            racc["mg"].append(rmg)
            for j, x in enumerate(aux["s_in"]):
                x = x[:, :-1].reshape(-1, x.shape[-1]).float()
                with torch.autocast("cuda", enabled=False):
                    d2 = (x.pow(2).sum(1, keepdim=True) - 2 * x @ pl[j].t() + pl[j].pow(2).sum(1)[None]).clamp_min(0)
                two = d2.sqrt().topk(2, dim=1, largest=False)
                acc[j]["mg"].append(two.values[:, 1] - two.values[:, 0])
                acc[j]["q"].append(two.values[:, 0])
                acc[j]["nrm"].append(x.norm(dim=1))
                acc[j]["code"].append(two.indices[:, 0])
        del model
        out = []
        for j, a in enumerate(acc):
            mg, q, nrm, code = (torch.cat(a[k]) for k in ("mg", "q", "nrm", "code"))
            f = torch.bincount(code, minlength=pl.shape[1]).double()
            f = f[f > 0] / f.sum()
            out.append(dict(block=books["N"] + 1 + j, margin_mean=float(mg.mean()) / sp[j],
                            margin_median=float(mg.median()) / sp[j], size=float(nrm.mean()) / sp[j],
                            qerr=float(q.mean()) / sp[j], code_entropy_bits=float(-(f * f.log2()).sum()),
                            near_boundary=float((mg < 0.02 * sp[j]).float().mean())))
            print(f"{name} block {out[-1]['block']}: margin mean {out[-1]['margin_mean']:.3f} median "
                  f"{out[-1]['margin_median']:.3f}, size {out[-1]['size']:.3f}, dist to chosen {out[-1]['qerr']:.3f}, "
                  f"margin < 0.02: {100 * out[-1]['near_boundary']:.1f}%", flush=True)
        res["models"][name] = out
        rn, rmg = torch.cat(racc["rn"]), torch.cat(racc["mg"])
        res.setdefault("rvq", {})[name] = [dict(stage=j + 1, margin_mean=float(rmg[:, j].mean()) / sp_rvq[j],
                                                margin_median=float(rmg[:, j].median()) / sp_rvq[j],
                                                size=float(rn[:, j].mean()) / sp_rvq[j],
                                                qerr=float(rn[:, j + 1].mean()) / sp_rvq[j],
                                                near_boundary=float((rmg[:, j] < 0.02 * sp_rvq[j]).float().mean()))
                                           for j in range(rvq.shape[0])]
        res["spacing_rvq"] = sp_rvq
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "plbooks.json"), "w"), indent=1)


def pen_by_k(lg, every=1000):
    """mean closeness penalty |u - sum book|^2 on the training batches (logged every 100 steps) per 1k-step block"""
    h = lg.get("train_log") or []
    out = {}
    for r in h:
        if "pen" in r:
            out.setdefault(min(r["step"] // every + 1, 10) * every, []).append(r["pen"])
    return {k: float(np.mean(v)) for k, v in sorted(out.items())}


def report_cmd(out_md="report/r21_resid.md", out_png="report/r21_resid.png"):
    books = torch.load(BOOKS, map_location="cpu", weights_only=False)
    sp = book_spacing(books["rvq"])
    M = len(sp)
    ep = jl(os.path.join(OUT, "endpoints.json")) or {}
    L = ["# r21: дрейф кодов RVQ по ступеням и остаток на входе ступени", "",
         "Окна — те же 64 окна val, что у дрейфа в обучении (32 704 токена). Остаток на входе ступени j: r₁ = u "
         "(RMS-нормированное состояние после блока N, |u| = √512 = 22.6), r_{j+1} = r_j − книга_j[код_j]. "
         "Относительный остаток = средняя |r_j| / среднее расстояние между точками книги j. Запас выбора = расстояние "
         "до второй ближайшей точки книги минус расстояние до выбранной.", "",
         "Среднее расстояние между точками книги по ступеням: " + " / ".join(f"{x:.2f}" for x in sp) + ".", ""]
    logs = {n: jl(os.path.join("runs_vast", n, "log.json")) for n, _ in TRAJ}
    lab = dict(TRAJ)
    # 1) per-stage changes every 1k
    L += ["## Доля токенов, у которых код ступени изменился с шага 0, % (каждые 1k шагов)", ""]
    for n, nm in TRAJ:
        lg = logs[n]
        if not lg:
            continue
        L += [f"### {nm} (`{n}`)", "", "| шаг | " + " | ".join(f"ступень {j + 1}" for j in range(M)) +
              " | хоть одна | от предыдущего снимка по ступеням |", "|---" * (M + 3) + "|"]
        for d in lg["drift"][1:]:
            prev = d.get("codes_per_vsprev")
            L.append(f"| {d['step']} | " + " | ".join(f"{100 * x:.1f}" for x in d["codes_per_vs0"]) +
                     f" | {100 * d['codes_any_vs0']:.1f} | " +
                     (" / ".join(f"{100 * x:.1f}" for x in prev) if prev else "—") + " |")
        L.append("")
    # 2) residual trajectories (reruns with the residual log)
    have = [(n, nm) for n, nm in TRAJ if logs[n] and "resid_norm" in logs[n]["drift"][0]]
    if have:
        L += ["## Относительный остаток на входе ступеней по ходу обучения (повторы с логом)", "",
              "| прогон | шаг | " + " | ".join(f"вход ступени {j + 1}" for j in range(M)) + " | после ступени "
              f"{M} (абс.) | запас выбора / расстояние книги, ступени 1..{M} | энтропия кодов, бит |",
              "|---" * (M + 5) + "|"]
        for n, nm in have:
            for d in logs[n]["drift"]:
                rn = d["resid_norm"]
                L.append(f"| {nm} | {d['step']} | " + " | ".join(f"{rn[j] / sp[j]:.3f}" for j in range(M)) +
                         f" | {rn[M]:.2f} | " + " / ".join(f"{d['margin'][j] / sp[j]:.3f}" for j in range(M)) + " | " +
                         " / ".join(f"{x:.2f}" for x in d["code_entropy_bits"]) + " |")
        L.append("")
        L += ["Сжатие остатка к шагу 10k (|r_j|@10k / |r_j|@0):", "", "| прогон | " +
              " | ".join(f"ступень {j + 1}" for j in range(M)) + f" | после {M} |", "|---" * (M + 2) + "|"]
        for n, nm in have:
            a, b = logs[n]["drift"][0]["resid_norm"], logs[n]["drift"][-1]["resid_norm"]
            L.append(f"| {nm} | " + " | ".join(f"{b[j] / a[j]:.3f}" for j in range(M + 1)) + " |")
        L.append("")
    # 3) endpoints of the finished runs
    if ep:
        L += ["## Шаг 0 (A-mix) и шаг 10k готовых прогонов (веса bf16)", "",
              "| модель | " + " | ".join(f"отн. остаток, вход {j + 1}" for j in range(M)) + f" | после {M} (абс.) | "
              "смены кода по ступеням, % (в логе обучения) | запас / расстояние книги |", "|---" * (M + 4) + "|"]
        for n in ["Amix_s0"] + END_RUNS:
            if n not in ep:
                continue
            r = ep[n]
            rn = r["resid_norm"]
            chg = " / ".join(f"{100 * x:.1f}" for x in r["codes_per_vs0"])
            lgd = r.get("logged_codes_per_vs0")
            if lgd:
                chg += " (" + " / ".join(f"{100 * x:.1f}" for x in lgd) + ")"
            L.append(f"| {n} | " + " | ".join(f"{rn[j] / sp[j]:.3f}" for j in range(M)) + f" | {rn[M]:.2f} | {chg} | " +
                     " / ".join(f"{r['margin'][j] / sp[j]:.3f}" for j in range(M)) + " |")
        L += ["", "Смены кода здесь посчитаны по весам bf16, поэтому могут немного отличаться от значения в логе "
              "(веса fp32 в обучении).", ""]
    # 3b) per-layer books against the RVQ stages
    plb = jl(os.path.join(OUT, "plbooks.json"))
    if plb:
        L += ["## Книги per-layer против ступеней RVQ: запас выбора и размер состояния", "",
              "Всё — в долях среднего расстояния между точками своей книги. Размер = средняя |вход| (у per-layer вход — "
              "всё нормированное состояние, |s| = 22.6; у RVQ — остаток на входе ступени). До выбранной = среднее "
              "расстояние до выбранной точки (ошибка квантования). У RVQ «до выбранной» = |остаток после ступени|.", "",
              "| книга | ср. расстояние в книге | модель | запас: среднее / медиана | доля с запасом < 0.02 | размер | "
              "до выбранной |", "|---|---|---|---|---|---|---|"]
        for j in range(M):
            for n, nm in (("Amix_s0", "A-mix"), ("mix_perlayer", "обученный perlayer")):
                b = plb["rvq"][n][j]
                L.append(f"| RVQ ступень {j + 1} | {sp[j]:.2f} | {nm} | {b['margin_mean']:.3f} / "
                         f"{b['margin_median']:.3f} | {100 * b['near_boundary']:.1f}% | {b['size']:.3f} | "
                         f"{b['qerr']:.3f} |")
        for j in range(len(plb["spacing"])):
            for n, nm in (("Amix_s0", "A-mix"), ("mix_perlayer", "обученный perlayer")):
                b = plb["models"][n][j]
                L.append(f"| per-layer, блок {b['block']} | {plb['spacing'][j]:.2f} | {nm} | {b['margin_mean']:.3f} / "
                         f"{b['margin_median']:.3f} | {100 * b['near_boundary']:.1f}% | {b['size']:.3f} | "
                         f"{b['qerr']:.3f} |")
        L.append("")
    # 4) the penalty on the training batches
    L += ["## Штраф |u − Σ книга|² на батчах обучения (среднее за 1k шагов; на старте ≈ 227.5)", "",
          "| прогон | " + " | ".join(f"{k // 1000}k" for k in range(1000, 10001, 1000)) + " |", "|---" * 11 + "|"]
    for n, nm in TRAJ:
        lg = logs[n]
        if lg and n != "mix_dense":
            pk = pen_by_k(lg)
            L.append(f"| {nm} | " + " | ".join(f"{pk.get(k, float('nan')):.1f}" for k in range(1000, 10001, 1000)) + " |")
    L.append("")
    os.makedirs("report", exist_ok=True)
    open(out_md, "w", encoding="utf-8").write("\n".join(L))
    plot(logs, have, sp, M, out_png)
    print("\n".join(L))


def plot(logs, have, sp, M, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pal = ["#2a6fdb", "#1f9e77", "#c9a227", "#d6457a"]
    ncol = 3 if have else 1
    fig, axes = plt.subplots(1, ncol, figsize=(4.6 * ncol, 3.9))
    axes = np.atleast_1d(axes)
    styles = {"mix_rvqsem": ("-", "β 1e-4"), "mix_rvqsem_b0": ("--", "β 0"), "mix_rvqsem_rl": ("-", None),
              "mix_rvqsem_b0_rl": ("--", None)}
    ax = axes[0]
    for n, (ls, nm) in styles.items():
        lg = logs.get(n)
        if not lg or nm is None:
            continue
        st = [d["step"] for d in lg["drift"]]
        for j in range(M):
            ax.plot(st, [100 * d["codes_per_vs0"][j] for d in lg["drift"]], ls, color=pal[j % 4], lw=2,
                    label=f"ступень {j + 1}" if ls == "-" else None)
    ax.set_title("Код ступени изменился с шага 0\n(сплошные — β 1e-4, штрих — β 0)", fontsize=9)
    ax.set_ylabel("% токенов")
    if have:
        for n, (ls, _) in styles.items():
            lg = logs.get(n)
            if not lg or "resid_norm" not in lg["drift"][0]:
                continue
            st = [d["step"] for d in lg["drift"]]
            for j in range(1, M):
                axes[1].plot(st, [d["resid_norm"][j] / sp[j] for d in lg["drift"]], ls, color=pal[j % 4], lw=2,
                             label=f"ступень {j + 1}" if ls == "-" else None)
                axes[2].plot(st, [d["margin"][j] / sp[j] for d in lg["drift"]], ls, color=pal[j % 4], lw=2,
                             label=f"ступень {j + 1}" if ls == "-" else None)
        axes[1].set_title("|остаток на входе| / расстояние в книге\n(сплошные — β 1e-4, штрих — β 0)", fontsize=9)
        axes[2].set_title("Запас выбора кода / расстояние в книге\n(сплошные — β 1e-4, штрих — β 0)", fontsize=9)
    for a in axes:
        a.set_xlabel("шаг этапа 2")
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
        a.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["endpoints", "plbooks", "report"])
    a = ap.parse_args()
    dict(endpoints=endpoints_cmd, plbooks=plbooks_cmd, report=report_cmd)[a.cmd]()
