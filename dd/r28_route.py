"""Round 28, stage 4 (home): routing by the output. As perlayer r21 (start from A-mix, 10k steps, 8 experts in blocks
5-8), only the route changes (dd.r28_train variants):
  out     in every MoE block the block input -> logits through the frozen A-mix output layer (final norm + tied
          embedding), the most probable token -> its token group -> expert (table group -> expert of the block)
  inout   expert by the pair (group of the current token, group of the predicted token); the pair table of the block
  mixed   blocks 5-6 by the hash of the current token (r21 hash tables), blocks 7-8 as out
  oracle  as out but by the TRUE next token (an upper bound of the effect size only, not a model)
Token groups: spherical k-means (64) of the A-mix output embeddings (= the tied embedding). Each table is made balanced
by the A-mix loads of its block on the calibration windows: items (64 groups or 64 x 64 pairs) by load, largest first,
each to the least loaded of the 8 experts (dd.r21_moe.hash_table).
Commands:
  prep     -> runs_home/r28s4/route_tables.pt, route_tables.json (k-means, loads, balance per block)
  changed  share of tokens whose expert at 2k / 5k / 10k differs from step 0 (1024 val windows), per block; load
           balance at 10k; the per-window val of r21 hash if stage 3 did not leave it
  report   -> report/r28_route.md
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .r21_moe import CTX, Data, eval_ce, hash_table, load_base, load_moe
from .r28_train import E, ROUTE4, ROUTE_TABLES, RU, R28, build

OUT = "runs_home/r28s4"
LAB = dict(base="perlayer (наш а этапа 1, маршрут по книгам)")
G = 64
INIT, BOOKS, CALIB = "runs_vast/Amix_s0/model.pt", "runs_vast/r21_books/books.pt", "data_mix/calib_train.bin"


def kmeans_sph(X, k, iters=50, seed=0, w=None):
    """spherical k-means of the rows of X (normalised), rows weighted by w (default 1): k-means++ init, cosine
    assignment"""
    g = torch.Generator(device=X.device).manual_seed(seed)
    n = X.shape[0]
    w = torch.ones(n, device=X.device, dtype=X.dtype) if w is None else w.to(X.device, X.dtype)
    C = X[torch.multinomial(w / w.sum(), 1, generator=g)]
    for _ in range(1, k):
        d = (1 - (X @ C.t()).max(1).values).clamp_min(0) * w
        C = torch.cat([C, X[torch.multinomial(d / d.sum(), 1, generator=g)]])
    for it in range(iters):
        a = (X @ C.t()).argmax(1)
        Cn = torch.zeros_like(C).index_add_(0, a, X * w[:, None])
        empty = Cn.norm(dim=1) == 0
        Cn[empty] = C[empty]
        C = F.normalize(Cn, dim=1)
    a = (X @ C.t()).argmax(1)
    return a, C, float(((X * C[a]).sum(1) * w).sum() / w.sum())


@torch.no_grad()
def prep_cmd(a):
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    books = torch.load(BOOKS, map_location="cpu", weights_only=False)
    base = load_base(INIT, dev)
    # v2: the embeddings share a large common direction (v1, plain cosine k-means: mean cosine 0.88, one group with
    # half of the load) -> centre them; weight the tokens by their train frequency (+1) so that frequent tokens get
    # groups of their own and the 64 -> 8 merge can be level
    Eb = base.emb.weight.detach().float()
    X = F.normalize(Eb - Eb.mean(0, keepdim=True), dim=1)
    freq = torch.from_numpy(np.load("data_mix/freq.npy").astype(np.float64)).float().to(dev)
    tok_group, _, cos = kmeans_sph(X, G, seed=a.seed, w=freq + 1.0)
    sizes = torch.bincount(tok_group, minlength=G)
    gl = torch.zeros(G, device=dev).index_add_(0, tok_group, freq)
    print(f"k-means (centred, frequency-weighted): {G} groups, weighted mean cosine to the centre {cos:.3f}, group "
          f"sizes {int(sizes.min())}..{int(sizes.max())} tokens, train-frequency load of the largest group "
          f"{float(gl.max() / gl.sum()):.3f} ({time.time() - t0:.0f}s)", flush=True)
    # loads of A-mix (R28 at step 0 = A-mix; its lens = the A-mix output layer)
    rt0 = dict(tok_group=tok_group.cpu(), out=torch.zeros(4, G, dtype=torch.long),
               pair=torch.zeros(4, G * G, dtype=torch.long), next=torch.zeros(4, G, dtype=torch.long))
    m = R28(base, "out", dict(books, route=rt0)).to(dev).eval()
    arr = np.fromfile(CALIB, dtype=np.uint16).astype(np.int64)
    st = np.arange(0, arr.size - CTX - 1, CTX)
    st = st[np.linspace(0, st.size - 1, min(a.calib_windows, st.size)).round().astype(int)]
    cal = torch.from_numpy(np.stack([arr[x:x + CTX + 1] for x in st])).to(dev)
    M = m.M
    c_pred = torch.zeros(M, G, dtype=torch.float64, device=dev)
    c_pair = torch.zeros(M, G * G, dtype=torch.float64, device=dev)
    c_cur = torch.zeros(G, dtype=torch.float64, device=dev)
    c_next = torch.zeros(G, dtype=torch.float64, device=dev)
    same = torch.zeros(M, dtype=torch.float64, device=dev)
    ntok = 0
    for w0 in range(0, cal.shape[0], 16):
        idx = cal[w0:w0 + 16]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, xs = m(idx, keep=True)
        ok = (idx != 0).reshape(-1)
        gc = tok_group[idx].reshape(-1)[ok]
        c_cur += torch.bincount(gc, minlength=G).double()
        nx = idx[:, 1:].reshape(-1)
        c_next += torch.bincount(tok_group[nx[nx != 0]], minlength=G).double()
        for j in range(M):
            pred = m.lens_argmax(xs[j])
            gp = tok_group[pred].reshape(-1)[ok]
            c_pred[j] += torch.bincount(gp, minlength=G).double()
            c_pair[j] += torch.bincount(gc * G + gp, minlength=G * G).double()
            same[j] += (pred == idx).reshape(-1)[ok].double().sum()
        ntok += int(ok.sum())
    tabs = dict(tok_group=tok_group.cpu(),
                out=torch.stack([torch.from_numpy(hash_table(c_pred[j].cpu().numpy(), n=E, seed=400 + j)) for j in range(M)]),
                pair=torch.stack([torch.from_numpy(hash_table(c_pair[j].cpu().numpy(), n=E, seed=500 + j)) for j in range(M)]),
                next=torch.stack([torch.from_numpy(hash_table(c_next.cpu().numpy(), n=E, seed=600 + j)) for j in range(M)]))
    tabs = {k: v.long() for k, v in tabs.items()}

    def shares(cnt, tab):
        ld = np.bincount(tab.numpy(), weights=cnt, minlength=E)
        return (ld / ld.sum()).round(4).tolist()
    info = dict(groups=G, kmeans="centred, weighted by train frequency + 1", kmeans_cos=cos, group_sizes=sizes.tolist(),
                group_load_train=(gl / gl.sum()).cpu().numpy().round(5).tolist(), calib_tokens=ntok,
                group_load_cur=(c_cur / c_cur.sum()).cpu().numpy().round(5).tolist(),
                pred_is_current_token=[float(x) / ntok for x in same.cpu()], blocks={})
    for j in range(M):
        cp, cq = c_pred[j].cpu().numpy(), c_pair[j].cpu().numpy()
        info["blocks"][str(j)] = dict(
            group_load_pred=(cp / cp.sum()).round(5).tolist(),
            top_group_share_pred=float(cp.max() / cp.sum()),
            expert_share=dict(out=shares(cp, tabs["out"][j]), inout=shares(cq, tabs["pair"][j]),
                              oracle=shares(c_next.cpu().numpy(), tabs["next"][j]),
                              hash=shares(np.bincount(cal.reshape(-1).cpu().numpy(), minlength=books["hash"].shape[1])
                                          .astype(np.float64), books["hash"][j])))
        es = info["blocks"][str(j)]["expert_share"]
        print(f"block {5 + j}: predicted = current token {info['pred_is_current_token'][j]:.3f}; largest predicted group "
              f"{info['blocks'][str(j)]['top_group_share_pred']:.3f}; expert load max: out {max(es['out']):.3f}, inout "
              f"{max(es['inout']):.3f}, oracle {max(es['oracle']):.3f}, hash {max(es['hash']):.3f}", flush=True)
    torch.save(tabs, ROUTE_TABLES if a.out == OUT else os.path.join(a.out, "route_tables.pt"))
    json.dump(info, open(os.path.join(a.out, "route_tables.json"), "w"), indent=1)
    print(f"prep done ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------------------------------------ changed experts
@torch.no_grad()
def routes(m, D, offs, bs=16):
    out = [[] for _ in range(m.M)]
    for i in range(0, offs.numel(), bs):
        idx = D.win(offs[i:i + bs])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, es, _ = m(idx, keep=True)
        for j in range(m.M):
            out[j].append(es[j][:, :-1].cpu())
    return [torch.cat(o) for o in out]


@torch.no_grad()
def changed_cmd(a):
    """share of val tokens (pad excluded) whose expert at a save differs from the expert at step 0, per block"""
    from types import SimpleNamespace
    from .r28_train import load_trained
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    D = Data("data_mix", dev)
    offs = D.starts(64 if a.smoke else 1024)
    p = os.path.join(a.out, "changed_smoke.json" if a.smoke else "changed.json")
    res = json.load(open(p)) if os.path.exists(p) else {}
    pad = (D.win(offs)[:, :-1] == 0).cpu()
    runs = [("base", "runs_home/r28s1", "r28_base")] + [(v, a.out, f"r28_{v}") for v in ROUTE4]
    if a.smoke:
        runs = [(v, "runs_home/r28t", f"r28t_{v}") for v in ROUTE4 if os.path.exists(f"runs_home/r28t/r28t_{v}")]
    for v, out, name in runs:
        if name in res or not os.path.exists(os.path.join(out, name)):
            continue
        m0, _ = build(v, INIT, BOOKS, dev)
        r0 = routes(m0.eval(), D, offs)
        del m0
        r = {}
        saves = sorted(int(f.split("_")[1]) for f in os.listdir(os.path.join(out, name)) if f.startswith("model_"))
        for st in saves:
            ns = SimpleNamespace(variant=v, init_from=INIT, books=BOOKS, seed=0, out=out, name=name)
            m = load_trained(ns, dev, D, step=st)
            rs = routes(m, D, offs)
            r[str(st)] = dict(changed=[float(((rs[j] != r0[j]) & ~pad).sum() / (~pad).sum()) for j in range(m.M)],
                              load=[np.bincount(rs[j][~pad].numpy(), minlength=E).tolist() for j in range(m.M)])
            del m
            torch.cuda.empty_cache()
        res[name] = dict(variant=v, saves=r, load0=[np.bincount(r0[j][~pad].numpy(), minlength=E).tolist()
                                                   for j in range(len(r0))])
        json.dump(res, open(p, "w"), indent=1)
        print(f"{name}: changed experts " + "; ".join(f"{st}: " + " / ".join(f"{x:.3f}" for x in r[st]["changed"])
                                                    for st in r), flush=True)
    # r21 hash per window (stage 3 leaves it in runs_home/r28s3/hash/val_full_r21.npz)
    hp = "runs_home/r28s3/hash/val_full_r21.npz"
    if not a.smoke and not os.path.exists(hp) and not os.path.exists(os.path.join(a.out, "hash_r21_val_full.npz")):
        h21 = load_moe("runs_vast/mix_hash", dev=dev)
        _, _, S, C = eval_ce(h21, D, D.starts(), per_window=True)
        np.savez(os.path.join(a.out, "hash_r21_val_full.npz"), S=S, C=C)


# ------------------------------------------------------------------------------------------------ report
def ratio_se(num, den):
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    r = num.sum() / den.sum()
    n = len(den)
    return float(r), float(math.sqrt(((num - r * den) ** 2).sum() / (n - 1) / n) / den.mean())


def npz(path):
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return z["S"].sum(1), z["C"].sum(1)


def report_cmd(a):
    from .r28_report1 import spec_se
    out = "runs_home/r28t" if a.smoke else a.out
    pre = "r28t_" if a.smoke else "r28_"
    info = json.load(open(os.path.join(a.out, "route_tables.json"))) if os.path.exists(
        os.path.join(a.out, "route_tables.json")) else None
    cf = os.path.join(a.out, "changed_smoke.json" if a.smoke else "changed.json")
    ch = json.load(open(cf)) if os.path.exists(cf) else {}
    refs = dict(perlayer=npz("runs_home/r28s1/refs/perlayer_r21_val_full_windows.npz"),
                hash=npz("runs_home/r28s3/hash/val_full_r21.npz") or npz(os.path.join(a.out, "hash_r21_val_full.npz")),
                dense=npz("runs_home/r28s1/refs/dense_val_full_windows.npz"))
    L = ["# Раунд 28, этап 4: маршрут по выходу (perlayer r21, меняется только маршрут; дома)\n"]
    L.append("Все варианты — как perlayer r21: старт с A-mix, 10k шагов, 8 экспертов в блоках 5–8. Группы токенов — "
             "сферический k-means (64) выходных эмбеддингов A-mix; таблицы «группа (пара групп) → эксперт» — свои у "
             "каждого блока, с ровной нагрузкой по A-mix на калибровке. Предсказанный токен — самый вероятный у "
             "замороженного выходного слоя A-mix (финальная норма + эмбеддинг) на входе блока. г) смотрит на настоящий "
             "следующий токен — это верхняя граница размера эффекта, а не модель.\n")
    if info:
        L.append(f"k-means: средний косинус к центру {info['kmeans_cos']:.3f}, размеры групп "
                 f"{min(info['group_sizes'])}…{max(info['group_sizes'])} токенов. Предсказанный токен совпадает с текущим "
                 "(блоки 5 / 6 / 7 / 8): " + " / ".join(f"{x:.3f}" for x in info["pred_is_current_token"]) + ".\n")
        L.append("| блок | крупнейшая группа предсказаний | макс. доля эксперта (калибровка, A-mix): по выходу / вход × "
                 "выход / подсмотренный / hash |")
        L.append("|---|---|---|")
        for j in range(4):
            b = info["blocks"][str(j)]
            es = b["expert_share"]
            L.append(f"| {5 + j} | {b['top_group_share_pred']:.3f} | {max(es['out']):.3f} / {max(es['inout']):.3f} / "
                     f"{max(es['oracle']):.3f} / {max(es['hash']):.3f} |")
    L.append("\n## Качество и различие экспертов на 10k\n")
    L.append("Качество — полный val, парно по окнам, ± 2 SE; perlayer, hash, dense-cont — модели r21. Специализация — "
             "чужой / ноль (1024 окна, все пары, ± 2 SE — дельта-метод с запасом); снято подбором — доля разницы «свой "
             "против чужого», которую снимает подбор down; сменили эксперта — доля токенов val, у которых эксперт на 10k "
             "не тот, что на шаге 0.\n")
    L.append("| вариант | шаг 0 − A-mix | val CE | Δ к perlayer | Δ к hash | Δ к dense-cont | чужой / ноль: блоки 5 / 6 / 7 / 8 "
             "(± 2 SE блока 5) | снято подбором | сменили эксперта: блоки 5 / 6 / 7 / 8 | макс. доля эксперта на 10k |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for v in ("base",) + ROUTE4:
        rdir = "runs_home/r28s1" if v == "base" else out
        name = (pre if v != "base" else "r28_") + v
        rp = os.path.join(rdir, name, "r28.json")
        if not os.path.exists(rp):
            continue
        r = json.load(open(rp))
        s2 = r.get("stage2")
        vf = (s2 or {}).get("val_full") or r.get("val_full")
        w = npz(os.path.join(rdir, name, "val_full_windows.npz"))
        cells = []
        for k in ("perlayer", "hash", "dense"):
            ref = refs[k]
            if w is None or ref is None or ref[0].shape != w[0].shape:
                cells.append("—")
            else:
                d = ratio_se(w[0] - ref[0], w[1])
                cells.append(f"{d[0]:+.4f} ± {2 * d[1]:.4f}")
        if s2:
            ms = s2["measure"]
            q = [spec_se(ms["spec"][str(j)], ms["load"][j]) for j in range(4)]
            spec = " / ".join(f"{x['spec']:.3f}" for x in q) + f" (± {2 * q[0]['spec_se']:.3f})"
            rf = " / ".join(f"{ms['summary'][str(j)]['refit_share']:.2f}" for j in range(4))
        else:
            spec = rf = "—"
        c = ch.get(name, {}).get("saves", {})
        last = c.get(str(max(map(int, c))) if c else "", {})
        chs = " / ".join(f"{x:.3f}" for x in last["changed"]) if last else "—"
        mx = (" / ".join(f"{max(ld) / max(1, sum(ld)):.2f}" for ld in last["load"])) if last else "—"
        L.append(f"| {LAB.get(v, RU[v])} | {r['ce0'] - r['ce_amix']:+.5f} | {vf['ce']:.4f} | " + " | ".join(cells) +
                 f" | {spec} | {rf} | {chs} | {mx} |")
    if ch:
        L.append("\n## Доля токенов, сменивших эксперта, по ходу обучения (2k / 5k / 10k)\n")
        L.append("| вариант | блок 5 | блок 6 | блок 7 | блок 8 |")
        L.append("|---|---|---|---|---|")
        for name, rr in ch.items():
            sv = rr["saves"]
            ks = sorted(sv, key=int)
            L.append(f"| {LAB.get(rr['variant'], RU[rr['variant']])} | " + " | ".join(" / ".join(f"{sv[k]['changed'][j]:.3f}" for k in ks)
                                                          for j in range(4)) + " |")
    txt = "\n".join(L) + "\n"
    open(os.path.join("report", "r28_route" + ("_smoke" if a.smoke else "") + ".md"), "w", encoding="utf-8").write(txt)
    print(txt)


def vast_cmd(a):
    """on a vast machine: stage-1 training (saves 2k / 5k / 10k with their measurements) and stage 2 of a.variant"""
    from . import r28_train
    r28_train.main(["train", "--variant", a.variant, "--out", OUT])
    r28_train.main(["stage2", "--variant", a.variant, "--out", OUT])


def fetchwait_cmd(a):
    """at home: wait for the fetched vast job r28s4_<variant>, copy r28.json, the per-window val and the saves to
    runs_home/r28s4/r28_<variant>; exit 1 if the job failed or after a.wait_h hours"""
    import shutil
    import sys
    job = f"r28s4_{a.variant}"
    src, dst = os.path.join("runs_vast", job), os.path.join(OUT, f"r28_{a.variant}")
    t0 = time.time()
    while not (os.path.exists(os.path.join(src, "status.json")) and os.path.exists(os.path.join(src, "vast.json"))):
        if time.time() - t0 > a.wait_h * 3600:
            sys.exit(f"{job}: no results after {a.wait_h} h")
        time.sleep(60)
    st = json.load(open(os.path.join(src, "status.json")))
    if st.get("state") != "done":
        sys.exit(f"{job}: state {st.get('state')} (stage {st.get('stage')}, rc {st.get('rc')})")
    need = ["r28.json", "val_full_windows.npz"] + [f"model_{x}_bf16.pt" for x in (2000, 5000, 10000)]
    miss = [f for f in need if not os.path.exists(os.path.join(src, f))]
    if miss:
        sys.exit(f"{job}: missing {miss}")
    os.makedirs(dst, exist_ok=True)
    for f in need:
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    assert "stage2" in json.load(open(os.path.join(dst, "r28.json"))), "stage 2 missing"
    print(f"{job}: copied {len(need)} files to {dst} (waited {(time.time() - t0) / 60:.0f} min)", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "changed", "report", "vast", "fetchwait"])
    ap.add_argument("--variant", default="out", choices=list(ROUTE4))
    ap.add_argument("--wait_h", type=float, default=8.0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--calib_windows", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    dict(prep=prep_cmd, changed=changed_cmd, report=report_cmd, vast=vast_cmd, fetchwait=fetchwait_cmd)[a.cmd](a)


if __name__ == "__main__":
    main()
