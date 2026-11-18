"""Round 6: adaptivity of the QG gate on TinyStories, on finished checkpoints (no training).
  python -m dd.ts_adapt run runs_ts/QG25_s0 [...]     -> <run>/adapt.json
  python -m dd.ts_adapt report runs_ts report          -> report/r6_adapt_tables.md, report/r6_adapt.png
Tokens: the same 608 val windows as dd.ts_eval (tok.npz).  Position t: the gate decides on the state of the
current token cur[t]; the "difficulty" of t is the loss of predicting nxt[t] (own, or A_s0's on the same token).
Token table (as in rounds 4-5): "current token -> group open" per (block, group), majority over N_TR random train
windows, tested on the val windows; substitution = every gate replaced by the table lookup."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import binom, spearmanr

from .ts_eval import BS, CTX, auroc, batches, load_run, val_starts, vocab_masks
from .ts_train import load_stream

N_VAL_WIN = 608          # = what dd.ts_eval stored in tok.npz (38 batches of 16)
N_TR, MIN_SEEN = 2000, 5
RUNS = ["QG25_s0", "QG25_s1", "QG35_s0", "QG50_s0", "QGsplit25_s0", "QGsplit35_s0", "QGsplit50_s0"]
LABEL = {"QG25_s0": "QG25 s0", "QG25_s1": "QG25 s1", "QG35_s0": "QG35", "QG50_s0": "QG50",
         "QGsplit25_s0": "QG-split25", "QGsplit35_s0": "QG-split35", "QGsplit50_s0": "QG-split50"}
CATS = ["word_start", "word_cont", "punct", "newline", "name", "function", "eot", "story_first5"]
CAT_RU = {"word_start": "начало слова", "word_cont": "продолжение слова", "punct": "пунктуация",
          "newline": "перевод строки", "name": "имя", "function": "частое служебное слово",
          "eot": "конец истории (EOT)", "story_first5": "первые 5 токенов истории"}
POS_BINS = [(1, 5), (6, 20), (21, 100), (101, 300), (301, 10 ** 6)]


def train_windows(n=N_TR, seed=0):
    mm = np.memmap(os.path.join("data_ts", "train.bin"), dtype=np.uint16, mode="r")
    offs = np.sort(np.random.default_rng(seed).integers(0, mm.size - CTX - 1, n))
    return torch.from_numpy(np.stack([np.asarray(mm[o:o + CTX + 1]) for o in offs]).astype(np.int64))


def story_pos(W):
    """position of every token since the last EOT in its window (EOT itself = 0); -1 if no EOT seen yet"""
    pos = np.full(W.shape, -1, np.int64)
    for i, w in enumerate(W):
        last = -1
        for t, x in enumerate(w):
            if x == 1:
                last = t
            if last >= 0:
                pos[i, t] = t - last
    return pos


def categories(v, prev, ids, pos):
    alnum_end = np.array([bool(x) and x[-1].isalnum() for x in v["s"]])
    nl = np.array([x == "" for x in v["st"]])
    nl[:2] = False                                                    # pad, EOT
    alpha = v["alpha"][ids]
    cont = alpha & ~v["word_start"][ids] & alnum_end[prev] & (prev != 1)
    return dict(word_start=alpha & ~cont, word_cont=cont, punct=v["punct"][ids], newline=nl[ids],
                name=v["name"][ids], function=v["func"][ids] & alpha, eot=ids == 1,
                story_first5=(pos >= 1) & (pos <= 5))


def pb_pmf(p):
    """Poisson-binomial pmf for independent Bernoulli(p_i)"""
    f = np.array([1.0])
    for q in p:
        f = np.convolve(f, [1 - q, q])
    return f


def spread_stats(dec, cur, V, min_tok=20):
    """dec: N, L, G (0/1) decisions; cur: N current-token ids.  Per block: observed distribution of the number of
    open groups vs 8 independent gates (binomial with the block share; Poisson-binomial with per-group shares);
    within-token variance (tokens seen >= min_tok times) vs the binomial with the token's own share."""
    N, L, G = dec.shape
    cn = np.bincount(cur, minlength=V)
    keep = cn[cur] >= min_tok
    k_tok = int((cn >= min_tok).sum())
    c = dec.sum(2)
    sp = []
    for l in range(L):
        x = c[:, l]
        p = float(x.mean() / G)
        pg = dec[:, l].mean(0)
        b = binom(G, p)
        pb = pb_pmf(pg)
        mu_t = (np.bincount(cur, x, V) / np.maximum(cn, 1))[cur]
        var_within = float(((x - mu_t)[keep] ** 2).sum() / max(keep.sum() - k_tok, 1))
        pt = mu_t[keep] / G
        sp.append(dict(p=p, obs=dict(p0=float((x == 0).mean()), le2=float((x <= 2).mean()), ge6=float((x >= 6).mean()),
                                     var=float(x.var())),
                       binom=dict(p0=float(b.pmf(0)), le2=float(b.cdf(2)), ge6=float(b.sf(5)), var=G * p * (1 - p)),
                       poisbin=dict(p0=float(pb[0]), le2=float(pb[:3].sum()), ge6=float(pb[6:].sum()),
                                    var=float((pg * (1 - pg)).sum())),
                       var_within_token=var_within, var_within_binom_own=float((G * pt * (1 - pt)).mean()),
                       hist=np.bincount(x.astype(np.int64), minlength=G + 1).tolist()))
    tot = c.sum(1)
    p_all = tot.mean() / (L * G)
    pg_all = dec.mean(0).ravel()
    cc = np.corrcoef(c.T)
    resid = c - np.stack([(np.bincount(cur, c[:, l], V) / np.maximum(cn, 1))[cur] for l in range(L)], 1)
    cr = np.corrcoef(resid[keep].T)
    rt = resid[keep].sum(1)
    return dict(layer=sp, total_var=float(tot.var()), total_var_binom=float(L * G * p_all * (1 - p_all)),
                block_corr_within_token=float(cr[~np.eye(L, dtype=bool)].mean()),
                total_var_within_token=float(rt.var()), total_var_within_token_indep=float(resid[keep].var(0).sum()),
                total_var_poisbin=float((pg_all * (1 - pg_all)).sum()),
                total_hist=np.bincount(tot.astype(np.int64), minlength=L * G + 1).tolist(),
                block_corr_mean=float(cc[~np.eye(L, dtype=bool)].mean()))


def respread(run, V=8192):
    """recompute the spread block of adapt.json on the CPU from tok.npz (decisions are identical, see check)"""
    z = np.load(os.path.join(run, "tok.npz"))
    dec = np.unpackbits(z["open_bits"][..., None], axis=-1, bitorder="little")[..., :8].astype(np.float32)
    p = os.path.join(run, "adapt.json")
    res = json.load(open(p))
    res["spread"] = spread_stats(dec, z["cur"].astype(np.int64), V)
    json.dump(res, open(p, "w"), indent=1)


@torch.no_grad()
def analyse(run, ref=os.path.join("runs_ts", "A_s0")):
    from tokenizers import Tokenizer
    dev = "cuda"
    model, cfg = load_run(run)
    L, G, V = cfg.n_layers, cfg.depth_groups, cfg.vocab
    # ---------------- token table on random train windows
    cnt = torch.zeros(V, device=dev)
    op = torch.zeros(L, V, G, device=dev)
    tw = train_windows()
    for i in range(0, tw.shape[0], BS):
        idx = tw[i:i + BS].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, aux = model(idx)
        t = idx.flatten()
        cnt.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
        for l in range(L):
            op[l].index_add_(0, t, aux["dd"][l].reshape(-1, G).float())
    glob = op.sum(1) / cnt.sum()                                         # L, G
    p_tab = torch.where((cnt >= MIN_SEEN)[None, :, None], op / cnt.clamp_min(1)[None, :, None], glob[:, None, :])
    tab = p_tab > 0.5
    # ---------------- val windows: own loss / top-1, decisions, substitution
    val = load_stream(os.path.join("data_ts", "val.bin"), dev)
    starts = val_starts(val)[:N_VAL_WIN]
    cur_idx = {}

    def gate_fn(blk, m, zg):
        return tab[blk.l][cur_idx["idx"]].float()

    LOSS, OK, DEC, base_sum, sub_sum, sub_open, n = [], [], [], 0.0, 0.0, 0.0, 0
    for idx in batches(val, starts):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, aux = model(idx)
        tgt = idx[:, 1:]
        tl = F.cross_entropy(lg[:, :-1].float().flatten(0, 1), tgt.flatten(), reduction="none").view(tgt.shape)
        LOSS.append(tl.cpu().numpy())
        OK.append((lg[:, :-1].argmax(-1) == tgt).cpu().numpy())
        DEC.append(torch.stack(aux["dd"], 2)[:, :-1].round().to(torch.uint8).cpu().numpy())   # B, T, L, G
        cur_idx["idx"] = idx
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg2, aux2 = model(idx, ctl={"hc_gate_fn": gate_fn})
        sub_sum += float(F.cross_entropy(lg2[:, :-1].float().flatten(0, 1), tgt.flatten(), reduction="sum"))
        sub_open += float(torch.stack(aux2["dd"], 2)[:, :-1].float().mean()) * tgt.numel()
        base_sum += float(tl.sum())
        n += tgt.numel()
    loss, ok = np.concatenate(LOSS).ravel(), np.concatenate(OK).ravel()
    dec = np.concatenate(DEC).reshape(-1, L, G).astype(np.float32)
    rz, oz = np.load(os.path.join(ref, "tok.npz")), np.load(os.path.join(run, "tok.npz"))
    a_loss = rz["loss"].astype(np.float32)
    stored = np.unpackbits(oz["open_bits"][..., None], axis=-1, bitorder="little")[..., :G]
    res = dict(run=run, n_tok=int(loss.size),
               check=dict(decisions_equal_tok_npz=float((stored == dec).mean()),
                          loss_max_abs_diff_tok_npz=float(np.abs(oz["loss"].astype(np.float32) - loss).max())))
    cur = oz["cur"].astype(np.int64)
    nxt = oz["ids"].astype(np.int64)
    W = np.concatenate([cur.reshape(-1, CTX), nxt.reshape(-1, CTX)[:, -1:]], 1)        # windows, CTX+1
    c = dec.sum(2)                                                                     # N, L  (0..8)
    tot = c.sum(1)                                                                     # N     (0..64)
    wrong = ~ok
    # ---------------- (a) difficulty correlation
    rho_l = [float(spearmanr(c[:, l], a_loss).statistic) for l in range(L)]

    def by_tok_mean(x):
        return (np.bincount(cur, x, V) / np.maximum(np.bincount(cur, minlength=V), 1))[cur]

    r_open, r_loss = tot - by_tok_mean(tot), a_loss - by_tok_mean(a_loss)
    tab_tot = p_tab.sum(2).sum(0)[torch.from_numpy(cur).to(dev)].cpu().numpy()       # table-predicted open count
    res["difficulty"] = dict(
        rho_total=float(spearmanr(tot, a_loss).statistic), rho_layer=rho_l, rho_layer_mean=float(np.mean(rho_l)),
        rho_own_total=float(spearmanr(tot, loss).statistic),
        rho_within_token=float(spearmanr(r_open, r_loss).statistic),
        rho_table_pred=float(spearmanr(tab_tot, a_loss).statistic))
    # ---------------- (b) own top-1 error
    res["own_error"] = dict(
        error_rate=float(wrong.mean()), auroc_total=auroc(tot, wrong),
        auroc_layer=[auroc(c[:, l], wrong) for l in range(L)],
        auroc_within_token=auroc(r_open, wrong), auroc_table_pred=auroc(tab_tot, wrong),
        auroc_A_loss=auroc(a_loss, wrong))
    # ---------------- (c) categories (current token = what the gate sees; next token = what is predicted)
    tok = Tokenizer.from_file(os.path.join("data_ts", "tokenizer.json"))
    v = vocab_masks(tok, np.load(os.path.join("data_ts", "freq.npy")))
    pos_w = story_pos(W)
    prv = np.concatenate([np.full((W.shape[0], 1), -1), W[:, :-2]], 1).ravel()
    prv_safe = np.where(prv < 0, 1, prv)
    pos_cur, pos_nxt = pos_w[:, :-1].ravel(), pos_w[:, 1:].ravel()
    cats = dict(current=categories(v, prv_safe, cur, pos_cur), next=categories(v, cur, nxt, pos_nxt))
    mean_share = float(tot.mean() / (L * G))
    res["mean_open_share"] = mean_share
    res["by_category"] = {view: {k: dict(token_share=float(m.mean()), open_share=float(tot[m].mean() / (L * G)),
                                         A_loss=float(a_loss[m].mean()), own_error=float(wrong[m].mean()))
                                 for k, m in cm.items()} for view, cm in cats.items()}
    res["by_story_pos"] = [dict(lo=lo, hi=hi, token_share=float(((pos_cur >= lo) & (pos_cur <= hi)).mean()),
                                open_share=float(tot[(pos_cur >= lo) & (pos_cur <= hi)].mean() / (L * G)),
                                A_loss=float(a_loss[(pos_cur >= lo) & (pos_cur <= hi)].mean()))
                           for lo, hi in POS_BINS]
    # ---------------- (d) spread vs independent random gates
    res["spread"] = spread_stats(dec, cur, V)
    # ---------------- (e) predictability from the current token alone
    t_cur = torch.from_numpy(cur).to(dev)
    acc, base, r2 = [], [], []
    for l in range(L):
        pred = tab[l][t_cur].cpu().numpy()
        acc.append(float((pred == (dec[:, l] > 0.5)).mean()))
        gb = (glob[l] > 0.5).cpu().numpy()
        base.append(float((gb[None, :] == (dec[:, l] > 0.5)).mean()))
        pc = p_tab[l][t_cur].sum(-1).cpu().numpy()
        r2.append(float(1 - ((c[:, l] - pc) ** 2).sum() / ((c[:, l] - c[:, l].mean()) ** 2).sum()))
    r2_tot = float(1 - ((tot - tab_tot) ** 2).sum() / ((tot - tot.mean()) ** 2).sum())
    seen_share = float((cnt[t_cur] >= MIN_SEEN).float().mean())
    res["token_table"] = dict(acc_layer=acc, base_layer=base, acc=float(np.mean(acc)), base=float(np.mean(base)),
                              kappa=float((np.mean(acc) - np.mean(base)) / (1 - np.mean(base))),
                              r2_layer=r2, r2_total=r2_tot, table_open_share=float(tab[:, t_cur].float().mean()),
                              seen_share=seen_share, n_train_tokens=int(cnt.sum()),
                              val_ce_base=base_sum / n, val_ce_sub=sub_sum / n, sub_open_share=sub_open / n)
    json.dump(res, open(os.path.join(run, "adapt.json"), "w"), indent=1)
    d, e, tt = res["difficulty"], res["own_error"], res["token_table"]
    print(f"{run}: rho {d['rho_total']:.3f} (blocks mean {d['rho_layer_mean']:.3f}, within-token {d['rho_within_token']:.3f}, "
          f"table {d['rho_table_pred']:.3f}); AUROC err {e['auroc_total']:.3f}; table acc {tt['acc']:.3f} base {tt['base']:.3f} "
          f"R2 {tt['r2_total']:.3f}; subst dCE {tt['val_ce_sub'] - tt['val_ce_base']:+.3f}; check {res['check']}", flush=True)


# ====================================================================================== report
def report(base="runs_ts", out="report"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .report import INK2  # noqa: F401  (shared rcParams)
    R = [(n, json.load(open(os.path.join(base, n, "adapt.json")))) for n in RUNS
         if os.path.exists(os.path.join(base, n, "adapt.json"))]
    ev = {n: json.load(open(os.path.join(base, n, "ts_eval.json"), encoding="utf-8")) for n, _ in R}
    hdr = "| | " + " | ".join(LABEL[n] for n, _ in R) + " |"
    sep = "|---" * (len(R) + 1) + "|"
    L_ = ["# TinyStories: адаптивность гейта QG (готовые чекпойнты, 608 окон val = 311k токенов)\n"]

    def row(name, f, fmt="{:.3f}"):
        L_.append(f"| {name} | " + " | ".join(fmt.format(f(r)) for _, r in R) + " |")

    L_ += ["## Связь с трудностью (Спирмен с потерей A на том же токене) и с ошибкой самой модели\n", hdr, sep]
    row("открыто в среднем, доля", lambda r: r["mean_open_share"])
    row("ρ: сумма по блокам", lambda r: r["difficulty"]["rho_total"])
    row("ρ: среднее по блокам", lambda r: r["difficulty"]["rho_layer_mean"])
    for l in range(8):
        row(f"ρ: блок {l}", lambda r, l=l: r["difficulty"]["rho_layer"][l])
    row("ρ со своей потерей", lambda r: r["difficulty"]["rho_own_total"])
    row("ρ внутри токена (отклонения от среднего по токену)", lambda r: r["difficulty"]["rho_within_token"])
    row("ρ прогноза таблицы токенов", lambda r: r["difficulty"]["rho_table_pred"])
    row("доля ошибок топ-1", lambda r: r["own_error"]["error_rate"])
    row("AUROC открыто → ошибка топ-1", lambda r: r["own_error"]["auroc_total"])
    for l in range(8):
        row(f"AUROC: блок {l}", lambda r, l=l: r["own_error"]["auroc_layer"][l])
    row("AUROC внутри токена", lambda r: r["own_error"]["auroc_within_token"])
    row("AUROC прогноза таблицы токенов", lambda r: r["own_error"]["auroc_table_pred"])
    row("для сравнения: AUROC потери A → ошибка", lambda r: r["own_error"]["auroc_A_loss"])
    L_.append("")
    for view, title in (("current", "текущий токен (то, на чём решает гейт)"), ("next", "предсказываемый токен")):
        L_ += [f"## Доля открытых групп по типу: {title}\n",
               "Категории пересекаются (имя и служебное слово — тоже начало слова). В скобках — доля токенов.\n",
               "| тип | " + " | ".join(LABEL[n] for n, _ in R) + " | потеря A |", "|---" * (len(R) + 2) + "|"]
        for k in CATS:
            r0 = R[0][1]["by_category"][view][k]
            L_.append(f"| {CAT_RU[k]} ({100 * r0['token_share']:.1f}%) | " +
                      " | ".join(f"{r['by_category'][view][k]['open_share']:.3f}" for _, r in R) + f" | {r0['A_loss']:.2f} |")
        row("все токены", lambda r: r["mean_open_share"])
        L_.append("")
    L_ += ["## Доля открытых групп по позиции текущего токена в истории (только окна, где видно начало истории)\n",
           "| позиция | " + " | ".join(LABEL[n] for n, _ in R) + " | потеря A |", "|---" * (len(R) + 2) + "|"]
    for i, (lo, hi) in enumerate(POS_BINS):
        r0 = R[0][1]["by_story_pos"][i]
        L_.append(f"| {lo}–{hi if hi < 10 ** 6 else '…'} ({100 * r0['token_share']:.1f}%) | " +
                  " | ".join(f"{r['by_story_pos'][i]['open_share']:.3f}" for _, r in R) + f" | {r0['A_loss']:.2f} |")
    L_.append("")
    L_ += ["## Разброс: число открытых групп в блоке (из 8) против независимых случайных гейтов\n",
           "Бином — 8 независимых гейтов с долей блока; Пуассон-бином — со своей долей каждой группы. Среднее по 8 блокам.\n",
           hdr, sep]
    for key, name in (("p0", "доля токенов с 0 открытых"), ("le2", "доля с ≤ 2"), ("ge6", "доля с ≥ 6")):
        row(f"{name}: факт", lambda r, k=key: np.mean([s["obs"][k] for s in r["spread"]["layer"]]))
        row(f"{name}: бином", lambda r, k=key: np.mean([s["binom"][k] for s in r["spread"]["layer"]]))
        row(f"{name}: Пуассон-бином", lambda r, k=key: np.mean([s["poisbin"][k] for s in r["spread"]["layer"]]))
    row("дисперсия / бином (блок)", lambda r: np.mean([s["obs"]["var"] / s["binom"]["var"] for s in r["spread"]["layer"]]), "{:.2f}")
    row("дисперсия / Пуассон-бином (блок)", lambda r: np.mean([s["obs"]["var"] / s["poisbin"]["var"] for s in r["spread"]["layer"]]), "{:.2f}")
    row("дисперсия внутри токена / бином с общей долей (блок)", lambda r: np.mean([s["var_within_token"] / s["binom"]["var"] for s in r["spread"]["layer"]]), "{:.2f}")
    row("дисперсия внутри токена / бином с долей этого токена (блок)", lambda r: np.mean([s["var_within_token"] / s["var_within_binom_own"] for s in r["spread"]["layer"]]), "{:.2f}")
    row("дисперсия суммы по 64 / бином", lambda r: r["spread"]["total_var"] / r["spread"]["total_var_binom"], "{:.2f}")
    row("дисперсия суммы по 64 / Пуассон-бином", lambda r: r["spread"]["total_var"] / r["spread"]["total_var_poisbin"], "{:.2f}")
    row("корреляция числа открытых между блоками (средняя)", lambda r: r["spread"]["block_corr_mean"])
    row("то же внутри токена (отклонения от среднего по токену)", lambda r: r["spread"]["block_corr_within_token"])
    row("дисперсия суммы внутри токена / при независимых блоках", lambda r: r["spread"]["total_var_within_token"] / r["spread"]["total_var_within_token_indep"], "{:.2f}")
    L_.append("")
    L_ += ["## Предсказуемость по одному текущему токену (таблица «токен → группа открыта», как в раундах 4–5)\n",
           f"Таблица — большинство по {N_TR} случайным окнам train (~1M токенов), токены, встреченные < {MIN_SEEN} раз, "
           "— доля группы в целом. Проверка — на окнах val. База — одно решение на группу без учёта токена.\n", hdr, sep]
    row("верных решений, таблица", lambda r: r["token_table"]["acc"])
    row("верных решений, база", lambda r: r["token_table"]["base"])
    row("доля устранённых ошибок базы (κ)", lambda r: r["token_table"]["kappa"])
    for l in range(8):
        L_.append(f"| верных: блок {l}, таблица / база | " + " | ".join(
            f"{r['token_table']['acc_layer'][l]:.3f} / {r['token_table']['base_layer'][l]:.3f}" for _, r in R) + " |")
    row("R² числа открытых в блоке (среднее)", lambda r: np.mean(r["token_table"]["r2_layer"]))
    row("R² суммы открытых по 64", lambda r: r["token_table"]["r2_total"])
    row("доля открытых по таблице", lambda r: r["token_table"]["table_open_share"])
    row("подстановка таблицы: доля открытых", lambda r: r["token_table"]["sub_open_share"])
    row("подстановка таблицы: ΔCE (608 окон)", lambda r: r["token_table"]["val_ce_sub"] - r["token_table"]["val_ce_base"], "{:+.3f}")
    row("токены val, встреченные в train ≥ 5 раз", lambda r: r["token_table"]["seen_share"])
    L_.append("")
    L_.append("Проверка: решения и потери совпадают с tok.npz из dd.ts_eval: " +
              ", ".join(f"{LABEL[n]} {r['check']['decisions_equal_tok_npz']:.4f}" for n, r in R) + ".\n")
    open(os.path.join(out, "r6_adapt_tables.md"), "w", encoding="utf-8").write("\n".join(L_))
    print("\n".join(L_))
    # ---------------- figure: per-block rho, spread vs binomial, story position
    col = {"QG25_s0": "#9cc3ee", "QG25_s1": "#6aa3e3", "QG35_s0": "#2a78d6", "QG50_s0": "#1a4f91",
           "QGsplit25_s0": "#8fdcbd", "QGsplit35_s0": "#1baf7a", "QGsplit50_s0": "#0f6e4c"}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for n, r in R:
        ax[0].plot(range(8), r["difficulty"]["rho_layer"], "-o", color=col[n], lw=1.6, ms=4, label=LABEL[n])
        ax[1].plot(range(8), [s["obs"]["var"] / s["binom"]["var"] for s in r["spread"]["layer"]], "-o", color=col[n],
                   lw=1.6, ms=4, label=LABEL[n])
        ax[2].plot(range(len(POS_BINS)), [b["open_share"] / r["mean_open_share"] for b in r["by_story_pos"]], "-o",
                   color=col[n], lw=1.6, ms=4, label=LABEL[n])
    ax[0].set_title("ρ(открыто в блоке, потеря A)", loc="left")
    ax[0].set_xlabel("блок")
    ax[1].axhline(1, color=INK2, lw=1, ls=":")
    ax[1].set_title("дисперсия числа открытых в блоке / бином", loc="left")
    ax[1].set_xlabel("блок")
    ax[2].axhline(1, color=INK2, lw=1, ls=":")
    ax[2].set_xticks(range(len(POS_BINS)), [f"{lo}–{hi}" if hi < 10 ** 6 else f"{lo}+" for lo, hi in POS_BINS])
    ax[2].set_title("открыто по позиции в истории / среднее прогона", loc="left")
    ax[2].set_xlabel("позиция текущего токена в истории")
    ax[2].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "r6_adapt.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    if sys.argv[1] == "spread":
        for r in sys.argv[2:]:
            respread(r)
    elif sys.argv[1] == "run":
        for r in sys.argv[2:]:
            analyse(r)
            torch.cuda.empty_cache()
    else:
        report(*sys.argv[2:])
