"""Build train/val/test splits of the synthetic language, packed into fixed-length rows.

Usage:  python -m dd.data --n_train 2000000 --out data
Rows are T tokens; each sentence starts with <bos>; sentences never straddle rows;
the tail of a row is <pad> (id 0).  Val/test also carry per-token annotation arrays.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from multiprocessing import Pool

import numpy as np

from . import grammar as G2
from . import grammar3 as G3
from .grammar import bracketed

class _V4:
    build_vocab = staticmethod(G3.build_vocab)
    flatten = staticmethod(G3.flatten4)

    def __init__(self, k_choices):
        self.k_choices = k_choices

    def Generator(self, seed):
        return G3.Generator4(seed, k_choices=self.k_choices)


class _V5(_V4):
    def Generator(self, seed):
        return G3.Generator5(seed, k_choices=self.k_choices)


LANG = {2: G2, 3: G3}
VERSION = 2

T = 128
MAX_SENT = 100
ANN = ["tree_depth", "emb_depth", "is_head", "opens", "closes", "agree", "attractor",
       "agree_dist", "copy", "close", "flip", "kind"]


def lang(version, k_choices=(3, 4)):
    if version == 5:
        return _V5(k_choices)
    return _V4(k_choices) if version == 4 else LANG[version]


def _gen_chunk(args):
    seed, n, keep_meta, version, max_sent, k_choices = args
    L = lang(version, k_choices)
    vocab = L.build_vocab()
    g = L.Generator(seed)
    out = []
    while len(out) < n:
        s = g.sentence()
        f, spans = L.flatten(s, vocab)
        if len(f["tok"]) > max_sent:
            continue
        if keep_meta:
            f["lp"] = g.lp
            out.append((f, spans, bracketed(s)))
        else:
            out.append((tuple(f["tok"]),))
    return out


def generate(n, seed, keep_meta, workers=20, version=2, max_sent=MAX_SENT, k_choices=(3, 4)):
    per = (n + workers - 1) // workers
    with Pool(workers) as p:
        chunks = p.map(_gen_chunk, [(seed * 1000 + i, per, keep_meta, version, max_sent, k_choices)
                                    for i in range(workers)])
    return [x for c in chunks for x in c][:n]


def pack(sents, keep_meta):
    """sents: list of token lists (and annotation dicts). Greedy row packing."""
    global ANN
    if keep_meta:
        ANN = [k for k in sents[0] if k not in ("tok", "lp")]
    rows, ann_rows, sid_rows, pos_rows = [], [], [], []
    cur, cur_ann, cur_sid, cur_pos = [], {k: [] for k in ANN}, [], []

    def flush():
        pad = T - len(cur)
        rows.append(cur + [0] * pad)
        sid_rows.append(cur_sid + [-1] * pad)
        pos_rows.append(cur_pos + [-1] * pad)
        if keep_meta:
            ann_rows.append({k: cur_ann[k] + [0 if k not in ("flip", "kind", "nsw", "nk", "qm", "qi") else -1] * pad
                             for k in ANN})

    for i, s in enumerate(sents):
        toks = s["tok"] if keep_meta else list(s)
        if len(cur) + len(toks) > T:
            flush()
            cur, cur_ann, cur_sid, cur_pos = [], {k: [] for k in ANN}, [], []
        cur += toks
        cur_sid += [i] * len(toks)
        cur_pos += list(range(len(toks)))
        if keep_meta:
            for k in ANN:
                cur_ann[k] += s[k]
    if cur:
        flush()
    out = {"tok": np.array(rows, dtype=np.int16), "sid": np.array(sid_rows, dtype=np.int32),
           "pos": np.array(pos_rows, dtype=np.int16)}
    if keep_meta:
        for k in ANN:
            out[k] = np.array([r[k] for r in ann_rows], dtype=np.int16)
        out["sent_lp"] = np.array([s["lp"] for s in sents], dtype=np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=2_000_000)
    ap.add_argument("--n_val", type=int, default=20_000)
    ap.add_argument("--n_test", type=int, default=20_000)
    ap.add_argument("--out", default="data")
    ap.add_argument("--version", type=int, default=2)
    ap.add_argument("--max_sent", type=int, default=MAX_SENT)
    ap.add_argument("--k", default="3,4", help="numbers of names (v4)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    kc = tuple(int(x) for x in a.k.split(","))
    L = lang(a.version, kc)
    vocab = L.build_vocab()
    # acceptance rate of the length cap (needed to renormalise sentence log-probs)
    g = L.Generator(12345)
    n_try, n_acc = 50000, 0
    for _ in range(n_try):
        n_acc += len(L.flatten(g.sentence(), vocab)[0]["tok"]) <= a.max_sent
    log_acc = float(np.log(n_acc / n_try))
    print(f"length-cap acceptance {n_acc / n_try:.5f}")

    # val/test first (with metadata); train excludes every val/test sentence.
    held = generate(int((a.n_val + a.n_test) * 1.05), seed=7, keep_meta=True, version=a.version,
                    max_sent=a.max_sent, k_choices=kc)
    seen, val, test = set(), [], []
    for f, spans, br in held:
        f["lp"] -= log_acc            # log p(s | len <= max_sent)
        key = tuple(f["tok"])
        if key in seen:
            continue
        seen.add(key)
        (val if len(val) < a.n_val else test).append((f, spans, br))
        if len(test) >= a.n_test:
            break
    held_keys = set(seen)
    print(f"val {len(val)}  test {len(test)} (unique held-out sentences)")

    train_raw = generate(int(a.n_train * 1.02), seed=1, keep_meta=False, version=a.version,
                         max_sent=a.max_sent, k_choices=kc)
    train, dup_held, seen_tr = [], 0, set()
    for (toks,) in train_raw:
        if toks in held_keys:
            dup_held += 1
            continue
        seen_tr.add(toks)
        train.append(toks)
        if len(train) >= a.n_train:
            break
    print(f"train {len(train)} sentences, {len(seen_tr)} unique; dropped {dup_held} that "
          f"collided with val/test")

    tr = pack(train, keep_meta=False)
    np.save(os.path.join(a.out, "train_tok.npy"), tr["tok"])
    for name, split in (("val", val), ("test", test)):
        p = pack([f for f, _, _ in split], keep_meta=True)
        np.savez_compressed(os.path.join(a.out, f"{name}.npz"), **p)
        with open(os.path.join(a.out, f"{name}_trees.pkl"), "wb") as fh:
            pickle.dump([(sp, br) for _, sp, br in split], fh)
    n_tok = int((tr["tok"] != 0).sum())
    meta = dict(vocab=vocab.itos, cls=[vocab.cls[w] for w in vocab.itos],
                flip={str(k): v for k, v in vocab.flip.items()}, T=T,
                train_rows=int(tr["tok"].shape[0]), train_tokens=n_tok,
                fill=float(n_tok / tr["tok"].size), n_train_sent=len(train),
                n_unique_train=len(seen_tr), version=a.version, log_accept=log_acc, k_choices=kc)
    with open(os.path.join(a.out, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("vocab", "cls", "flip")}))


if __name__ == "__main__":
    main()
