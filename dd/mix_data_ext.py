"""Round 28, stage 3: more training data of the r21 mix, continuing each source right after the documents data_mix
used (so nothing of data_mix train or val is repeated). Same tokenizer, same sources and document order, every 100th
document skipped (it would have been val), equal token shares, stream layout [wiki][code][ts] as data_mix.

Where data_mix stopped, per source: dd.mix_data.take reads documents in batches of 1000 and stops at the end of the
batch in which both train and val are complete; train completes last for every source (val is 1% of the documents), so
with T train documents the last train document has index n - 1, where n - ceil(n / 100) = T and n - 1 is not a val
index; the generator consumed k0 = ceil(n / 1000) * 1000 documents. Check: the last (cut) document of the data_mix
train stream of the source must be a prefix of document n - 1 re-encoded, and the document before it must match fully.
Output (data_mix_ext/): train.bin (uint16), freq.npy, meta.json (bounds, per-source counts, k0), hashes.json,
tokenizer.json (checked equal to data_mix's).
    python -m dd.mix_data_ext [out] [train_per_src]"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

from .mix_data import SOURCES, TRAIN_PER_SRC, VAL_EVERY, code_docs, plain_encoder, take, ts_docs, wiki_docs
from .wiki_data import V, build_tokenizer, sha256

BASE = "data_mix"


def last_train_doc(T):
    """n such that the documents 0..n-1 hold T train documents and document n - 1 is a train document"""
    n = T
    while not (n - math.ceil(n / VAL_EVERY) == T and (n - 1) % VAL_EVERY != 0):
        n += 1
    return n


def check_cut(batch, n, k0, tok, stream_src):
    """batch = documents k0-1000 .. k0-1; the source's data_mix train stream must end with document n-1 (cut) preceded
    by the previous train document"""
    enc = tok.encode_batch(batch)
    i_last = n - 1 - (k0 - len(batch))
    assert 0 <= i_last < len(batch)
    ids_last = np.concatenate([[1], np.asarray(enc[i_last].ids, dtype=np.int64)])
    eots = np.flatnonzero(stream_src == 1)
    p = eots[-1]
    tail = stream_src[p:].astype(np.int64)
    assert tail.size <= ids_last.size and np.array_equal(ids_last[:tail.size], tail), "last document does not match"
    j = i_last - 1
    while (k0 - len(batch) + j) % VAL_EVERY == 0:              # the previous TRAIN document
        j -= 1
    prev = np.concatenate([[1], np.asarray(enc[j].ids, dtype=np.int64)])
    seg = stream_src[eots[-2]:p].astype(np.int64)
    assert np.array_equal(prev, seg), "the document before the last does not match"
    return dict(last_doc=int(n - 1), last_cut_tokens=int(tail.size), last_doc_tokens=int(ids_last.size))


def main(out="data_mix_ext", n_train=TRAIN_PER_SRC):
    n_train = int(n_train)
    os.makedirs(out, exist_ok=True)
    raw = os.path.join(out, "raw")
    t0 = time.time()
    tok = plain_encoder(build_tokenizer(os.path.join(out, "tokenizer.json")))
    tj = [json.load(open(os.path.join(d, "tokenizer.json"), encoding="utf-8")) for d in (out, BASE)]
    assert all(tj[0][k] == tj[1][k] for k in ("model", "added_tokens", "pre_tokenizer", "decoder")), "tokenizer differs"
    meta0 = json.load(open(os.path.join(BASE, "meta.json")))
    base_train = np.memmap(os.path.join(BASE, "train.bin"), dtype=np.uint16, mode="r")
    gens = {"wiki": wiki_docs(raw), "code": code_docs(raw), "ts": ts_docs(raw)}
    freq = np.zeros(V, np.int64)
    stats, tb = {}, [0]
    with open(os.path.join(out, "train.bin"), "wb") as ft, open(os.devnull, "wb") as fv:
        for si, s in enumerate(SOURCES):
            T = meta0["per_source"][s]["train_docs"]
            n = last_train_doc(T)
            k0 = math.ceil(n / 1000) * 1000
            g = gens[s]
            batch = []
            for k in range(k0):                                         # skip what data_mix consumed
                d = next(g)
                if k >= k0 - 1000:
                    batch.append(d)
            b0, b1 = meta0["train_bounds"][si], meta0["train_bounds"][si + 1]
            chk = check_cut(batch, n, k0, tok, np.asarray(base_train[max(b0, b1 - 2_000_000):b1]))
            print(f"{s}: data_mix used documents 0..{k0 - 1} (last train document {n - 1}); check ok {chk} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            got = take(g, tok, n_train, 0, ft, fv, freq, s)               # k restarts at 0: k0 % 100 == 0, same val pattern
            got.update(start_doc=k0, check=chk)
            stats[s] = got
            tb.append(tb[-1] + got["train"])
            print(f"{s}: {json.dumps(got)} ({time.time() - t0:.0f}s)", flush=True)
    np.save(os.path.join(out, "freq.npy"), freq)
    assert freq[0] == 0 and freq[1] == sum(stats[s]["train_docs"] for s in SOURCES)
    meta = dict(vocab=V, pad=0, eot=1, tokenizer=meta0["tokenizer"], sources=list(SOURCES), train_bounds=tb,
                train_tokens=tb[-1], per_source=stats, continues=BASE,
                source=meta0["source"] + f"; documents after those of {BASE} (start_doc per source)")
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    json.dump(dict(sha256={"train.bin": sha256(os.path.join(out, "train.bin"))}, meta={"train_bounds": tb}),
              open(os.path.join(out, "hashes.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "per_source"}), f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
