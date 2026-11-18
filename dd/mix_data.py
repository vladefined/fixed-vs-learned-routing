"""Round 21 data: a mix of English Wikipedia, Python code and TinyStories with one tokenizer (GPT-2 byte-level BPE,
ids + 2 as in dd.wiki_data: 0 = <pad>, never emitted; 1 = <|endoftext|> written BEFORE every document = the document
mask boundary), equal token shares, the source of every token known.

Sources, documents in file order; every 100th document -> val, the rest -> train; a source is complete when it has
exactly TRAIN_PER_SRC train tokens and VAL_PER_SRC val tokens (the last document is cut there; val documents after
that are dropped, not moved to train):
  0 wiki  wikimedia/wikipedia 20231101.en, parquet shards 0, 1, ... (as dd.wiki_data)
  1 code  codeparrot/codeparrot-clean (Python files from GitHub, deduplicated, open; bigcode/the-stack-smol is
          gated), file-000000000001.json.gz, ...  field "content"
  2 ts    roneneldan/TinyStories, TinyStories-train.txt (the file of data_ts)
Stream layout: [wiki][code][tinystories], train and val alike.  Training windows are drawn at random from the whole
stream (dd.ts_train --sampler perm), so every batch has ~1/3 of each source.  Source of token i of a split:
searchsorted(meta["<split>_bounds"], i, side="right") - 1; val also as val_src.bin (uint8).
Outputs: train.bin, val.bin (uint16), val_src.bin (uint8), freq.npy (train counts), meta.json, tokenizer.json,
hashes.json (sha256 of the streams).
    python -m dd.mix_data [data_mix] [train_per_src] [val_per_src]
    python -m dd.mix_data test <out> <train_per_src> <val_per_src>   -> local sources (pipeline tests only)"""
from __future__ import annotations

import glob
import gzip
import json
import os
import sys
import time

import numpy as np

from .wiki_data import V, build_tokenizer, sha256

SOURCES = ("wiki", "code", "ts")
TRAIN_PER_SRC = 166_000_000            # 3 x 166M = 498M tokens >= 30k steps x 32 windows x 513 tokens (stages 1 + 2)
VAL_PER_SRC = 1_500_000
VAL_EVERY = 100
WIKI = ("wikimedia/wikipedia", "20231101.en", 41)
CODE = ("codeparrot/codeparrot-clean", 54)
TS = ("roneneldan/TinyStories", "TinyStories-train.txt")


def wiki_docs(raw):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    repo, snap, n = WIKI
    for i in range(n):
        f = hf_hub_download(repo, f"{snap}/train-{i:05d}-of-{n:05d}.parquet", repo_type="dataset", local_dir=raw)
        for batch in pq.ParquetFile(f).iter_batches(batch_size=2000, columns=["text"]):
            for t in batch.column(0).to_pylist():
                if t and t.strip():
                    yield t
        os.remove(f)


def code_docs(raw):
    from huggingface_hub import hf_hub_download
    repo, n = CODE
    for i in range(1, n + 1):
        f = hf_hub_download(repo, f"file-{i:012d}.json.gz", repo_type="dataset", local_dir=raw)
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                t = json.loads(line)["content"]
                if t and t.strip():
                    yield t
        os.remove(f)


def ts_docs(raw):
    from huggingface_hub import hf_hub_download
    from .ts_data import stories
    repo, fn = TS
    local = os.path.join("data_ts", "raw", fn)
    f = local if os.path.exists(local) else hf_hub_download(repo, fn, repo_type="dataset", local_dir=raw)
    yield from stories(f)


def test_sources():
    """local stand-ins for the pipeline tests: Wikipedia val articles (decoded from data_wiki/val.bin), Python files
    of the venv, TinyStories-valid.txt"""
    from tokenizers import Tokenizer
    from .ts_data import stories
    tok = Tokenizer.from_file(os.path.join("data_wiki", "tokenizer.json"))

    def wiki():
        a = np.fromfile(os.path.join("data_wiki", "val.bin"), dtype=np.uint16)
        cut = np.flatnonzero(a == 1)
        for s, e in zip(cut, list(cut[1:]) + [a.size]):
            yield tok.decode(a[s + 1:e].astype(np.int64).tolist())

    def code():
        for f in sorted(glob.glob(os.path.join(".venv", "Lib", "site-packages", "**", "*.py"), recursive=True)):
            t = open(f, encoding="utf-8", errors="replace").read()
            if t.strip():
                yield t

    def ts():
        yield from stories(os.path.join("data_ts", "raw", "TinyStories-valid.txt"))
    return {"wiki": wiki(), "code": code(), "ts": ts()}


def plain_encoder(tok):
    """the same BPE without the added tokens: a literal "<pad>" or "<|endoftext|>" in a text (Python code of NLP
    projects has them) is encoded as ordinary text, so ids 0 / 1 never appear inside a document"""
    from tokenizers import Tokenizer
    j = json.loads(tok.to_str())
    j["added_tokens"] = []
    return Tokenizer.from_str(json.dumps(j))


def batched(it, n):
    b = []
    for x in it:
        b.append(x)
        if len(b) == n:
            yield b
            b = []
    if b:
        yield b


def take(docs, tok, n_train, n_val, ft, fv, freq, log_name):
    """encode documents until n_train train / n_val val tokens; write them; return counts"""
    k = 0
    got = dict(train=0, val=0, train_docs=0, val_docs=0)
    t0 = time.time()
    for b in batched(docs, 1000):
        enc = tok.encode_batch(b)
        tr, va = [], []
        for e in enc:
            ids = np.concatenate([[1], np.asarray(e.ids, dtype=np.int64)])
            if k % VAL_EVERY == 0:
                if got["val"] < n_val:                        # the last val document is cut: equal val shares
                    ids = ids[:n_val - got["val"]]
                    va.append(ids)
                    got["val"] += ids.size
                    got["val_docs"] += 1
            elif got["train"] < n_train:
                ids = ids[:n_train - got["train"]]
                tr.append(ids)
                got["train"] += ids.size
                got["train_docs"] += 1
            k += 1
        for lst, fh, split in ((tr, ft, "train"), (va, fv, "val")):
            if lst:
                arr = np.concatenate(lst).astype(np.uint16)
                fh.write(arr.tobytes())
                if split == "train":
                    freq += np.bincount(arr, minlength=V)
        if got["train"] >= n_train and got["val"] >= n_val:
            break
        if k % 50000 < 1000:
            print(f"  {log_name}: {k} docs, train {got['train'] / 1e6:.1f}M val {got['val'] / 1e6:.2f}M "
                  f"({time.time() - t0:.0f}s)", flush=True)
    assert got["train"] >= n_train and got["val"] >= n_val, f"{log_name}: source exhausted {got}"
    return got


def main(out="data_mix", n_train=TRAIN_PER_SRC, n_val=VAL_PER_SRC, test=False):
    n_train, n_val = int(n_train), int(n_val)
    os.makedirs(out, exist_ok=True)
    raw = os.path.join(out, "raw")
    t0 = time.time()
    tok = plain_encoder(build_tokenizer(os.path.join(out, "tokenizer.json")))
    gens = test_sources() if test else {"wiki": wiki_docs(raw), "code": code_docs(raw), "ts": ts_docs(raw)}
    freq = np.zeros(V, np.int64)
    stats, tb, vb = {}, [0], [0]
    with open(os.path.join(out, "train.bin"), "wb") as ft, open(os.path.join(out, "val.bin"), "wb") as fv:
        for s in SOURCES:
            g = take(gens[s], tok, n_train, n_val, ft, fv, freq, s)
            stats[s] = g
            tb.append(tb[-1] + g["train"])
            vb.append(vb[-1] + g["val"])
            print(f"{s}: {json.dumps(g)} ({time.time() - t0:.0f}s)", flush=True)
    val_src = np.repeat(np.arange(len(SOURCES), dtype=np.uint8), np.diff(vb))
    val_src.tofile(os.path.join(out, "val_src.bin"))
    np.save(os.path.join(out, "freq.npy"), freq)
    assert freq[0] == 0, "<pad> must never be emitted"
    assert freq[1] == sum(stats[s]["train_docs"] for s in SOURCES), "id 1 only at document starts"
    meta = dict(vocab=V, pad=0, eot=1, tokenizer="GPT-2 byte-level BPE, ids + 2 (0 = <pad>, 1 = <|endoftext|>)",
                sources=list(SOURCES), train_bounds=tb, val_bounds=vb, train_tokens=tb[-1], val_tokens=vb[-1],
                per_source=stats, test=bool(test),
                source=("local test sources" if test else
                        f"{WIKI[0]} {WIKI[1]} | {CODE[0]} | {TS[0]} {TS[1]}; every {VAL_EVERY}th document -> val"))
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    json.dump(dict(sha256={f: sha256(os.path.join(out, f)) for f in ("train.bin", "val.bin", "val_src.bin")},
                   meta={k: meta[k] for k in ("vocab", "train_bounds", "val_bounds")}),
              open(os.path.join(out, "hashes.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "per_source"}), f"({time.time() - t0:.0f}s)", flush=True)


def source_of(meta, split, pos):
    """source index of token positions pos (numpy) of a split"""
    return np.searchsorted(np.asarray(meta[f"{split}_bounds"]), pos, side="right") - 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        main(sys.argv[2], sys.argv[3], sys.argv[4], test=True)
    else:
        main(*sys.argv[1:])
