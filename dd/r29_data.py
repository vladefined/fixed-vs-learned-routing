"""Round 29 data: two new sources for the r21 mix model, GPT-2 BPE ids + 2 as data_mix (0 = <pad>, 1 = <|endoftext|>
before every document), plain encoder (no added tokens inside documents).
  med  close source: English texts of a narrow domain -- PubMed abstracts (MedRAG/pubmed, chunk/pubmed23n*.jsonl,
       title + abstract; open, no login)
  ru   far source: another language -- Russian Wikipedia (wikimedia/wikipedia 20231101.ru, one parquet shard)
Every VAL_EVERY-th document -> val (until VAL tokens), the rest -> train (until TRAIN tokens); the last document of each
split is cut. Outputs data_r29/<src>/{train.bin, val.bin, meta.json}.
    python -m dd.r29_data med|ru
"""
import json
import os
import sys
import time

import numpy as np

from .mix_data import batched, plain_encoder
from .wiki_data import V, sha256

OUT = "data_r29"
TRAIN, VAL, VAL_EVERY = 16_500_000, 500_000, 30
RAW = "A:/AI/hf_cache/r29_raw"


def med_docs():
    from huggingface_hub import hf_hub_download
    for i in range(1, 60):
        f = hf_hub_download("MedRAG/pubmed", f"chunk/pubmed23n{i:04d}.jsonl", repo_type="dataset", local_dir=RAW)
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                t = (r.get("title") or "").strip()
                c = (r.get("content") or "").strip()
                if c:
                    yield (t + "\n" + c) if t else c


def ru_docs():
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().dataset_info("wikimedia/wikipedia", files_metadata=True)
    fs = sorted((s.size, s.rfilename) for s in info.siblings if s.rfilename.startswith("20231101.ru/"))
    for _, name in fs[:3]:                                     # the smallest shards first
        f = hf_hub_download("wikimedia/wikipedia", name, repo_type="dataset", local_dir=RAW)
        for batch in pq.ParquetFile(f).iter_batches(batch_size=1000, columns=["text"]):
            for t in batch.column(0).to_pylist():
                if t and t.strip():
                    yield t


def main(src):
    from tokenizers import Tokenizer
    t0 = time.time()
    out = os.path.join(OUT, src)
    os.makedirs(out, exist_ok=True)
    tok = plain_encoder(Tokenizer.from_file("data_mix/tokenizer.json"))
    docs = dict(med=med_docs, ru=ru_docs)[src]()
    got = dict(train=0, val=0, train_docs=0, val_docs=0, chars=0)
    k = 0
    freq = np.zeros(V, np.int64)
    with open(os.path.join(out, "train.bin"), "wb") as ft, open(os.path.join(out, "val.bin"), "wb") as fv:
        for b in batched(docs, 500):
            enc = tok.encode_batch(b)
            tr, va = [], []
            for txt, e in zip(b, enc):
                ids = np.concatenate([[1], np.asarray(e.ids, dtype=np.int64)])
                if k % VAL_EVERY == 0:
                    if got["val"] < VAL:
                        ids = ids[:VAL - got["val"]]
                        va.append(ids)
                        got["val"] += ids.size
                        got["val_docs"] += 1
                elif got["train"] < TRAIN:
                    ids = ids[:TRAIN - got["train"]]
                    tr.append(ids)
                    got["train"] += ids.size
                    got["train_docs"] += 1
                    got["chars"] += len(txt)
                k += 1
            for lst, fh, split in ((tr, ft, "train"), (va, fv, "val")):
                if lst:
                    arr = np.concatenate(lst).astype(np.uint16)
                    fh.write(arr.tobytes())
                    if split == "train":
                        freq += np.bincount(arr, minlength=V)
            if got["train"] >= TRAIN and got["val"] >= VAL:
                break
            if k % 20000 < 500:
                print(f"{src}: {k} docs, train {got['train'] / 1e6:.1f}M val {got['val'] / 1e6:.2f}M "
                      f"({time.time() - t0:.0f}s)", flush=True)
    assert got["train"] >= TRAIN and got["val"] >= VAL, got
    np.save(os.path.join(out, "freq.npy"), freq)
    meta = dict(src=src, **got, tokens_per_1000_chars=1000 * got["train"] / max(1, got["chars"]),
                sha256={f: sha256(os.path.join(out, f)) for f in ("train.bin", "val.bin")}, time=time.time() - t0)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"{src}: done {json.dumps(meta)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
