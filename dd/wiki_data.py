"""English Wikipedia (wikimedia/wikipedia, snapshot 20231101.en on HF) -> GPT-2 BPE token streams for dd.ts_train.

Tokenizer: GPT-2 byte-level BPE (HF `gpt2`), ids shifted by 2 so that our conventions hold: 0 = <pad> (never emitted,
excluded from the loss), 1 = <|endoftext|> written BEFORE every article (document mask boundary); GPT-2 id i -> i + 2,
vocabulary 50258.  Our TinyStories BPE (8192) is not used: it was trained on children's stories and fragments
encyclopedic text (names, terms, numbers); the fragmentation of both on the val text is recorded in meta.json.
Articles of shards 0 .. n-1; every 100th article goes to val, the rest to train.
Streams: <out>/train.bin, <out>/val.bin (uint16), freq.npy, meta.json, tokenizer.json, hashes.json (sha256).
python -m dd.wiki_data [data_wiki] [n_shards]"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

REPO, SNAP = "wikimedia/wikipedia", "20231101.en"
N_FILES = 41
VAL_EVERY = 100
V = 50258


def build_tokenizer(path):
    from tokenizers import Tokenizer
    j = json.loads(Tokenizer.from_pretrained("gpt2").to_str())
    vocab = j["model"]["vocab"]
    eot = vocab.pop("<|endoftext|>")
    assert eot == 50256 and len(vocab) == 50256
    new = {"<pad>": 0, "<|endoftext|>": 1}
    new.update({t: i + 2 for t, i in vocab.items()})
    j["model"]["vocab"] = new
    sp = dict(single_word=False, lstrip=False, rstrip=False, normalized=False, special=True)
    j["added_tokens"] = [dict(id=0, content="<pad>", **sp), dict(id=1, content="<|endoftext|>", **sp)]
    tok = Tokenizer.from_str(json.dumps(j))
    assert tok.get_vocab_size() == V and tok.token_to_id("<pad>") == 0 and tok.token_to_id("<|endoftext|>") == 1
    ref = Tokenizer.from_pretrained("gpt2")
    s = "The Battle of Hastings was fought on 14 October 1066 between the Norman-French army of William."
    assert [i + 2 for i in ref.encode(s).ids] == tok.encode(s).ids and tok.decode(tok.encode(s).ids) == s
    tok.save(path)
    return tok


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main(out="data_wiki", n_shards=4):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    n_shards = int(n_shards)
    os.makedirs(out, exist_ok=True)
    raw = os.path.join(out, "raw")
    t0 = time.time()
    files = [hf_hub_download(REPO, f"{SNAP}/train-{i:05d}-of-{N_FILES:05d}.parquet", repo_type="dataset", local_dir=raw)
             for i in range(n_shards)]
    print(f"downloaded {n_shards} shards in {time.time() - t0:.0f}s", flush=True)
    tok = build_tokenizer(os.path.join(out, "tokenizer.json"))
    freq = np.zeros(V, np.int64)
    n = dict(train=0, val=0)
    arts = dict(train=0, val=0)
    k = 0
    sample = []
    t1 = time.time()
    with open(os.path.join(out, "train.bin"), "wb") as ft, open(os.path.join(out, "val.bin"), "wb") as fv:
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=4000, columns=["text"]):
                texts = [t for t in batch.column(0).to_pylist() if t and t.strip()]
                enc = tok.encode_batch(texts)
                tr, va = [], []
                for t, e in zip(texts, enc):
                    (va if k % VAL_EVERY == 0 else tr).append(np.concatenate([[1], e.ids]))
                    if k % VAL_EVERY == 0 and len(sample) < 2000:
                        sample.append(t)
                    k += 1
                for split, lst, fh in (("train", tr, ft), ("val", va, fv)):
                    if lst:
                        arr = np.concatenate(lst).astype(np.uint16)
                        fh.write(arr.tobytes())
                        n[split] += arr.size
                        arts[split] += len(lst)
                        if split == "train":
                            freq += np.bincount(arr, minlength=V)
            print(f"{os.path.basename(f)}: train {n['train'] / 1e6:.0f}M val {n['val'] / 1e6:.1f}M tokens "
                  f"({time.time() - t1:.0f}s)", flush=True)
    np.save(os.path.join(out, "freq.npy"), freq)
    # fragmentation of encyclopedic text: our TinyStories BPE vs GPT-2 (tokens per 1000 characters of val text)
    frag = {}
    chars = sum(len(t) for t in sample)
    frag["gpt2_tokens_per_1000_chars"] = round(1000 * sum(len(e.ids) for e in tok.encode_batch(sample)) / chars, 1)
    if os.path.exists(os.path.join("data_ts", "tokenizer.json")):
        from tokenizers import Tokenizer
        ts = Tokenizer.from_file(os.path.join("data_ts", "tokenizer.json"))
        frag["tinystories_bpe8k_tokens_per_1000_chars"] = round(1000 * sum(len(e.ids) for e in ts.encode_batch(sample)) / chars, 1)
    meta = dict(vocab=V, pad=0, eot=1, tokenizer="GPT-2 byte-level BPE, ids + 2 (0 = <pad>, 1 = <|endoftext|>)",
                source=f"{REPO} {SNAP}, shards 0..{n_shards - 1} of {N_FILES}, every {VAL_EVERY}th article -> val",
                train_tokens=int(n["train"]), val_tokens=int(n["val"]), train_articles=int(arts["train"]),
                val_articles=int(arts["val"]), fragmentation_on_val_sample=frag)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    json.dump(dict(sha256={f: sha256(os.path.join(out, f)) for f in ("train.bin", "val.bin")},
                   meta={k: meta[k] for k in ("vocab", "train_tokens", "val_tokens", "train_articles", "val_articles")}),
              open(os.path.join(out, "hashes.json"), "w"), indent=1)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
