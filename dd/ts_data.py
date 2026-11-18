"""TinyStories (roneneldan/TinyStories, TinyStories-{train,valid}.txt) -> byte-level BPE token streams.

Tokenizer: our own byte-level BPE, vocab 8192, trained on the whole train split with HF `tokenizers`
(id 0 = <pad>, never emitted -- the model treats 0 as padding; id 1 = <|endoftext|>, written BEFORE every story,
so the model's document mask (segments start at id 1) keeps attention inside one story).
Streams: <out>/train.bin, <out>/val.bin (uint16), <out>/freq.npy (train token counts), <out>/meta.json,
<out>/tokenizer.json.  Text is read and encoded in chunks, so host memory stays small.
python -m dd.ts_data [data_ts] [vocab]"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

EOT = "<|endoftext|>"


def stories(path, chunk_bytes=1 << 24):
    """Yield stories (str) from a TinyStories .txt file (stories separated by <|endoftext|> lines)."""
    buf = ""
    with open(path, encoding="utf-8", errors="replace") as fh:
        while True:
            s = fh.read(chunk_bytes)
            if not s:
                break
            buf += s
            parts = buf.split(EOT)
            buf = parts.pop()
            for p in parts:
                p = p.strip()
                if p:
                    yield p
    if buf.strip():
        yield buf.strip()


def batched(it, n):
    b = []
    for x in it:
        b.append(x)
        if len(b) == n:
            yield b
            b = []
    if b:
        yield b


def main(out="data_ts", vocab=8192):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    vocab = int(vocab)
    raw = os.path.join(out, "raw")
    tr_txt, va_txt = os.path.join(raw, "TinyStories-train.txt"), os.path.join(raw, "TinyStories-valid.txt")
    tok_path = os.path.join(out, "tokenizer.json")
    t0 = time.time()
    if not os.path.exists(tok_path):
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tr = trainers.BpeTrainer(vocab_size=vocab, min_frequency=2, special_tokens=["<pad>", EOT],
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
        tok.train_from_iterator(stories(tr_txt), trainer=tr)
        tok.save(tok_path)
        print(f"tokenizer trained in {time.time() - t0:.0f}s", flush=True)
    tok = Tokenizer.from_file(tok_path)
    assert tok.token_to_id("<pad>") == 0 and tok.token_to_id(EOT) == 1
    meta = dict(vocab=tok.get_vocab_size(), pad=0, eot=1, tokenizer="byte-level BPE (HF tokenizers), trained on train")
    freq = np.zeros(tok.get_vocab_size(), np.int64)
    for split, path in (("val", va_txt), ("train", tr_txt)):
        t1 = time.time()
        n_tok = n_st = 0
        with open(os.path.join(out, f"{split}.bin"), "wb") as fo:
            for b in batched(stories(path), 20000):
                enc = tok.encode_batch(b)
                arr = np.concatenate([np.concatenate([[1], e.ids]) for e in enc]).astype(np.uint16)
                fo.write(arr.tobytes())
                n_tok += arr.size
                n_st += len(b)
                if split == "train":
                    freq += np.bincount(arr, minlength=freq.size)
        meta[f"{split}_tokens"], meta[f"{split}_stories"] = int(n_tok), int(n_st)
        print(f"{split}: {n_st} stories, {n_tok / 1e6:.1f}M tokens ({n_tok / n_st:.0f}/story) in {time.time() - t1:.0f}s",
              flush=True)
    np.save(os.path.join(out, "freq.npy"), freq)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
