"""Round 28, stage 3: tests of dd.r28_long and dd.mix_data_ext.

(1) where data_mix stopped: simulate dd.mix_data.take on stub documents and check that the formula of mix_data_ext gives
exactly the number of documents the generator consumed, and that the cut check accepts the true stop and rejects a
wrong one; (2) the lr of the continuation: warmup to the end-of-stage-1 lr, then constant, equal to the last lr of the
first phase; (3) hash routing in the R28 model computes the r21 MoE-hash model exactly (r21 weights copied in), and
starts as A-mix; (4) checkpoints of dd.r28_long load back (dense in the r21 format, hash in the r28 format).
(3) and (4) need the r21 files and a GPU (skipped otherwise).
"""
import math
import os
import types

import numpy as np
import pytest
import torch

from dd import mix_data, mix_data_ext

torch.backends.cuda.matmul.allow_tf32 = False
R21 = all(os.path.exists(p) for p in ("runs_vast/Amix_s0/model.pt", "runs_vast/r21_books/books.pt",
                                       "runs_vast/mix_hash/model_bf16.pt")) and torch.cuda.is_available()


# ------------------------------------------------------------------------------------------------ data stop point
class StubTok:
    """document i -> ids 2 + (i * 7 + t) % 40000 for t < len_i (deterministic, no 0 / 1 inside)"""

    def encode_batch(self, docs):
        return [types.SimpleNamespace(ids=[2 + (int(d.split(":")[0]) * 7 + t) % 40000 for t in range(int(d.split(":")[1]))])
                for d in docs]


def stub_docs(n, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(n):
        yield f"{i}:{int(rng.integers(20, 400))}"


class Counting:
    def __init__(self, it):
        self.it, self.n = it, 0

    def __iter__(self):
        return self

    def __next__(self):
        x = next(self.it)
        self.n += 1
        return x


@pytest.mark.parametrize("seed,n_train,n_val", [(0, 600_000, 2_000), (1, 1_234_567, 5_000), (2, 333_333, 100)])
def test_stop_point_formula(tmp_path, seed, n_train, n_val):
    tok = StubTok()
    g = Counting(stub_docs(100_000, seed))
    freq = np.zeros(mix_data.V, np.int64)
    with open(tmp_path / "t.bin", "wb") as ft, open(tmp_path / "v.bin", "wb") as fv:
        got = mix_data.take(g, tok, n_train, n_val, ft, fv, freq, "stub")
    n = mix_data_ext.last_train_doc(got["train_docs"])
    assert math.ceil(n / 1000) * 1000 == g.n
    # the cut check: true stop accepted, a stop one batch later rejected
    stream = np.fromfile(tmp_path / "t.bin", dtype=np.uint16)
    docs = list(stub_docs(g.n + 1000, seed))
    k0 = g.n
    mix_data_ext.check_cut(docs[k0 - 1000:k0], n, k0, tok, stream)
    with pytest.raises(AssertionError):
        mix_data_ext.check_cut(docs[k0:k0 + 1000], n + 1000, k0 + 1000, tok, stream)


# ------------------------------------------------------------------------------------------------ lr
def test_lr_continuation():
    from dd.r28_long import args_defaults, lr_cont, lr_first
    a = args_defaults(types.SimpleNamespace())
    end = lr_first(a, 10000, 10000)
    assert abs(end - 0.1 * a.lr) < 1e-15
    assert abs(lr_first(a, 9999, 10000) - end) < 1e-10
    assert abs(lr_cont(a, 0) - end / a.warmup) < 1e-15
    assert all(abs(lr_cont(a, s) - end) < 1e-15 for s in (a.warmup - 1, a.warmup, 5000, 29999))


# ------------------------------------------------------------------------------------------------ hash routing
@pytest.mark.skipif(not R21, reason="r21 files / GPU missing")
def test_hash_equals_r21_hash():
    from dd.r21_moe import load_moe
    from dd.r28_long import args_defaults, new_model
    dev = torch.device("cuda")
    a = args_defaults(types.SimpleNamespace())
    h21 = load_moe("runs_vast/mix_hash", dev=dev)
    m, _ = new_model(a, "hash", dev)
    st = {k: v for k, v in h21.base.state_dict().items()}
    missing = m.base.load_state_dict(st, strict=False)
    assert not missing.missing_keys
    with torch.no_grad():
        for j in range(m.M):
            m.w1[j].copy_(h21.experts[j].w1)
            m.w3[j].copy_(h21.experts[j].w3)
            m.w2[j].copy_(h21.experts[j].w2)
        assert torch.equal(m.hash_table, h21.hash_table)
    x = torch.randint(2, 50000, (2, 257), device=dev)
    x[:, 0] = 1
    with torch.no_grad():
        a_, _ = m.eval()(x)
        b_, aux = h21.eval()(x)
        _, es, _ = m(x, keep=True)
    assert all(torch.equal(es[j], aux["eid"][j]) for j in range(m.M))
    assert (a_ - b_).abs().max() < 1e-3


@pytest.mark.skipif(not R21, reason="r21 files / GPU missing")
def test_hash_step0_is_amix():
    from dd.r21_moe import MoE, load_base
    from dd.r28_long import args_defaults, new_model
    dev = torch.device("cuda")
    a = args_defaults(types.SimpleNamespace())
    m, books = new_model(a, "hash", dev)
    ref = MoE(load_base(a.init_from, dev), "dense", books["N"], books=books).to(dev)
    x = torch.randint(2, 50000, (2, 129), device=dev)
    x[:, 0] = 1
    with torch.no_grad():
        assert (m.eval()(x)[0] - ref.eval()(x)[0]).abs().max() < 1e-3


# ------------------------------------------------------------------------------------------------ checkpoints
@pytest.mark.skipif(not R21, reason="r21 files / GPU missing")
@pytest.mark.parametrize("model", ["dense", "hash"])
def test_save_load_roundtrip(tmp_path, model):
    from dd.r21_moe import load_moe
    from dd.r28_long import args_defaults, load, new_model, save
    dev = torch.device("cuda")
    a = args_defaults(types.SimpleNamespace(out=str(tmp_path), smoke=True, pl_first=str(tmp_path / "pl")))
    m, _ = new_model(a, model, dev)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.01 * torch.randn_like(p))
    os.makedirs(tmp_path / model)
    save(m, model, str(tmp_path / model / "model_80_bf16.pt"))
    m2 = load(a, model, 80, dev)
    for (n1, p1), (n2, p2) in zip(m.named_parameters(), m2.named_parameters()):
        assert n1 == n2 and torch.equal(p1.detach().to(torch.bfloat16).float(), p2.detach())
    if model == "dense":
        m3 = load_moe(str(tmp_path / model), dev=dev, fname="model_80_bf16.pt")
        x = torch.randint(2, 50000, (1, 65), device=dev)
        x[:, 0] = 1
        with torch.no_grad():
            assert (m2(x)[0] - m3(x)[0]).abs().max() < 1e-4
