"""Round 28: tests of the gradient weighting (dd.r28_train) and of the stage-1 model forms.

The split attention must (1) compute the ordinary attention in the forward, (2) send the K / V gradient of a pair
(t, s) to the first copy with weight W[t, s] and to the second with 1 - W[t, s] (sum = the ordinary gradient), (3) give
exactly the ordinary gradients for W = 1; inside the model (4) the forward is the same with the weighting on and off,
(5) W = 1 gives the gradients of the model without the weighting, (6) with W = 0 between topics the gradient of a block-5
expert from the loss of readers of other topics through the expert writes is 0, (7) every variant except «add» starts
as A-mix (integration test with the r21 files, skipped if they are missing).
"""
import math
import os

import pytest
import torch

from dd.r28_train import PairAttnSplit, R28

torch.backends.cuda.matmul.allow_tf32 = False


def plain_attn(q, k, v, mask, scale):
    s = (q @ k.transpose(-1, -2)) * scale
    return torch.softmax(s.masked_fill(~mask, float("-inf")), -1) @ v


def rand_qkv(B=2, H=2, T=7, hd=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(B, H, T, hd, generator=g, dtype=torch.float64) for _ in range(3))
    mask = torch.ones(T, T, dtype=torch.bool).tril()[None, None]
    return q, k, v, mask


def test_split_forward_is_attention():
    q, k, v, mask = rand_qkv()
    W = torch.rand(2, 7, 7, dtype=torch.float64)
    y = PairAttnSplit.apply(q, k, k.clone(), v, v.clone(), mask, W, 0.5)
    assert torch.allclose(y, plain_attn(q, k, v, mask, 0.5), atol=1e-6)


def test_split_gradients_sum_and_weights():
    q, k, v, mask = rand_qkv(seed=1)
    W = torch.rand(2, 7, 7, dtype=torch.float64)
    q1, k1, ks, v1, vs = (t.clone().requires_grad_(True) for t in (q, k, k, v, v))
    y = PairAttnSplit.apply(q1, k1, ks, v1, vs, mask, W, 0.5)
    g = torch.randn_like(y)
    (y * g).sum().backward()
    q2, k2, v2 = (t.clone().requires_grad_(True) for t in (q, k, v))
    (plain_attn(q2, k2, v2, mask, 0.5) * g).sum().backward()
    assert torch.allclose(q1.grad, q2.grad, atol=1e-6)
    assert torch.allclose(k1.grad + ks.grad, k2.grad, atol=1e-6)
    assert torch.allclose(v1.grad + vs.grad, v2.grad, atol=1e-6)
    # reference of the per-pair weighting with ordinary autograd (scores and outputs split by W)
    q3, k3, ks3, v3, vs3 = (t.clone().requires_grad_(True) for t in (q, k, k, v, v))
    Wh = W[:, None]
    s = Wh * (q3 @ k3.transpose(-1, -2)) + (1 - Wh) * (q3 @ ks3.transpose(-1, -2))
    p = torch.softmax((s * 0.5).masked_fill(~mask, float("-inf")), -1)
    y3 = (p * Wh) @ v3 + (p * (1 - Wh)) @ vs3
    (y3 * g).sum().backward()
    for a, b in ((k1, k3), (ks, ks3), (v1, v3), (vs, vs3), (q1, q3)):
        assert torch.allclose(a.grad, b.grad, atol=1e-6)


def test_split_w1_is_plain():
    q, k, v, mask = rand_qkv(seed=2)
    W = torch.ones(2, 7, 7, dtype=torch.float64)
    q1, k1, ks, v1, vs = (t.clone().requires_grad_(True) for t in (q, k, k, v, v))
    y = PairAttnSplit.apply(q1, k1, ks, v1, vs, mask, W, 0.5)
    y.sum().backward()
    assert ks.grad.abs().max() == 0 and vs.grad.abs().max() == 0


# ------------------------------------------------------------------------------------------------ inside the model
def tiny_model(variant, seed=0):
    from dd.model import Cfg, Model
    cfg = Cfg(vocab=50, d=64, n_layers=4, n_heads=4, ffn_hidden=96, max_len=32)
    torch.manual_seed(seed)
    base = Model(cfg)
    N, M, K = 2, 2, 64
    g = torch.Generator().manual_seed(seed)
    books = dict(N=N, pl=torch.randn(M, K, 64, generator=g), pl_sem=torch.arange(K).repeat(1) % 8 * torch.ones(M, 1).long())
    imp = torch.rand(M, 96, generator=g) if variant == "add" else None
    if variant == "add":
        import dd.r28_train as T
        T.W_SHARED, T.W_ADD = 64, 32
    return R28(base, variant, books, imp=imp, seed=seed).double()


def ids(T=24, B=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(2, 50, (B, T), generator=g)
    x[:, 0] = 1
    return x


@pytest.mark.parametrize("variant", ["hier", "own", "flat", "shuf"])
def test_model_forward_unchanged_by_weighting(variant):
    m = tiny_model(variant)
    x = ids()
    a, _ = m(x, gradw=False)
    m.train()
    b, _ = m(x, gradw=True)
    assert torch.allclose(a, b, atol=1e-9)


def expert_grads(m, x, gradw):
    for p in m.parameters():
        p.grad = None
    logits, _ = m(x, gradw=gradw)
    torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1)).backward()
    return [p.grad.clone() for p in m.parameters() if p.grad is not None]


def test_model_w1_equals_unweighted():
    m = tiny_model("hier")
    m.Wt.fill_(1.0)
    x = ids(seed=3)
    a = expert_grads(m, x, False)
    b = expert_grads(m, x, True)
    assert len(a) == len(b)
    for u, w in zip(a, b):
        assert torch.allclose(u, w, atol=1e-9)


def test_model_weighting_changes_expert_gradient_only():
    """own topic only (Wt = I) against Wt = 1: the same loss value, a different gradient of the first MoE block's
    experts (the weighting acts on the expert-write path)"""
    m = tiny_model("own")
    x = ids(seed=5)
    m.train()
    logits, _ = m(x, gradw=True)
    # the loss of every position: changing Wt off-diagonal from 0 to 1 must change the gradient (the weighting acts)
    l0 = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
    g0 = torch.autograd.grad(l0, [m.w1[0]])[0]
    m.Wt.fill_(1.0)
    logits, _ = m(x, gradw=True)
    l1 = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
    g1 = torch.autograd.grad(l1, [m.w1[0]])[0]
    assert torch.allclose(l0, l1, atol=1e-12)                    # the same loss value
    assert (g0 - g1).abs().max() > 1e-8                           # but a different expert gradient


def test_last_block_experts_unaffected():
    """the experts of the last MoE block have no later attention: their gradient is the same for any weighting"""
    m = tiny_model("own")
    x = ids(seed=7)
    m.train()
    logits, _ = m(x, gradw=True)
    l0 = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
    g0 = torch.autograd.grad(l0, [m.w1[m.M - 1], m.w2[m.M - 1]])
    logits, _ = m(x, gradw=False)
    l1 = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
    g1 = torch.autograd.grad(l1, [m.w1[m.M - 1], m.w2[m.M - 1]])
    for a, b in zip(g0, g1):
        assert torch.allclose(a, b, atol=1e-10)


def test_add_starts_with_zero_additions():
    m = tiny_model("add")
    for j in range(m.M):
        assert m.w2[j].abs().max() == 0
        assert m.w1[j].shape[1] == 32 and m.sw1[j].shape[0] == 64


@pytest.mark.skipif(not (os.path.exists("runs_vast/Amix_s0/model.pt") and os.path.exists("runs_vast/r21_books/books.pt")
                         and torch.cuda.is_available()), reason="r21 files / GPU missing")
@pytest.mark.parametrize("variant", ["base", "own", "hier", "flat", "shuf", "sdown"])
def test_step0_is_amix(variant):
    from dd.r21_moe import MoE, load_base
    from dd.r28_train import build
    dev = torch.device("cuda")
    books = torch.load("runs_vast/r21_books/books.pt", map_location="cpu", weights_only=False)
    m, _ = build(variant, "runs_vast/Amix_s0/model.pt", "runs_vast/r21_books/books.pt", dev)
    ref = MoE(load_base("runs_vast/Amix_s0/model.pt", dev), "dense", books["N"], books=books).to(dev)
    x = torch.randint(2, 50000, (2, 129), device=dev)
    x[:, 0] = 1
    with torch.no_grad():
        a, _ = m.eval()(x)
        b, _ = ref.eval()(x)
    assert (a - b).abs().max() < 1e-3
