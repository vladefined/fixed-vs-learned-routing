"""Round 28, stage 4: tests of the routes by the output (dd.r28_train variants out / inout / mixed / oracle,
dd.r28_route).

(1) out: the expert = table[group[argmax of the frozen lens at the block input]], computed by hand; (2) the lens is
frozen: changing the model's own final norm and embedding does not change the route; (3) the lens copies are not in
the state dict (rebuilt from A-mix, never rounded to bf16 on save); (4) inout uses the pair (current group, predicted
group); (5) mixed: blocks 5-6 hash of the current token, 7-8 as out; (6) oracle: the group of the true next token;
(7) the route is the same in the training forward and in the measurement forward from a cached state (b0 > 0);
(8) spherical k-means gives valid, deterministic groups; (9) with the real tables every variant starts as A-mix
(needs the r21 files, the tables of dd.r28_route prep and a GPU).
"""
import os

import pytest
import torch

from dd.r28_train import R28, ROUTE4, ROUTE_TABLES

torch.backends.cuda.matmul.allow_tf32 = False
V, G, N, M = 50, 4, 2, 4


def tiny(variant, seed=0):
    from dd.model import Cfg, Model
    cfg = Cfg(vocab=V, d=64, n_layers=N + M, n_heads=4, ffn_hidden=96, max_len=32)
    torch.manual_seed(seed)
    base = Model(cfg)
    g = torch.Generator().manual_seed(seed)
    books = dict(N=N, pl=torch.randn(M, 64, 64, generator=g), pl_sem=torch.randint(0, 8, (M, 64), generator=g),
                 hash=torch.randint(0, 8, (M, V), generator=g))
    books["route"] = dict(tok_group=torch.randint(0, G, (V,), generator=g), out=torch.randint(0, 8, (M, G), generator=g),
                          pair=torch.randint(0, 8, (M, G * G), generator=g), next=torch.randint(0, 8, (M, G), generator=g))
    return R28(base, variant, books).eval(), books


def ids(T=24, B=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(2, V, (B, T), generator=g)
    x[:, 0] = 1
    return x


def lens_by_hand(m, x):
    xf = x.float()
    h = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + m.base.nf.eps) * m.base.nf.w.detach()
    return (h.to(torch.bfloat16) @ m.base.emb.weight.detach().to(torch.bfloat16).t()).argmax(-1)


def states(m, x):
    with torch.no_grad():
        _, es, xs = m(x, keep=True)
    return es, xs


def test_out_route_by_hand():
    m, b = tiny("out")
    x = ids()
    es, xs = states(m, x)
    for j in range(M):
        want = b["route"]["out"][j][b["route"]["tok_group"][lens_by_hand(m, xs[j])]]
        assert torch.equal(es[j], want)


def test_lens_frozen_and_not_saved():
    m, _ = tiny("out")
    x = ids()
    es, xs = states(m, x)
    with torch.no_grad():
        m.base.nf.w.mul_(-3.0)
        m.base.emb.weight.add_(torch.randn_like(m.base.emb.weight))
    for j in range(M):
        assert torch.equal(m.route(j, xs[j], x), es[j])
    sd = m.state_dict()
    assert not any(k.startswith(("lens_", "tok_group", "out_table", "pair_table", "next_table")) for k in sd)


def test_inout_pairs():
    m, b = tiny("inout")
    x = ids()
    es, xs = states(m, x)
    tg = b["route"]["tok_group"]
    for j in range(M):
        want = b["route"]["pair"][j][tg[x] * G + tg[lens_by_hand(m, xs[j])]]
        assert torch.equal(es[j], want)


def test_mixed():
    m, b = tiny("mixed")
    x = ids()
    es, xs = states(m, x)
    for j in range(M):
        if j < 2:
            want = b["hash"][j][x]
        else:
            want = b["route"]["out"][j][b["route"]["tok_group"][lens_by_hand(m, xs[j])]]
        assert torch.equal(es[j], want)


def test_oracle_next_token():
    m, b = tiny("oracle")
    x = ids()
    es, _ = states(m, x)
    tg = b["route"]["tok_group"]
    for j in range(M):
        assert torch.equal(es[j][:, :-1], b["route"]["next"][j][tg[x[:, 1:]]])


@pytest.mark.parametrize("variant", ROUTE4)
def test_route_same_from_cached_state(variant):
    """the measurement forward (dd.r28_measure.Cache.rise: b0 = N + j, x = cached block input) routes as training"""
    m, _ = tiny(variant)
    x = ids()
    es, xs = states(m, x)
    for j in range(M):
        with torch.no_grad():
            _, es2, _ = m(x, b0=N + j, x=xs[j])
        assert all(torch.equal(es2[jj], es[jj]) for jj in range(j, M))


def test_kmeans_groups():
    from dd.r28_route import kmeans_sph
    X = torch.nn.functional.normalize(torch.randn(500, 16, generator=torch.Generator().manual_seed(1)), dim=1)
    a1, C1, cos1 = kmeans_sph(X, 8, iters=20, seed=3)
    a2, _, _ = kmeans_sph(X, 8, iters=20, seed=3)
    assert torch.equal(a1, a2) and a1.min() >= 0 and a1.max() < 8 and torch.bincount(a1, minlength=8).min() > 0
    assert torch.allclose(C1.norm(dim=1), torch.ones(8), atol=1e-5) and 0 < cos1 <= 1


@pytest.mark.skipif(not (os.path.exists("runs_vast/Amix_s0/model.pt") and os.path.exists(ROUTE_TABLES)
                         and torch.cuda.is_available()), reason="r21 files / route tables / GPU missing")
@pytest.mark.parametrize("variant", ROUTE4)
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
        assert (m.eval()(x)[0] - ref.eval()(x)[0]).abs().max() < 1e-3
        _, es, _ = m(x, keep=True)
    assert all(int(e.min()) >= 0 and int(e.max()) < 8 for e in es.values())
