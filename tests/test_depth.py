import math

import pytest
import torch
import torch.nn.functional as F

from dd.model import Cfg, Model, apply_rope

NORM = dict(budget=True, norm_state=True)
VARIANTS = {
    "B": dict(**NORM, depth_groups=256, update="min"),
    "B_dd": dict(**NORM, depth_groups=256, update="dd"),
    "B_pull": dict(**NORM, depth_groups=256, update="pull"),
    "C": dict(**NORM, depth_groups=256, update="min", update_src="ffn"),
    "E": dict(**NORM, depth_groups=8, norm_groups=8, update="min"),
    "F": dict(**NORM, depth_groups=1, update="min"),
    "G": dict(**NORM, depth_groups=8, norm_groups=8, update="min", attn_depth_bias=True),
    "B_legacy": dict(budget=True, depth_groups=256, update="full", logit_clamp=0.0),
}


def small(**kw):
    if kw.get("depth_groups") == 256:
        kw = {**kw, "depth_groups": 64}
    return Cfg(vocab=50, d=64, n_layers=5, n_heads=8, ffn_hidden=96, max_len=32, **kw)


def randomize(model, gate_scale=3.0):
    """Non-trivial weights: Q of depth embeddings, gate biases, slopes."""
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith(".Q") or n.endswith("slope"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
            if n.endswith("wd.bias"):
                p.copy_(torch.randn(p.shape, generator=g) * gate_scale)
            if n.endswith("wd.weight"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)


@pytest.mark.parametrize("name", list(VARIANTS))
@pytest.mark.parametrize("gate_scale", [0.0, 3.0, 30.0])
def test_depth_never_exceeds_block_index(name, gate_scale):
    torch.manual_seed(1)
    m = Model(small(**VARIANTS[name]))
    randomize(m, gate_scale)
    idx = torch.randint(1, 50, (4, 32))
    _, aux = m(idx, collect=True)
    prev = torch.zeros_like(aux["depths"][0])
    for l, dep in enumerate(aux["depths"]):
        assert (dep <= l + 1).all(), f"depth {dep.max()} > {l + 1} after block {l}"
        assert (dep >= prev).all(), "depth must be non-decreasing"
        assert (aux["dd"][l] >= 0).all()
        prev = dep


@pytest.mark.parametrize("name", list(VARIANTS))
def test_zero_dd_leaves_state_unchanged(name):
    torch.manual_seed(2)
    m = Model(small(**VARIANTS[name]))
    randomize(m)
    with torch.no_grad():
        for b in m.blocks:          # sigmoid(-1e4) == 0 exactly -> dd == 0
            b.wd.weight.zero_()
            b.wd.bias.fill_(-1e4)
    idx = torch.randint(1, 50, (3, 32))
    _, aux = m(idx, collect=True)
    x0 = aux["xs"][0]
    for l in range(1, len(aux["xs"])):
        assert torch.allclose(aux["xs"][l], x0, atol=1e-6, rtol=0), f"state changed in block {l - 1} with dd=0"
        assert torch.count_nonzero(aux["depths"][l - 1]) == 0


@pytest.mark.parametrize("name", ["A", "A_norm"] + list(VARIANTS))
def test_causality(name):
    torch.manual_seed(3)
    m = Model(small(**VARIANTS.get(name, {"A_norm": dict(norm_state=True)}.get(name, {}))))
    if not name.startswith("A"):
        randomize(m)
    idx = torch.randint(1, 50, (2, 32))
    idx2 = idx.clone()
    t = 17
    idx2[:, t:] = torch.randint(1, 50, (2, 32 - t))
    l1, a1 = m(idx, collect=True)
    l2, a2 = m(idx2, collect=True)
    assert torch.allclose(l1[:, :t], l2[:, :t], atol=1e-5)
    if not name.startswith("A"):
        assert torch.allclose(a1["depth"][:, :t], a2["depth"][:, :t], atol=1e-6)
    # and the future does change (sanity): logits after t differ
    assert not torch.allclose(l1[:, t:], l2[:, t:])


def test_gradient_reaches_future_insensitive_positions():
    """d logits[t] / d emb[s] == 0 for s > t (causality via autograd)."""
    torch.manual_seed(4)
    m = Model(small(**VARIANTS["G"]))
    randomize(m)
    idx = torch.randint(1, 50, (1, 32))
    x = m.emb(idx).detach().requires_grad_(True)
    h = m.emb.register_forward_hook(lambda mod, inp, out: x)
    logits, _ = m(idx)
    logits[0, 10].sum().backward()
    h.remove()
    assert x.grad[0, 11:].abs().max() == 0
    assert x.grad[0, :11].abs().max() > 0


@pytest.mark.parametrize("use_doc", [False, True])
def test_attention_depth_bias_matches_explicit_formula(use_doc):
    """Fused implementation == softmax(qk/sqrt(hd) + slope_h*(d_q - d_k)) with causal mask."""
    torch.manual_seed(5)
    cfg = small(**VARIANTS["G"])
    m = Model(cfg)
    randomize(m)
    blk = m.blocks[2]
    B, T, d, H, hd = 2, 32, cfg.d, cfg.n_heads, cfg.d // cfg.n_heads
    z = torch.randn(B, T, d)
    depth = torch.rand(B, T, H) * 3
    rope = m.rope(T, z.device)
    from dd.model import doc_causal_mask
    idx = torch.randint(2, 50, (B, T))
    idx[:, 0] = 1
    idx[0, 11] = 1
    idx[1, 20] = 1
    mask = doc_causal_mask(idx) if use_doc else None
    got = blk.attn(z, rope, depth, mask)
    q, k, v = blk.qkv(z).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
    q, k = apply_rope(q, *rope), apply_rope(k, *rope)
    s = q @ k.transpose(-1, -2) / math.sqrt(hd)
    dq = depth.transpose(1, 2)[..., :, None]
    dk = depth.transpose(1, 2)[..., None, :]
    s = s + blk.slope.view(1, H, 1, 1) * (dq - dk)
    allowed = mask if use_doc else torch.ones(T, T, dtype=torch.bool).tril()
    s = s.masked_fill(~allowed, float("-inf"))
    ref = blk.o((F.softmax(s, -1) @ v).transpose(1, 2).reshape(B, T, d))
    assert torch.allclose(got, ref, atol=1e-5)


def test_budget_accumulates():
    """A coordinate that skips early blocks can take a larger step later (allowed grows)."""
    torch.manual_seed(6)
    m = Model(small(**VARIANTS["B"]))
    with torch.no_grad():
        for l, b in enumerate(m.blocks):
            b.wd.weight.zero_()
            b.wd.bias.fill_(-1e4 if l < 4 else 1e4)   # skip blocks 0-3, spend all in block 4
    idx = torch.randint(1, 50, (1, 8))
    _, aux = m(idx, collect=True)
    assert torch.allclose(aux["dd"][4], torch.full_like(aux["dd"][4], 5.0))
    assert torch.allclose(aux["depth"], torch.full_like(aux["depth"], 5.0))


def test_cost_equals_mean_final_depth():
    from dd.model import lm_loss
    torch.manual_seed(7)
    m = Model(small(**VARIANTS["B"], lam=0.1))
    randomize(m)
    idx = torch.randint(1, 50, (2, 32))
    _, _, cost, _, aux = lm_loss(m, idx)
    assert torch.allclose(cost, aux["depth"].mean(), atol=1e-5)


@pytest.mark.parametrize("name", ["A", "G"])
def test_sentences_are_isolated(name):
    """With the document mask, changing a previous sentence does not affect the next one."""
    torch.manual_seed(8)
    m = Model(small(**VARIANTS.get(name, {})))
    if name != "A":
        randomize(m)
    idx = torch.randint(2, 50, (2, 32))
    idx[:, 0] = 1
    idx[:, 12] = 1                       # second sentence starts at 12
    idx2 = idx.clone()
    idx2[:, 1:12] = torch.randint(2, 50, (2, 11))
    l1, _ = m(idx)
    l2, _ = m(idx2)
    assert torch.allclose(l1[:, 12:], l2[:, 12:], atol=1e-5)


@pytest.mark.parametrize("name", ["B", "B_dd", "B_pull", "C", "E", "F", "G"])
def test_state_is_normalised(name):
    torch.manual_seed(9)
    cfg = small(**VARIANTS[name])
    m = Model(cfg)
    randomize(m)
    idx = torch.randint(1, 50, (2, 32))
    _, aux = m(idx, collect=True)
    G = cfg.norm_groups
    for x in aux["xs"]:
        xg = x.view(*x.shape[:-1], G, cfg.d // G)
        assert torch.allclose(xg.pow(2).mean(-1), torch.ones(()), atol=1e-4)


def test_min_bounds_update_and_depth_counts_fully():
    """update=min: coordinate change uses min(dd,1) while depth gets the full dd."""
    torch.manual_seed(10)
    m = Model(small(**VARIANTS["B"]))
    with torch.no_grad():
        for l, b in enumerate(m.blocks):
            b.wd.weight.zero_()
            b.wd.bias.fill_(-1e4 if l < 3 else 1e4)   # skip 3 blocks, then dd = 4 in block 3
    idx = torch.randint(1, 50, (1, 8))
    _, aux = m(idx, collect=True)
    assert torch.allclose(aux["dd"][3], torch.full_like(aux["dd"][3], 4.0))
    assert torch.allclose(aux["depths"][3], torch.full_like(aux["depths"][3], 4.0))


def test_pull_with_full_gate_overwrites_state():
    from dd.model import normalize
    torch.manual_seed(11)
    m = Model(small(**VARIANTS["B_pull"]))
    with torch.no_grad():
        for b in m.blocks:
            b.wd.weight.zero_()
            b.wd.bias.fill_(1e4)                     # dd = allowed >= 1 -> alpha = 1
    blk = m.blocks[0]
    captured = {}
    h = blk.w2.register_forward_hook(lambda mod, i, o: captured.__setitem__("u", o))
    h2 = blk.o.register_forward_hook(lambda mod, i, o: captured.__setitem__("a", o))
    idx = torch.randint(1, 50, (1, 8))
    _, aux = m(idx, collect=True)
    h.remove(); h2.remove()
    want = normalize(torch.tanh(captured["a"] + captured["u"]), floor=0.1)
    assert torch.allclose(aux["xs"][1], want, atol=5e-4)   # x+(t-x) rounding, amplified by normalising small t


# ----------------------------------------------------------------------------- round 3: H, I, J, A_groupnorm
R3 = {
    "H": dict(budget=True, norm_state=True, depth_groups=8, norm_groups=8, update="min", budget_global=True),
    "I": dict(budget=True, depth_groups=8, update="min", budget_global=True, rel_cost=True),
    "J": dict(budget=True, depth_groups=8, update="min", budget_global=True, rel_cost=True, state_noise=0.02),
}


@pytest.mark.parametrize("name", list(R3))
def test_global_budget_bounds_and_spend_all_early(name):
    torch.manual_seed(12)
    cfg = small(**R3[name])
    m = Model(cfg)
    m.eval()
    with torch.no_grad():
        for l, b in enumerate(m.blocks):
            b.wd.weight.zero_()
            b.wd.bias.fill_(1e4 if l == 0 else -1e4)   # spend the whole budget in block 0
    idx = torch.randint(1, 50, (2, 16))
    _, aux = m(idx, collect=True)
    R = cfg.n_layers
    assert torch.allclose(aux["dd"][0], torch.full_like(aux["dd"][0], float(R)))
    for l, dep in enumerate(aux["depths"]):
        assert (dep <= R).all()
        assert torch.allclose(dep, torch.full_like(dep, float(R)))
    # exhausted budget: later blocks cannot change the state even with an open gate
    with torch.no_grad():
        for b in m.blocks:
            b.wd.bias.fill_(1e4)
    _, aux = m(idx, collect=True)
    for l in range(2, len(aux["xs"])):
        assert torch.allclose(aux["xs"][l], aux["xs"][1], atol=1e-6)


@pytest.mark.parametrize("name", list(R3))
def test_r3_zero_dd_unchanged_and_causal(name):
    torch.manual_seed(13)
    m = Model(small(**R3[name]))
    m.eval()
    randomize(m)
    idx = torch.randint(1, 50, (2, 32))
    idx2 = idx.clone()
    idx2[:, 20:] = torch.randint(1, 50, (2, 12))
    l1, a1 = m(idx, collect=True)
    l2, _ = m(idx2)
    assert torch.allclose(l1[:, :20], l2[:, :20], atol=1e-5)
    with torch.no_grad():
        for b in m.blocks:
            b.wd.weight.zero_()
            b.wd.bias.fill_(-1e4)
    _, aux = m(idx, collect=True)
    for l in range(1, len(aux["xs"])):
        assert torch.allclose(aux["xs"][l], aux["xs"][0], atol=1e-6)


def test_rel_cost_formula_and_floor():
    from dd.model import lm_loss
    torch.manual_seed(14)
    cfg = small(**R3["I"], lam=1.0)
    m = Model(cfg)
    m.eval()
    randomize(m)
    with torch.no_grad():
        for b in m.blocks:
            b.s_ema.copy_(torch.rand(cfg.d) + 0.5)
            b.s_ema[0] = 0.0                         # below the floor -> uses 1e-3 * mean
    idx = torch.randint(1, 50, (2, 16))
    _, _, cost, _, aux = lm_loss(m, idx)
    _, a2 = m(idx, collect=True)
    want = 0.0
    for l, b in enumerate(m.blocks):
        dx = a2["xs"][l + 1] - a2["xs"][l]
        sfl = torch.maximum(b.s_ema, 1e-3 * b.s_ema.mean())
        want = want + (dx.abs() / sfl).mean()
    assert torch.allclose(cost, want, rtol=1e-4)


def test_noise_only_in_training():
    torch.manual_seed(15)
    m = Model(small(**R3["J"]))
    idx = torch.randint(1, 50, (2, 16))
    m.eval()
    assert torch.equal(m(idx)[0], m(idx)[0])
    m.train()
    assert not torch.equal(m(idx)[0], m(idx)[0])


def test_groupnorm_baseline_is_normalised_per_group():
    torch.manual_seed(16)
    cfg = small(norm_state=True, norm_groups=8)
    m = Model(cfg)
    _, aux = m(torch.randint(1, 50, (2, 16)), collect=True)
    for x in aux["xs"]:
        xg = x.view(*x.shape[:-1], 8, cfg.d // 8)
        assert torch.allclose(xg.pow(2).mean(-1), torch.ones(()), atol=1e-4)


def test_zero_rel_below_freezes_groups():
    torch.manual_seed(17)
    m = Model(small(**R3["H"]))
    m.eval()
    randomize(m)
    idx = torch.randint(1, 50, (2, 16))
    _, aux = m(idx, collect=True, ctl={"zero_rel_below": 1e9})   # every group counts as standing
    for l in range(1, len(aux["xs"])):
        assert torch.allclose(aux["xs"][l], aux["xs"][0], atol=1e-6)
        assert torch.count_nonzero(aux["depths"][l - 1]) == 0


# ----------------------------------------------------------------------------- variant K (hard-concrete)
K = dict(budget=True, gate_type="hc", depth_groups=8, norm_state=True, norm_groups=8, update="min",
         attn_depth_bias=True)


def randomize_k(m):
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if ".dt_" in n or n.endswith("slope"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)
            if n.endswith("gate_head.bias"):
                p.copy_(torch.randn(p.shape, generator=g) * 2.0)
            if n.endswith("gate_head.weight"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)


@pytest.mark.parametrize("kv", ["K", "Kfree", "Knorm"])
def test_k_eval_gates_are_binary_and_standing_groups_bit_identical(kv):
    torch.manual_seed(20)
    extra = {"K": {}, "Kfree": dict(norm_state=False, hc_tanh=False), "Knorm": dict(hc_tanh=False)}[kv]
    cfg = small(**{**K, **extra})
    m = Model(cfg)
    m.eval()
    randomize_k(m)
    idx = torch.randint(1, 50, (2, 32))
    _, aux = m(idx, collect=True)
    G, gs = cfg.depth_groups, cfg.d // cfg.depth_groups
    for l, mm in enumerate(aux["dd"]):
        assert set(mm.unique().tolist()) <= {0.0, 1.0}
        assert 0 < mm.mean() < 1                                   # mixture of moving and standing groups
        x0 = aux["xs"][l].view(2, 32, G, gs)
        x1 = aux["xs"][l + 1].view(2, 32, G, gs)
        st = mm == 0
        assert torch.equal(x0[st], x1[st]), "standing group changed"
        assert (x0[~st] != x1[~st]).any()
        dep = aux["depths"][l]
        assert torch.equal(dep, dep.round()) and (dep <= l + 1).all()
    assert torch.equal(aux["depth"], torch.stack(aux["dd"]).sum(0))


def test_k_causal():
    torch.manual_seed(21)
    m = Model(small(**K))
    m.eval()
    randomize_k(m)
    idx = torch.randint(1, 50, (2, 32))
    idx2 = idx.clone()
    idx2[:, 19:] = torch.randint(1, 50, (2, 13))
    l1, a1 = m(idx, collect=True)
    l2, a2 = m(idx2, collect=True)
    assert torch.allclose(l1[:, :19], l2[:, :19], atol=1e-5)
    assert torch.equal(a1["depth"][:, :19], a2["depth"][:, :19])


def test_k_open_probability_closed_form():
    import math
    torch.manual_seed(22)
    beta, gam, zeta = 2 / 3, -0.1, 1.1
    z = torch.tensor([-3.0, -1.0, 0.0, 1.5, 4.0])
    u = torch.rand(200000, 5).clamp(1e-6, 1 - 1e-6)
    s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + z) / beta)
    mc = ((s * (zeta - gam) + gam).clamp(0, 1) > 0).float().mean(0)
    closed = torch.sigmoid(z - beta * math.log(-gam / zeta))
    assert torch.allclose(mc, closed, atol=0.01)


def test_k_training_gates_hit_exact_zero_and_one():
    torch.manual_seed(23)
    m = Model(small(**K))
    m.train()
    randomize_k(m)
    _, aux = m(torch.randint(1, 50, (4, 32)))
    allm = torch.cat([x.flatten() for x in aux["dd"]])
    assert (allm == 0).float().mean() > 0.05 and (allm == 1).float().mean() > 0.05
    assert ((allm > 0) & (allm < 1)).any()


def test_k_loss_uses_mu():
    from dd.model import lm_loss
    torch.manual_seed(24)
    m = Model(small(**K))
    idx = torch.randint(1, 50, (2, 16))
    m.eval()
    loss0, ce0, cost0, _, aux = lm_loss(m, idx)
    assert torch.allclose(loss0, ce0)
    with torch.no_grad():
        m.mu.fill_(2.0)
    loss1, ce1, cost1, _, aux1 = lm_loss(m, idx)
    frac = torch.stack(aux1["frac_layer"]).mean()
    assert torch.allclose(loss1 - ce1, 2.0 * frac, atol=1e-5)


# ----------------------------------------------------------------------------- round 4
Q = dict(quant=True, q_delta=0.15, q_levels=8)


def randomize_q(m, scale=3.0):
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("o.weight") or n.endswith("w2.weight"):
                p.copy_(torch.randn(p.shape, generator=g) * scale / math.sqrt(p.shape[1]))


@pytest.mark.parametrize("cost", ["none", "l1", "l0"])
def test_q_state_on_grid_and_standing_coords_bit_identical(cost):
    torch.manual_seed(30)
    cfg = small(**Q, q_cost=cost)
    m = Model(cfg)
    m.eval()
    randomize_q(m)
    idx = torch.randint(1, 50, (2, 32))
    _, aux = m(idx, collect=True)
    D, K = cfg.q_delta, cfg.q_levels
    for l, x in enumerate(aux["xs"]):
        lv = x / D
        assert torch.allclose(lv, lv.round(), atol=1e-4) and lv.abs().max() <= K + 1e-4
        assert torch.equal(x, D * lv.round()), "state is exactly delta * integer level"
    for l, n in enumerate(aux["dd"]):
        x0, x1 = aux["xs"][l], aux["xs"][l + 1]
        assert torch.equal(n, n.round())
        assert torch.allclose(n, ((x1 - x0) / D).abs(), atol=1e-4)
        st = n == 0
        assert st.float().mean() > 0.05 and (~st).float().mean() > 0.05
        assert torch.equal(x0[st], x1[st]), "standing coordinate changed"
        assert torch.equal(aux["rc"][l], (n > 0).float())


def test_q_straight_through_gradient_and_clipping():
    from dd.model import quantize
    v = torch.tensor([0.01, 0.2, -0.4, 1.3, 5.0, -5.0], requires_grad=True)
    xq, lvl = quantize(v, 0.15, 8)
    assert torch.equal(xq, 0.15 * torch.tensor([0., 1., -3., 8., 8., -8.]))  # round(1.3/.15)=9 -> clamp 8
    xq.sum().backward()
    # identity inside [-8.5, 8.5] levels, zero outside (1.3/.15 = 8.67 is outside)
    assert torch.allclose(v.grad, torch.tensor([1., 1., 1., 0., 0., 0.]))


def test_q_causal_and_embedding_gets_gradient():
    from dd.model import lm_loss
    torch.manual_seed(31)
    m = Model(small(**Q, q_cost="l1"))
    randomize_q(m)
    m.eval()
    idx = torch.randint(1, 50, (2, 32))
    idx2 = idx.clone()
    idx2[:, 20:] = torch.randint(1, 50, (2, 12))
    l1, _ = m(idx)
    l2, _ = m(idx2)
    assert torch.allclose(l1[:, :20], l2[:, :20], atol=1e-5)
    m.train()
    with torch.no_grad():
        m.mu.fill_(1.0)
    loss, ce, cost, _, aux = lm_loss(m, idx)
    assert torch.allclose(loss - ce, cost, atol=1e-6)
    tm = (idx != 0).float().unsqueeze(-1)
    want = torch.stack([(n * tm).sum() / (tm.sum() * m.cfg.d) for n in aux["dd"]]).mean()
    assert torch.allclose(cost, want)
    loss.backward()
    assert m.emb.weight.grad.abs().sum() > 0
    assert all(b.w1.weight.grad.abs().sum() > 0 for b in m.blocks)


def test_q_l0_cost_forward_is_share_and_gradient_is_soft():
    from dd.model import lm_loss
    torch.manual_seed(32)
    m = Model(small(**Q, q_cost="l0"))
    randomize_q(m)
    idx = torch.randint(1, 50, (2, 16))
    with torch.no_grad():
        m.mu.fill_(1.0)
    loss, ce, cost, _, aux = lm_loss(m, idx)
    share = torch.stack([(n > 0).float().mean() for n in aux["dd"]]).mean()
    assert torch.allclose(cost, share, atol=1e-6)
    (loss - ce).backward()
    assert m.blocks[0].w2.weight.grad.abs().sum() > 0


def test_ip_scale_is_masked_median_with_floor():
    from dd.model import update_scale_ema
    torch.manual_seed(33)
    cfg = small(budget=True, depth_groups=8, update="min", budget_global=True, rel_cost=True, rel_scale="median")
    m = Model(cfg)
    randomize(m)
    idx = torch.randint(1, 50, (3, 16))
    idx[0, 10:] = 0                                    # padding must not enter the statistics
    _, aux = m(idx)
    update_scale_ema(m, aux)
    msk = idx != 0
    for l, b in enumerate(m.blocks):
        ax = aux["xin"][l][msk].abs()
        assert torch.allclose(b.s_med, ax.median(0).values)
        assert torch.allclose(b.s_mabs, ax.mean(0))
        assert torch.allclose(b.s_floor(), torch.maximum(b.s_med, 0.1 * b.s_mabs))
    with torch.no_grad():
        m.blocks[0].s_med.zero_()
    assert torch.allclose(m.blocks[0].s_floor(), 0.1 * m.blocks[0].s_mabs)


KSTE = dict(K, gate_type="ste")


def test_kste_binary_same_in_train_and_eval_and_standing_bit_identical():
    torch.manual_seed(34)
    cfg = small(**KSTE)
    m = Model(cfg)
    randomize_k(m)
    idx = torch.randint(1, 50, (2, 32))
    m.eval()
    le, ae = m(idx, collect=True)
    m.train()
    lt, at = m(idx, collect=True)
    assert torch.equal(le, lt)
    G, gs = cfg.depth_groups, cfg.d // cfg.depth_groups
    for l, mm in enumerate(ae["dd"]):
        assert set(mm.unique().tolist()) <= {0.0, 1.0} and 0 < mm.mean() < 1
        x0 = ae["xs"][l].view(2, 32, G, gs)
        x1 = ae["xs"][l + 1].view(2, 32, G, gs)
        st = mm == 0
        assert torch.equal(x0[st], x1[st])
        xg = x1[~st]
        assert torch.allclose(xg.pow(2).mean(-1), torch.ones(()), atol=1e-4)


def test_kste_closed_gate_gets_gradient_and_cost_gradient_is_sigmoid():
    from dd.model import lm_loss
    torch.manual_seed(35)
    m = Model(small(**KSTE))
    randomize_k(m)
    with torch.no_grad():
        m.blocks[1].gate_head.bias.fill_(-1.0)          # block 1 fully closed (small weights)
        m.blocks[1].gate_head.weight.mul_(0.01)
    idx = torch.randint(1, 50, (2, 16))
    loss, ce, cost, _, aux = lm_loss(m, idx)
    assert aux["dd"][1].sum() == 0
    ce.backward()
    assert m.blocks[1].gate_head.bias.grad.abs().sum() > 0, "closed gates must still learn (STE)"
    # cost: forward = hard share, gradient = d mean sigmoid(z)
    m.zero_grad()
    with torch.no_grad():
        m.mu.fill_(1.0)
    loss, ce, cost, _, aux = lm_loss(m, idx)
    frac = torch.stack(aux["frac_layer"])
    hard = torch.stack([(d > 0).float().mean() for d in aux["dd"]])
    assert torch.allclose(frac, hard)


@pytest.mark.parametrize("kind", ["A_groupnorm_free", "E_free", "G_free"])
def test_free_coords_skip_normalisation(kind):
    torch.manual_seed(36)
    base = {"A_groupnorm_free": dict(norm_state=True, norm_groups=8),
            "E_free": dict(**NORM, depth_groups=8, norm_groups=8, update="min"),
            "G_free": dict(**NORM, depth_groups=8, norm_groups=8, update="min", attn_depth_bias=True)}[kind]
    cfg = small(**base, free_per_group=2)
    m = Model(cfg)
    m.eval()
    if cfg.budget:
        randomize(m)
    idx = torch.randint(1, 50, (2, 16))
    _, aux = m(idx, collect=True)
    G, gs = 8, cfg.d // 8
    for x in aux["xs"]:
        xg = x.view(2, 16, G, gs)
        assert torch.allclose(xg[..., :-2].pow(2).mean(-1), torch.ones(()), atol=1e-4)
    # free coordinates follow x + gate * tanh(a + u) exactly, without rescaling: |change| <= 1 per block
    for l in range(len(aux["xs"]) - 1):
        f0 = aux["xs"][l].view(2, 16, G, gs)[..., -2:]
        f1 = aux["xs"][l + 1].view(2, 16, G, gs)[..., -2:]
        assert (f1 - f0).abs().max() <= 1.0 + 1e-5
        assert (f1 - f0).abs().max() > 1e-3
    e = m.emb(idx).view(2, 16, G, gs)[..., -2:]
    assert torch.allclose(aux["xs"][0].view(2, 16, G, gs)[..., -2:], e)


# ----------------------------------------------------------------------------- round 5
QGC = dict(quant=True, q_delta=0.15, q_levels=8, q_cost="l0", depth_groups=8)


def randomize_qg(m, scale=3.0, gate_scale=1.0):
    randomize_q(m, scale)
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("gate_head.weight"):
                p.copy_(torch.randn(p.shape, generator=g) * gate_scale)
            if n.endswith("gate_head.bias"):
                p.zero_()
            if n.endswith("dt_a") or n.endswith("dt_f"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)


@pytest.mark.parametrize("split", [False, True])
def test_qg_gate_binary_standing_groups_bit_identical_and_on_grid(split):
    torch.manual_seed(40)
    cfg = small(**QGC, qg="gate", qg_split=split)
    m = Model(cfg)
    randomize_qg(m)
    idx = torch.randint(1, 50, (2, 32))
    m.eval()
    le, ae = m(idx, collect=True)
    m.train()
    lt, _ = m(idx, collect=True)
    assert torch.equal(le, lt), "deterministic gate: train == eval"
    G, gs, D = 8, cfg.d // 8, cfg.q_delta
    prev = torch.zeros(2, 32, G)
    for l, mm in enumerate(ae["dd"]):
        assert set(mm.unique().tolist()) <= {0.0, 1.0} and 0 < mm.mean() < 1
        x0 = ae["xs"][l].view(2, 32, G, gs)
        x1 = ae["xs"][l + 1].view(2, 32, G, gs)
        st = mm == 0
        assert torch.equal(x0[st], x1[st]), "standing group changed"
        assert torch.equal(ae["xs"][l + 1], D * (ae["xs"][l + 1] / D).round())
        real = (x0 != x1).any(-1).float()
        assert torch.equal(ae["rc"][l], real) and (real <= mm).all()
        assert torch.equal(ae["depths"][l], prev + mm)
        prev = ae["depths"][l]


def test_qg_nogate_moved_is_any_coordinate_and_cost_surrogate_is_max():
    from dd.model import lm_loss
    torch.manual_seed(41)
    cfg = small(**QGC, qg="nogate")
    m = Model(cfg)
    randomize_qg(m, scale=0.4)
    idx = torch.randint(1, 50, (2, 32))
    _, aux = m(idx, collect=True)
    G, gs = 8, cfg.d // 8
    for l, mv in enumerate(aux["dd"]):
        x0 = aux["xs"][l].view(2, 32, G, gs)
        x1 = aux["xs"][l + 1].view(2, 32, G, gs)
        assert torch.equal(mv.detach(), (x0 != x1).any(-1).float())
        assert 0 < mv.mean() < 1
    with torch.no_grad():
        m.mu.fill_(1.0)
    loss, ce, cost, _, aux = lm_loss(m, idx)
    assert torch.allclose(cost, torch.stack([mv.mean() for mv in aux["dd"]]).mean(), atol=1e-6)
    (loss - ce).backward()
    assert m.blocks[0].w2.weight.grad.abs().sum() > 0


def test_qg_split_group_depends_only_on_its_head_and_ffn_block():
    torch.manual_seed(42)
    cfg = small(**QGC, qg="gate", qg_split=True)
    m = Model(cfg)
    randomize_qg(m)
    b = m.blocks[0]
    d, G, gs = cfg.d, 8, cfg.d // 8
    hb = cfg.ffn_hidden // G
    W2 = b.w2.weight * b.w2_mask
    Wo = b.o.weight * b.o_mask
    for g in range(G):
        rows = slice(g * gs, (g + 1) * gs)
        other_h = torch.ones(cfg.ffn_hidden, dtype=torch.bool)
        other_h[g * hb:(g + 1) * hb] = False
        assert W2[rows][:, other_h].abs().sum() == 0 and W2[rows][:, ~other_h].abs().sum() > 0
        other_d = torch.ones(d, dtype=torch.bool)
        other_d[g * b.hd:(g + 1) * b.hd] = False
        assert Wo[rows][:, other_d].abs().sum() == 0
    # gradient never reaches the masked entries' effect: masked product has zero grad outside blocks
    idx = torch.randint(1, 50, (2, 16))
    from dd.model import lm_loss
    loss, *_ = lm_loss(m, idx)
    loss.backward()
    assert (b.w2.weight.grad * (1 - b.w2_mask)).abs().sum() == 0
    assert (b.o.weight.grad * (1 - b.o_mask)).abs().sum() == 0


@pytest.mark.parametrize("parts", ["ffn", "head"])
def test_qg_split_parts_cut_only_the_named_part(parts):
    """Round 6: QG-splitFFN cuts only the FFN (W_O full), QG-splitHead only W_O (FFN shared)."""
    from dd.model import lm_loss
    torch.manual_seed(43)
    cfg = small(**QGC, qg="gate", qg_split=True, qg_split_parts=parts)
    assert cfg.split_ffn == (parts == "ffn") and cfg.split_head == (parts == "head")
    m = Model(cfg)
    randomize_qg(m)
    b = m.blocks[0]
    if parts == "ffn":
        assert b.o_mask is None and b.w2_mask is not None
    else:
        assert b.w2_mask is None and b.o_mask is not None
    idx = torch.randint(1, 50, (2, 16))
    loss, *_ = lm_loss(m, idx)
    loss.backward()
    mask = b.w2_mask if parts == "ffn" else b.o_mask
    w = b.w2.weight if parts == "ffn" else b.o.weight
    assert (w.grad * (1 - mask)).abs().sum() == 0 and (w.grad * mask).abs().sum() > 0
    full = b.o.weight if parts == "ffn" else b.w2.weight
    assert (full.grad != 0).float().mean() > 0.5, "the uncut part stays dense"
    # standing groups bit-identical holds for both
    m.eval()
    _, ae = m(idx, collect=True)
    for l, mm in enumerate(ae["dd"]):
        x0 = ae["xs"][l].view(2, 16, 8, -1)
        x1 = ae["xs"][l + 1].view(2, 16, 8, -1)
        assert torch.equal(x0[mm == 0], x1[mm == 0])


def test_qg_closed_gate_learns_and_cost_is_hard_share():
    from dd.model import lm_loss
    torch.manual_seed(43)
    m = Model(small(**QGC, qg="gate"))
    randomize_qg(m)
    with torch.no_grad():
        m.blocks[1].gate_head.bias.fill_(-50.0)
        m.blocks[1].gate_head.weight.mul_(0.0)
        m.blocks[1].gate_head.bias.fill_(-1.0)
    idx = torch.randint(1, 50, (2, 16))
    loss, ce, cost, _, aux = lm_loss(m, idx)
    assert aux["dd"][1].sum() == 0
    ce.backward()
    assert m.blocks[1].gate_head.bias.grad.abs().sum() > 0, "closed gates must still learn (STE)"
    hard = torch.stack([(d > 0).float().mean() for d in aux["dd"]]).mean()
    assert torch.allclose(cost, hard, atol=1e-6)


@pytest.mark.parametrize("variant", ["Q0", "QG"])
def test_per_state_grid_step_and_embedding_scale(variant):
    """Round 6, part 2: grid step per state (q_delta_layers) and embedding multiplier (emb_scale).
    The embedding is on grid 0; an open (QG) or every (Q0) group of block l lands on grid l + 1; a closed QG group
    keeps its previous value bit for bit (so it stays on the grid it was on); the tied output uses the raw embedding."""
    torch.manual_seed(44)
    steps = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
    kw = dict(quant=True, q_levels=8, q_delta_layers=steps, emb_scale=4.0)
    cfg = small(**kw, **(dict(q_cost="l0", qg="gate", depth_groups=8) if variant == "QG" else dict(q_cost="none")))
    assert cfg.qd(0) == 0.02 and cfg.qd(5) == 0.5
    m = Model(cfg)
    assert abs(float(m.emb.weight.std()) - 0.02 / 4.0) < 0.002, "embedding init: scaled embedding has std delta_0"
    if variant == "QG":
        randomize_qg(m)
    else:
        randomize_q(m, 3.0)
    idx = torch.randint(1, 50, (2, 32))
    m.eval()
    lg, ae = m(idx, collect=True)
    x0 = ae["xs"][0]
    assert torch.equal(x0, 0.02 * (m.emb(idx) * 4.0 / 0.02).round().clamp(-8, 8))
    for l in range(cfg.n_layers):
        D = steps[l + 1]
        xa, xb = ae["xs"][l], ae["xs"][l + 1]
        assert xb.abs().max() <= 8 * D + 1e-6 or variant == "QG"
        if variant == "Q0":
            assert torch.equal(xb, D * (xb / D).round())
        else:
            G, gs = 8, cfg.d // 8
            mm = ae["dd"][l]
            a, b = xa.view(2, 32, G, gs), xb.view(2, 32, G, gs)
            assert torch.equal(a[mm == 0], b[mm == 0]), "closed group changed"
            ob = b[mm == 1]
            assert torch.equal(ob, D * (ob / D).round()) and ob.abs().max() <= 8 * D + 1e-6
    ref = m.nf(ae["xs"][-1]) @ m.emb.weight.t()
    assert torch.allclose(lg, ref)


def test_r7_gate_controls_and_asplit():
    """Round 7: QG with a mask that is not learned.  random: share ~p, redrawn every pass, no gate head, closed
    groups bit-exact; static: the same k groups per block on every token, each group open in the same number of
    blocks; position: probability from the position in the story (since the last EOT).  A-split: plain residual
    with the QG-split cut (head h and FFN block g write only into group g), no grid, no gate."""
    torch.manual_seed(7)
    base = dict(quant=True, q_levels=8, q_delta=0.1, q_cost="l0", qg="gate", depth_groups=8)
    # random
    m = Model(small(**base, qg_mask="random", qg_mask_p=0.25))
    assert not any("gate_head" in n for n, _ in m.named_parameters())
    randomize_qg(m)
    idx = torch.randint(2, 50, (4, 32))
    _, a1 = m(idx, collect=True)
    _, a2 = m(idx, collect=True)
    d1 = torch.stack(a1["dd"])
    assert abs(float(d1.mean()) - 0.25) < 0.05 and not torch.equal(d1, torch.stack(a2["dd"]))
    for l in range(5):
        xa, xb = a1["xs"][l].view(4, 32, 8, 8), a1["xs"][l + 1].view(4, 32, 8, 8)
        assert torch.equal(xa[a1["dd"][l] == 0], xb[a1["dd"][l] == 0])
    # static: 2 of 8 groups per block, groups (2l, 2l+1) mod 8
    m = Model(small(**base, qg_mask="static", qg_mask_p=0.25))
    _, a = m(idx)
    opened = torch.zeros(8)
    for l, dd in enumerate(a["dd"]):
        row = dd[0, 0]
        assert torch.equal(dd, row.expand_as(dd)) and int(row.sum()) == 2
        assert row[(2 * l) % 8] == 1 and row[(2 * l + 1) % 8] == 1
        opened += row
    assert int(opened.sum()) == 10
    # position: p by story position, p0 before the first EOT
    cfg = small(**base, qg_mask="position", qg_pos_edges=(0, 1, 6), qg_pos_probs=(0.0, 1.0, 0.0), qg_pos_p0=1.0)
    m = Model(cfg)
    idx = torch.tensor([[5, 6, 7, 1, 8, 9, 10, 11, 12, 13, 14, 15]])
    p = m.pos_prob(idx)[0, :, 0]
    assert p.tolist() == [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0]
    _, a = m(idx)
    assert torch.equal(a["dd"][0][0, :, 0], p)
    # A-split: same masks as QG-split, no grid
    m = Model(small(depth_groups=8, qg_split=True))
    b = m.blocks[0]
    assert b.o_mask is not None and b.w2_mask is not None and not m.cfg.quant
    with torch.no_grad():
        for blk in m.blocks:
            blk.w2.weight.normal_()
            blk.o.weight.normal_()
    x = torch.randn(2, 12, 64)
    rope = m.rope(12, x.device)
    xg = x.clone()
    # output group g changes only through FFN block g / head g: zero those inputs, the group must not change
    z = b.n2(x + b.attn(b.n1(x), rope))
    h = F.silu(b.w1(z)) * b.w3(z)
    u = F.linear(h, b.w2.weight * b.w2_mask)
    h0 = h.clone()
    h0[..., :12] = 0                                   # FFN block 0 (hidden 96 / 8 = 12 units)
    u0 = F.linear(h0, b.w2.weight * b.w2_mask)
    assert torch.allclose(u[..., 8:], u0[..., 8:]) and not torch.allclose(u[..., :8], u0[..., :8])
    y, _, _, _ = b(xg, None, rope)
    assert torch.allclose(y, x + b.attn(b.n1(x), rope) + u, atol=1e-5)


def test_r7_q0split_cut():
    """Round 7: Q0-split = Q0 grid with the QG-split cut (masked W_O and w2), no gate: the FFN output of group g
    depends only on FFN block g, and the block output equals the quantised x + a + masked FFN."""
    torch.manual_seed(8)
    cfg = small(quant=True, q_levels=8, q_delta=0.1, q_cost="none", depth_groups=8, qg_split=True)
    m = Model(cfg)
    b = m.blocks[1]
    assert b.o_mask is not None and b.w2_mask is not None
    x = 0.1 * torch.randint(-8, 9, (2, 12, 64)).float()
    rope = m.rope(12, x.device)
    a = b.attn(b.n1(x), rope)
    z = b.n2(x + a)
    h = F.silu(b.w1(z)) * b.w3(z)
    u = F.linear(h, b.w2.weight * b.w2_mask)
    h0 = h.clone()
    h0[..., 12:24] = 0                                 # FFN block 1 -> output coordinates 8..15
    u0 = F.linear(h0, b.w2.weight * b.w2_mask)
    assert torch.allclose(u[..., :8], u0[..., :8]) and torch.allclose(u[..., 16:], u0[..., 16:])
    y, _, _, _ = b(x, torch.zeros(2, 12, 1), rope)
    assert torch.equal(y, 0.1 * torch.clamp(torch.round((x + (a + u).float()) / 0.1), -8, 8))


def test_r7_tokengate():
    """Round 7: QG-tokengate = learned QG gate whose input is only the current token's embedding: the open mask of
    every block is a function of the token id alone (same token -> same mask in any context and position), the
    closed groups stay bit-identical, the gate gets gradient and still drives the cost (mu targets the share)."""
    torch.manual_seed(9)
    base = dict(quant=True, q_levels=8, q_delta=0.1, q_cost="l0", qg="gate", depth_groups=8)
    m = Model(small(**base, qg_gate_in="token"))
    randomize_qg(m)
    assert all(hasattr(b, "ng") and hasattr(b, "gate_head") for b in m.blocks)
    idx = torch.randint(2, 50, (4, 32))
    idx[0, 3] = idx[1, 20] = idx[2, 7] = 17                        # token 17 in three contexts / positions
    _, a = m(idx, collect=True)
    for l, dd in enumerate(a["dd"]):
        sh = float(dd.mean())
        assert 0.05 < sh < 0.95, sh                                 # the mask is not trivial
        assert torch.equal(dd[0, 3], dd[1, 20]) and torch.equal(dd[0, 3], dd[2, 7])
        for t in range(2, 50):                                      # in general: mask depends on the token only
            sel = dd[idx == t]
            if sel.shape[0] > 1:
                assert torch.equal(sel, sel[:1].expand_as(sel))
        xa, xb = a["xs"][l].view(4, 32, 8, 8), a["xs"][l + 1].view(4, 32, 8, 8)
        assert torch.equal(xa[dd.detach() == 0], xb[dd.detach() == 0])
    # the default gate (QG) does depend on context: the same token gets different masks somewhere
    q = Model(small(**base))
    randomize_qg(q)
    _, aq = q(idx)
    assert any(not torch.equal(dd[0, 3], dd[1, 20]) or not torch.equal(dd[0, 3], dd[2, 7]) for dd in aq["dd"])
    # gradient reaches the gate and its norm through the cost
    from dd.model import lm_loss
    m.mu.fill_(1.0)
    loss, ce, cost, paid, _ = lm_loss(m, idx)
    loss.backward()
    assert m.blocks[2].gate_head.weight.grad.abs().sum() > 0 and m.blocks[2].ng.w.grad.abs().sum() > 0


def test_r8_staticinit_and_hybrid():
    """Round 8: QG-staticinit starts exactly from the static mask (groups 2l, 2l+1 open in block l, the rest closed)
    and the gate still learns (gradient on the bias); QG-hybrid keeps group 2l always open whatever the gate says,
    with no gate gradient on it, and the paid share counts it."""
    from dd.model import lm_loss
    torch.manual_seed(10)
    base = dict(quant=True, q_levels=8, q_delta=0.1, q_cost="l0", qg="gate", depth_groups=8)
    m = Model(small(**base, qg_static_init=True))
    idx = torch.randint(2, 50, (4, 32))
    _, a = m(idx)
    for l, dd in enumerate(a["dd"]):
        want = torch.zeros(8)
        want[[(2 * l) % 8, (2 * l + 1) % 8]] = 1
        assert torch.equal(dd.detach(), want.expand_as(dd)), l
    m.mu.fill_(1.0)
    loss = lm_loss(m, idx)[0]
    loss.backward()
    assert m.blocks[1].gate_head.bias.grad.abs().sum() > 0
    # hybrid: force group 2l open even when the gate wants everything closed
    h = Model(small(**base, qg_force_open=1))
    with torch.no_grad():
        for b in h.blocks:
            b.gate_head.bias.fill_(-50.0)
    _, a = h(idx)
    for l, dd in enumerate(a["dd"]):
        want = torch.zeros(8)
        want[(2 * l) % 8] = 1
        assert torch.equal(dd.detach(), want.expand_as(dd)), l
    h.mu.fill_(1.0)
    loss, ce, cost, paid, _ = lm_loss(h, idx)
    assert abs(float(cost) - 1 / 8) < 1e-6
    loss.backward()
    g = h.blocks[3].gate_head.bias.grad
    assert g[(2 * 3) % 8] == 0


def test_r8_ctrl_stream():
    """Round 8: QG-ctrl -- the control stream c (64 dims, c_0 = 0, c <- c + W_c zin in every block) reaches only the
    gates: with the gate's c-columns zeroed the model equals QG with the same weights; c gets gradient via the gates."""
    from dd.model import lm_loss
    torch.manual_seed(11)
    base = dict(quant=True, q_levels=8, q_delta=0.1, q_cost="l0", qg="gate", depth_groups=8)
    m = Model(small(**base, qg_ctrl=16))
    randomize_qg(m)
    q = Model(small(**base))
    sd = {k: v for k, v in m.state_dict().items() if not (".wc." in k or ".nc." in k)}
    for k in list(sd):
        if k.endswith("gate_head.weight"):
            sd[k] = sd[k][:, :64].clone()
    q.load_state_dict(sd)
    idx = torch.randint(2, 50, (3, 24))
    with torch.no_grad():
        for b in m.blocks:
            b.gate_head.weight[:, 64:] = 0
    lm_, am = m(idx)
    lq, aq = q(idx)
    assert torch.equal(lm_, lq) and all(torch.equal(a, b) for a, b in zip(am["dd"], aq["dd"]))
    with torch.no_grad():                                    # now let c matter: the gates change, the run still works
        for b in m.blocks:
            b.gate_head.weight[:, 64:].normal_(0, 3.0)
            b.wc.weight.normal_(0, 1.0)
    _, am2 = m(idx)
    assert any(not torch.equal(a, b) for a, b in zip(am2["dd"], aq["dd"]))
    m.mu.fill_(1.0)
    loss = lm_loss(m, idx)[0]
    loss.backward()
    assert m.blocks[0].wc.weight.grad.abs().sum() > 0 and m.blocks[3].nc.w.grad.abs().sum() > 0


def test_r9_update_rounding():
    """Round 9, updround: only the update is rounded -- x_new = clamp(x + delta*round((a+u)/delta), +-K delta); x is not
    re-rounded (an off-grid x keeps its fine part), closed groups stay bit-identical, Q0 and QG both run."""
    from dd.model import quantize_update
    x = torch.tensor([0.33, -0.07, 3.95, -3.9])
    upd = torch.tensor([0.26, 0.04, 0.3, -0.5])
    xn, step = quantize_update(x, upd, 0.1, 40)
    assert torch.allclose(xn, torch.tensor([0.63, -0.07, 4.0, -4.0])) and step.tolist() == [3.0, 0.0, 3.0, -5.0]
    torch.manual_seed(12)
    for kw in (dict(q_cost="none"), dict(q_cost="l0", qg="gate", depth_groups=8)):
        m = Model(small(quant=True, q_levels=8, q_delta=0.1, q_upd_round=True, **kw))
        if kw.get("qg"):
            randomize_qg(m)
        idx = torch.randint(2, 50, (2, 16))
        lg, a = m(idx, collect=True)
        assert torch.isfinite(lg).all()
        if kw.get("qg"):
            for l in range(4):
                xa, xb = a["xs"][l].view(2, 16, 8, 8), a["xs"][l + 1].view(2, 16, 8, 8)
                closed = a["dd"][l].detach() == 0
                assert torch.equal(xa[closed], xb[closed])


def test_r9_smooth_gate():
    """Round 9, H-Net-style gate: m = 1[p > 0.5] with no gradient in the write path, an open group writes
    Q(x + s(p) (a + u)) -- hnet s = p, dimmer s = 2p - 1; closed groups bit-identical; CE reaches the gate head only
    through s (open groups), the cost through the sigmoid surrogate."""
    from dd.model import lm_loss
    torch.manual_seed(13)
    base = dict(quant=True, q_levels=8, q_delta=0.1, q_cost="l0", qg="gate", depth_groups=8)
    idx = torch.randint(2, 50, (2, 16))
    for kind in ("hnet", "dimmer"):
        m = Model(small(**base, qg_smooth=kind))
        randomize_qg(m)
        lg, a = m(idx, collect=True)
        for l in range(5):
            dd = a["dd"][l].detach()
            assert set(dd.unique().tolist()) <= {0.0, 1.0}
            xa, xb = a["xs"][l].view(2, 16, 8, 8), a["xs"][l + 1].view(2, 16, 8, 8)
            assert torch.equal(xa[dd == 0], xb[dd == 0])
        # CE only (mu = 0): the gate head still gets gradient, through s
        ce = lm_loss(m, idx)[1]
        ce.backward()
        assert m.blocks[2].gate_head.weight.grad.abs().sum() > 0
    # with s forced to 1 the write equals plain QG with the same decisions
    m = Model(small(**base, qg_smooth="hnet"))
    randomize_qg(m)
    q = Model(small(**base))
    q.load_state_dict(m.state_dict())
    one = {"smooth_s_fn": lambda blk, sg: torch.ones_like(sg)}
    l1 = m(idx, ctl=one)[0]
    l2 = q(idx)[0]
    assert torch.allclose(l1, l2)


def test_r9_smooth_gate_fix():
    """Round 9 fix: closed groups write nothing forward (bit-identical) but their gate gets the QG25 straight-through
    gradient (as if the full update were written); open groups get the gradient through s(p).  Forward equals the
    unfixed variant exactly; the gate gradient of a closed group equals QG25's for the same weights."""
    from dd.model import lm_loss
    torch.manual_seed(14)
    base = dict(quant=True, q_levels=8, q_delta=0.1, q_cost="l0", qg="gate", depth_groups=8)   # mu = 0: CE only
    idx = torch.randint(2, 50, (2, 16))
    a = Model(small(**base, qg_smooth="hnet"))
    randomize_qg(a)
    b = Model(small(**base, qg_smooth="hnet", qg_smooth_closed_ste=True))
    b.load_state_dict(a.state_dict())
    la, aa = a(idx, collect=True)
    lb, ab = b(idx, collect=True)
    assert torch.equal(la, lb)
    for l in range(5):
        dd = ab["dd"][l].detach()
        xa, xb = ab["xs"][l].view(2, 16, 8, 8), ab["xs"][l + 1].view(2, 16, 8, 8)
        assert torch.equal(xa[dd == 0], xb[dd == 0])
    # last block: a closed group gets a gate gradient now, and it equals the QG25 one (nothing downstream differs)
    q = Model(small(**base))
    q.load_state_dict(a.state_dict())
    lm_loss(b, idx)[1].backward()
    lm_loss(q, idx)[1].backward()
    gb, gq = b.blocks[4].gate_head.weight.grad, q.blocks[4].gate_head.weight.grad
    _, aq = q(idx)
    closed = (aq["dd"][4].detach() == 0).all(0).all(0)                               # groups closed at every position
    assert gb.abs().sum() > 0
    if closed.any():
        assert torch.allclose(gb[closed], gq[closed], atol=1e-6)


def test_r10_nogrid():
    """Round 10, nogrid: the embedding and an open group's write x + a + u are neither rounded nor clipped; a closed
    group is bit-identical; QG and static both run; the gate still gets a gradient."""
    from dd.model import lm_loss
    torch.manual_seed(15)
    base = dict(quant=True, q_levels=8, q_delta=0.1, q_cost="l0", qg="gate", depth_groups=8, q_nogrid=True)
    idx = torch.randint(2, 50, (2, 16))
    for kw in (dict(), dict(qg_mask="static")):
        m = Model(small(**base, **kw))
        randomize_qg(m) if not kw else randomize_q(m, 3.0)
        ups = {}

        def fn(blk, upd, x):
            ups[blk.l] = upd.detach().clone()
            return upd
        lg, a = m(idx, collect=True, ctl={"q_upd_fn": fn})
        assert torch.equal(a["xs"][0], m.emb(idx).float())                     # embedding not rounded
        off_grid = beyond = False
        for l in range(5):
            x0, x1 = a["xs"][l].detach(), a["xs"][l + 1].detach()
            mf = a["dd"][l].detach().repeat_interleave(8, -1)
            assert torch.equal(x1[mf == 0], x0[mf == 0])                         # closed: not written
            assert torch.equal(x1[mf == 1], (x0 + ups[l])[mf == 1])              # open: x + a + u exactly
            r = x1 / 0.1
            off_grid |= bool((r - r.round()).abs().max() > 1e-3)
            beyond |= bool(x1.abs().max() > 8 * 0.1)                             # no clip at +-K delta
        assert off_grid and beyond
        assert 0 < float(torch.stack(a["dd"]).detach().mean()) < 1
    m.mu.fill_(1.0)
    q = Model(small(**base))
    randomize_qg(q)
    q.mu.fill_(1.0)
    lm_loss(q, idx)[0].backward()
    assert all(float(b.gate_head.weight.grad.abs().sum()) > 0 for b in q.blocks)


def test_r10_ordered_gate():
    """Round 10, QG-ordered: one gate logit per token and block; the open groups are always the first k of the queue
    (2l, 2l+1, ...) mod G; everything is open at init; with z in (1, 2) everywhere the mask is exactly static and the
    forward equals QGstatic with the same weights; the gate head gets a gradient; the paid share is k / G."""
    from dd.model import lm_loss
    torch.manual_seed(16)
    base = dict(quant=True, q_levels=8, q_delta=0.1, q_cost="l0", qg="gate", depth_groups=8)
    m = Model(small(**base, qg_ordered=True))
    assert m.blocks[0].gate_head.weight.shape == (1, 64)
    idx = torch.randint(2, 50, (4, 32))
    _, a = m(idx)
    assert all(bool((dd.detach() == 1).all()) for dd in a["dd"])                 # bias G - 1 + 2: all open
    randomize_qg(m)
    with torch.no_grad():
        for b in m.blocks:
            b.gate_head.bias.fill_(3.0)
    zs = {}

    def hook(blk, mm, zg):
        zs[blk.l] = zg.detach()
        return mm
    _, a = m(idx, ctl={"hc_gate_fn": hook})
    ks = set()
    ar = torch.arange(8)
    for l in range(5):
        qu = m.cfg.static_groups(l, 8)
        assert qu[:2] == [(2 * l) % 8, (2 * l + 1) % 8] and sorted(qu) == list(range(8))
        dd = a["dd"][l].detach()[..., qu]                                         # queue order
        k = dd.sum(-1)
        assert torch.equal(dd, (ar < k[..., None]).float())                       # always a prefix of the queue
        assert torch.allclose(zs[l][..., qu] - zs[l][..., qu[:1]], -ar.float().expand_as(dd))
        ks |= set(k.unique().tolist())
    assert len(ks) > 2
    m.mu.fill_(1.0)
    loss, ce, cost, paid, aux = lm_loss(m, idx)
    assert abs(float(cost) - float(torch.stack([dd.detach().mean() for dd in a["dd"]]).mean())) < 1e-6
    loss.backward()
    assert all(float(b.gate_head.weight.grad.abs().sum()) > 0 for b in m.blocks)
    s = Model(small(**base, qg_mask="static"))
    s.load_state_dict({k_: v for k_, v in m.state_dict().items() if "gate_head" not in k_})
    with torch.no_grad():
        for b in m.blocks:
            b.gate_head.weight.zero_()
            b.gate_head.bias.fill_(1.5)
    lo, ao = m(idx)
    ls, as_ = s(idx)
    assert torch.equal(lo, ls)
    assert all(torch.equal(x.detach(), y) for x, y in zip(ao["dd"], as_["dd"]))


def test_r11_ordered_perblock_mu():
    """Round 11, QG-ordered-perblock: cost = sum_l mu_l paid_l / L (equal mu -> the shared-mu cost of QG-ordered);
    the dual step moves every mu_l by eta * (paid_l - target); d cost / d paid_l = mu_l / L."""
    from dd.model import lm_loss
    from dd.ts_train import mu_step
    torch.manual_seed(17)
    base = dict(quant=True, q_levels=8, q_delta=0.1, q_cost="l0", qg="gate", depth_groups=8, qg_ordered=True)
    m = Model(small(**base, qg_mu_per_block=True))
    s = Model(small(**base))
    s.load_state_dict(m.state_dict())
    randomize_qg(m)
    s.load_state_dict(m.state_dict())
    idx = torch.randint(2, 50, (4, 32))
    m.mu.fill_(0.7)
    s.mu.fill_(0.7)
    lm, cm, *_ = lm_loss(m, idx)
    ls, cs, *_ = lm_loss(s, idx)
    assert torch.allclose(lm, ls, atol=1e-6) and torch.equal(cm, cs)
    mu = torch.tensor([0.0, 1.0, 2.0, 0.5, 3.0])
    m.mu.copy_(mu)
    loss, ce, cost, paid, aux = lm_loss(m, idx)
    pl = torch.stack([p.detach() for p in paid])
    assert abs(float(loss - ce) - float((mu * pl).mean())) < 1e-6
    # d cost / d paid_l = mu_l / L (block 0 with mu 0 pays nothing for its own share; its gate can still get a
    # gradient through the state of later blocks)
    g = torch.autograd.grad(loss - ce, paid, retain_graph=True)
    assert torch.allclose(torch.stack(g), mu / 5)
    loss.backward()
    assert all(float(b.gate_head.weight.grad.abs().sum()) > 0 for b in m.blocks)
    m.mu.copy_(mu)
    mu_step(m, paid, 0.25, 0.05)
    assert torch.allclose(m.mu, (mu + 0.05 * (pl - 0.25)).clamp_min(0))
    s.mu.fill_(0.7)
    mu_step(s, paid, 0.25, 0.05)                                                     # shared mu: one step for all
    assert torch.allclose(s.mu, torch.full((5,), 0.7) + 0.05 * (pl.mean() - 0.25))
    m.mu.fill_(0.01)
    mu_step(m, [torch.tensor(0.0)] * 5, 0.25, 0.05, step_max=0.002)
    assert torch.allclose(m.mu, torch.full((5,), 0.008))                            # capped step


def test_r11_randk_and_unequal_static():
    """Round 11 controls without a gate.  static-randk: in block l the first k groups of the queue (2l, 2l+1, ...)
    mod G are open, k drawn per (token, block) from the pmf, redrawn every forward pass, closed groups not written;
    static-unequal: block l opens exactly the first k_l groups of its queue on every token."""
    from dd.model import lm_loss
    torch.manual_seed(18)
    pk = (0.191088, 0.215270, 0.270210, 0.162760, 0.092127, 0.039750, 0.016540, 0.008515, 0.003742)
    pk = tuple(v / sum(pk) for v in pk)
    base = dict(quant=True, q_levels=8, q_delta=0.1, q_cost="l0", qg="gate", depth_groups=8)
    m = Model(small(**base, qg_mask="randk", qg_randk_probs=pk))
    assert not hasattr(m.blocks[0], "gate_head")
    randomize_q(m)
    idx = torch.randint(2, 50, (64, 32))
    _, a = m(idx, collect=True)
    ar = torch.arange(8)
    ks = []
    for l in range(5):
        qu = m.cfg.static_groups(l, 8)
        dd = a["dd"][l][..., qu]
        k = dd.sum(-1)
        assert torch.equal(dd, (ar < k[..., None]).float())                          # prefix of the queue
        ks.append(k.flatten())
        x0 = a["xs"][l].view(64, 32, 8, 8)
        x1 = a["xs"][l + 1].view(64, 32, 8, 8)
        cl = a["dd"][l] == 0
        assert torch.equal(x0[cl], x1[cl])                                            # closed groups bit-identical
    kk = torch.cat(ks)
    freq = torch.bincount(kk.long(), minlength=9).float() / kk.numel()
    assert (freq - torch.tensor(pk)).abs().max() < 0.02                              # 10240 draws
    assert abs(float(kk.mean()) - 2.0) < 0.05
    assert float((ks[0] == ks[1]).float().mean()) < 0.5                               # independent across blocks
    _, a2 = m(idx)
    assert not torch.equal(a["dd"][0], a2["dd"][0])                                   # redrawn every forward
    loss, ce, cost, paid, aux = lm_loss(m, idx)
    assert float(m.mu.abs().sum()) == 0 and abs(float(cost) - 0.25) < 0.02
    kl = (2, 2, 3, 1, 2)
    u = Model(small(**base, qg_mask="static", qg_static_k=kl))
    randomize_q(u)
    _, a = u(idx)
    for l in range(5):
        want = torch.zeros(8)
        want[[(2 * l + j) % 8 for j in range(kl[l])]] = 1
        assert torch.equal(a["dd"][l], want.expand(64, 32, 8))
    with pytest.raises(AssertionError):
        Model(small(**base, qg_mask="static", qg_static_k=(2, 2)))


def test_r12_depth_lookup_recompute():
    """Round 12: recomputing the depth-embedding lookups in the backward pass changes memory only -- the same loss
    bit for bit and the same gradients (up to summation order)."""
    from dd.model import lm_loss
    torch.manual_seed(19)
    base = dict(quant=True, q_cost="l0", qg="gate", depth_groups=8, q_nogrid=True, q_delta_layers=(0.3,) + (1.0,) * 5,
                emb_scale=8.0)
    a = Model(small(**base))
    b = Model(small(**base, depth_lookup_recompute=True))
    randomize_qg(a)
    b.load_state_dict(a.state_dict())
    idx = torch.randint(2, 50, (4, 32))
    a.mu.fill_(0.5)
    b.mu.fill_(0.5)
    la, lb = lm_loss(a, idx)[0], lm_loss(b, idx)[0]
    assert torch.equal(la, lb)
    la.backward()
    lb.backward()
    ga = dict(a.named_parameters())
    for n, p in b.named_parameters():
        if p.grad is not None:
            assert torch.allclose(ga[n].grad, p.grad, atol=1e-8, rtol=1e-6), n
    assert float(b.blocks[2].dt_f.grad.abs().sum()) > 0 and float(b.blocks[2].gate_head.weight.grad.abs().sum()) > 0


def test_r13_ordered_k_range():
    """Round 13, QG-ordered with k in [1, 3]: the first group of the queue is always open (no gate gradient through
    it), groups at queue positions >= 3 are never open, positions 1-2 follow z > j; at init k = 3 everywhere
    (bias kmax - 1 + 2); with k_range (0, G) the model equals plain QG-ordered bit for bit."""
    from dd.model import lm_loss
    torch.manual_seed(20)
    base = dict(quant=True, q_levels=8, q_delta=0.1, q_cost="l0", qg="gate", depth_groups=8, qg_ordered=True,
                qg_mu_per_block=True)
    m = Model(small(**base, qg_k_range=(1, 3)))
    assert float(m.blocks[0].gate_head.bias) == 4.0
    idx = torch.randint(2, 50, (4, 32))
    _, a = m(idx)
    for l in range(5):
        qu = m.cfg.static_groups(l, 8)
        assert torch.equal(a["dd"][l].detach()[..., qu], torch.tensor([1., 1, 1, 0, 0, 0, 0, 0]).expand(4, 32, 8))
    randomize_qg(m, gate_scale=3.0)
    _, a = m(idx)
    ks = set()
    ar = torch.arange(8)
    for l in range(5):
        qu = m.cfg.static_groups(l, 8)
        dd = a["dd"][l].detach()[..., qu]
        k = dd.sum(-1)
        assert torch.equal(dd, (ar < k[..., None]).float()) and int(k.min()) >= 1 and int(k.max()) <= 3
        ks |= set(k.unique().tolist())
    assert ks == {1.0, 2.0, 3.0}
    m.mu.fill_(1.0)
    loss, ce, cost, paid, aux = lm_loss(m, idx)
    loss.backward()
    assert all(float(b.gate_head.weight.grad.abs().sum()) > 0 for b in m.blocks)
    # forced groups carry no gradient to the gate: d dd / d z only through queue positions 1, 2
    m.zero_grad()
    _, a = m(idx)
    qu0 = m.cfg.static_groups(0, 8)
    a["dd"][0][..., [qu0[0]] + qu0[3:]].sum().backward(retain_graph=True)
    assert m.blocks[0].gate_head.bias.grad is None or float(m.blocks[0].gate_head.bias.grad.abs().sum()) == 0
    full = Model(small(**base, qg_k_range=(0, 8)))
    plain = Model(small(**base))
    plain.load_state_dict(full.state_dict())
    randomize_qg(full)
    plain.load_state_dict(full.state_dict())
    assert torch.equal(full(idx)[0], plain(idx)[0])


def test_r14_vq_shared_and_delta():
    """Round 14, VQ state on top of A.  shared: the state after the embedding and after every block is, per group,
    exactly a vector of the group's codebook; delta: every block's change per group is exactly a vector of that
    block's codebook, vector 0 is the zero update and is chosen when the block writes nothing.  Straight-through
    gradients reach the blocks and the embedding; loss = CE + beta * commitment; the EMA update initialises the
    codebooks from the batch, moves them towards the assigned means and restarts unused vectors (delta: 0 stays 0)."""
    from dd.model import lm_loss, vq_update
    torch.manual_seed(21)
    idx = torch.randint(2, 50, (4, 32))
    m = Model(small(budget=False, depth_groups=8, vq="shared", vq_k=16))
    assert m.vq_E.shape == (1, 8, 16, 8)
    _, a = m(idx)
    vq_update(m, a)                                              # data init: rows are batch vectors
    Z = a["vq_z"][3].reshape(-1, 8, 8)
    assert bool(m.vq_ready) and any(bool((Z[:, 2] == m.vq_E[0, 2, k]).all(-1).any()) for k in range(16))
    logits, a = m(idx, collect=True)
    assert len(a["vq_idx"]) == 6                                  # embedding + 5 blocks
    for i, x in enumerate(a["xs"]):
        xg = x.detach().view(4, 32, 8, 8)
        q = m.vq_E[0][torch.arange(8)[None, None, :], a["vq_idx"][i]]
        assert torch.equal(xg, q)                                 # exactly a codebook vector
    loss, ce, cost, _, aux = lm_loss(m, idx)
    zq = [((z - m.vq_E[0][torch.arange(8)[None, None, :], i]) ** 2).mean() for z, i in zip(aux["vq_z"], aux["vq_idx"])]
    assert torch.allclose(cost, torch.stack(zq).mean()) and torch.allclose(loss, ce + 0.25 * cost)
    loss.backward()
    assert float(m.blocks[0].w1.weight.grad.abs().sum()) > 0 and float(m.emb.weight.grad.abs().sum()) > 0
    E0, N0 = m.vq_E.clone(), m.vq_N.clone()
    with torch.no_grad():
        m.vq_N[0, 1, 5] = 0.1                                     # force one dead vector
    st = vq_update(m, aux)
    assert st[0][1] >= 1 and not torch.equal(m.vq_E, E0)
    Zall = torch.cat([z.reshape(-1, 8, 8) for z in aux["vq_z"]])
    assert bool((Zall[:, 1] == m.vq_E[0, 1, 5]).all(-1).any())   # restarted from a batch vector

    d = Model(small(budget=False, depth_groups=8, vq="delta", vq_k=8))
    assert d.vq_E.shape == (5, 8, 8, 8) and float(d.vq_E[:, :, 0].abs().sum()) == 0
    _, a = d(idx)
    vq_update(d, a)
    _, a = d(idx, collect=True)
    for l in range(5):
        dx = (a["xs"][l + 1] - a["xs"][l]).detach().view(4, 32, 8, 8)
        q = d.vq_E[l][torch.arange(8)[None, None, :], a["vq_idx"][l]]
        assert torch.allclose(dx, q, atol=1e-6)
    with torch.no_grad():
        d.vq_N[:, :, 3] = 0.1
    vq_update(d, a)
    assert float(d.vq_E[:, :, 0].abs().sum()) == 0 and float(d.vq_M[:, :, 0].abs().sum()) == 0
    with torch.no_grad():                                         # a block that writes nothing -> the zero update
        for b in d.blocks:
            b.o.weight.zero_()
            b.w2.weight.zero_()
    _, a = d(idx, collect=True)
    assert all(int(i.max()) == 0 for i in a["vq_idx"]) and torch.equal(a["xs"][0], a["xs"][-1])


def test_r15_vq_norm_and_init():
    """Round 15, normalised VQ: codebook vectors stay unit length (also after EMA and restarts); every quantised group
    equals |z| e_k with e_k the vector of largest cosine to z; the length gets an exact gradient, the direction a
    straight-through one; commitment = beta * mean |u - e|^2 (scale-free); delta has no zero vector.  Loading an A
    checkpoint into a VQ model only leaves the codebook buffers missing."""
    from dd.model import lm_loss, vq_update
    torch.manual_seed(22)
    idx = torch.randint(2, 50, (4, 32))
    for kind in ("shared", "delta"):
        m = Model(small(budget=False, depth_groups=8, vq=kind, vq_k=16, vq_norm=True, vq_beta=0.05))
        assert torch.allclose(m.vq_E.norm(dim=-1), torch.ones(m.vq_E.shape[:-1]))
        _, a = m(idx)
        vq_update(m, a)
        with torch.no_grad():
            m.vq_N[..., 3] = 0.1
        _, a = m(idx)
        vq_update(m, a)
        assert torch.allclose(m.vq_E.norm(dim=-1), torch.ones(m.vq_E.shape[:-1]), atol=1e-5)
        _, a = m(idx, collect=True)
        for i, (b, ix, r) in enumerate(zip(a["vq_book"], a["vq_idx"], a["vq_r"])):
            x_after = a["xs"][i if kind == "shared" else i + 1].detach().view(4, 32, 8, 8)
            val = x_after if kind == "shared" else x_after - a["xs"][i].detach().view(4, 32, 8, 8)
            e = m.vq_E[b][torch.arange(8)[None, None, :], ix]
            assert torch.allclose(val, r[..., None] * e, atol=1e-5)
            cos = torch.einsum("btgs,gks->btgk", a["vq_z"][i], m.vq_E[b])
            assert torch.equal(ix, cos.argmax(-1))
        loss, ce, cost, _, aux = lm_loss(m, idx)
        assert float(cost) <= 4.0 and torch.allclose(loss, ce + 0.05 * cost)
        loss.backward()
        assert float(m.blocks[0].w1.weight.grad.abs().sum()) > 0 and float(m.emb.weight.grad.abs().sum()) > 0
    a_model = Model(small(budget=False))
    v = Model(small(budget=False, depth_groups=8, vq="shared", vq_k=16, vq_norm=True))
    missing, unexpected = v.load_state_dict(a_model.state_dict(), strict=False)
    assert not unexpected and sorted(missing) == ["vq_E", "vq_M", "vq_N", "vq_ready"]


def test_r16_attractor():
    """Round 16, books-shared-attractor: after every block x_g = s_l * C_g softmax(beta C_g^T normalize(x_g)) with a
    unit-norm book shared by all blocks; a large beta approaches the hard arg max; att_hard gives exactly s_l * C[k*];
    the book and the blocks get gradients; with attract_off the model is plain A (same logits as A's weights)."""
    torch.manual_seed(23)
    idx = torch.randint(2, 50, (4, 32))
    sc = (2.0, 3.0, 4.0, 5.0, 6.0)
    m = Model(small(budget=False, depth_groups=8, attract=True, vq_k=16, attract_scale=sc))
    a_model = Model(small(budget=False))
    missing, unexpected = m.load_state_dict(a_model.state_dict(), strict=False)
    assert not unexpected and sorted(missing) == ["att_C", "att_beta"]
    assert torch.equal(m(idx, ctl={"attract_off": True})[0], a_model(idx)[0])
    m.att_beta.fill_(3.0)
    _, a = m(idx, collect=True)
    Cn = F.normalize(m.att_C, dim=-1).detach()
    for l in range(5):
        xh = a["att_xh"][l]
        w = torch.softmax(3.0 * torch.einsum("btgs,gks->btgk", xh, Cn), -1)
        want = sc[l] * torch.einsum("btgk,gks->btgs", w, Cn)
        assert torch.allclose(a["xs"][l + 1].detach().view(4, 32, 8, 8), want, atol=1e-5)
        assert torch.equal(a["att_idx"][l], torch.einsum("btgs,gks->btgk", xh, Cn).argmax(-1))
    logits, _ = m(idx)
    F.cross_entropy(logits[:, :-1].flatten(0, 1), idx[:, 1:].flatten()).backward()
    assert float(m.att_C.grad.abs().sum()) > 0 and float(m.blocks[0].w1.weight.grad.abs().sum()) > 0
    m.att_hard = True
    with torch.no_grad():
        _, ah = m(idx, collect=True)
    for l in range(5):
        hard = sc[l] * Cn[torch.arange(8)[None, None, :], ah["att_idx"][l]]
        assert torch.equal(ah["xs"][l + 1].view(4, 32, 8, 8), hard)
    m.att_hard = False
    m.att_beta.fill_(1e4)
    with torch.no_grad():
        ls, _ = m(idx)
        m.att_hard = True
        lh, _ = m(idx)
    assert torch.allclose(ls, lh, atol=1e-3)


def test_r16_init_from_checkpoint(tmp_path):
    """ts_train.init_from_checkpoint (the code path of the fine-tuning runs): an A checkpoint loads into VQ and
    attractor models, only the new buffers / book are missing, and the books are filled from the base activations
    (attractor: unit directions of A's states; VQ: batch vectors); a plain A loads with nothing missing."""
    from dd.ts_train import init_from_checkpoint
    torch.manual_seed(24)
    base = Model(small(budget=False))
    path = tmp_path / "a.pt"
    torch.save({"cfg": base.cfg.to_dict(), "state": base.state_dict()}, path)
    train = torch.randint(2, 50, (4000,), dtype=torch.int16)
    att = Model(small(budget=False, depth_groups=8, attract=True, vq_k=16, attract_scale=(1.0,) * 5))
    assert sorted(init_from_checkpoint(att, str(path), train, 31, 4, "cpu")) == ["att_C", "att_beta"]
    assert torch.allclose(att.att_C.norm(dim=-1), torch.ones(8, 16), atol=1e-5)
    assert torch.equal(att(train[:32].long()[None], ctl={"attract_off": True})[0], base(train[:32].long()[None])[0])
    for kind in ("shared", "delta"):
        v = Model(small(budget=False, depth_groups=8, vq=kind, vq_k=16, vq_norm=True))
        assert sorted(init_from_checkpoint(v, str(path), train, 31, 4, "cpu")) == ["vq_E", "vq_M", "vq_N", "vq_ready"]
        assert bool(v.vq_ready)
    a2 = Model(small(budget=False))
    assert init_from_checkpoint(a2, str(path), train, 31, 4, "cpu") == []


def test_r17_kmeans_and_gradual_insertion(tmp_path):
    """Round 17: spherical k-means finds well-separated directions; vq_off gives the base model; with vq_mix the
    output is (1 - g) z + g snap(z) -- g = 0 is the base model, g = 1 pure normalised VQ; init_from_checkpoint with
    k-means fills unit codebooks (shared: all depths pooled; delta: per block) and sets vq_ready."""
    from dd.model import spherical_kmeans
    from dd.ts_train import init_from_checkpoint
    torch.manual_seed(25)
    cent = F.normalize(torch.randn(4, 8), dim=-1)
    X = F.normalize(cent.repeat_interleave(200, 0) + 0.05 * torch.randn(800, 8), dim=-1)
    C, cnt = spherical_kmeans(X, 4, iters=30)
    assert torch.allclose(C.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert sorted(cnt.tolist()) == [200.0] * 4 and float((C @ cent.t()).max(0).values.min()) > 0.99
    base = Model(small(budget=False))
    path = tmp_path / "a.pt"
    torch.save({"cfg": base.cfg.to_dict(), "state": base.state_dict()}, path)
    train = torch.randint(2, 50, (6000,), dtype=torch.int16)
    idx = train[:32].long()[None]
    for kind in ("shared", "delta"):
        m = Model(small(budget=False, depth_groups=8, vq=kind, vq_k=16, vq_norm=True, vq_mix=True))
        init_from_checkpoint(m, str(path), train, 31, 4, "cpu", kmeans_windows=20)
        assert bool(m.vq_ready) and torch.allclose(m.vq_E.norm(dim=-1), torch.ones(m.vq_E.shape[:-1]), atol=1e-5)
        lb = base(idx)[0]
        assert torch.allclose(m(idx, ctl={"vq_off": True})[0], lb, atol=1e-5)
        m.vq_gamma.fill_(0.0)
        assert torch.allclose(m(idx)[0], lb, atol=1e-5)
        m.vq_gamma.fill_(1.0)
        pure = Model(small(budget=False, depth_groups=8, vq=kind, vq_k=16, vq_norm=True))
        pure.load_state_dict(m.state_dict())
        assert torch.equal(m(idx)[0], pure(idx)[0])
        m.vq_gamma.fill_(0.5)
        _, a = m(idx, collect=True)
        z0 = a["vq_z"][0] * a["vq_r"][0][..., None]                       # first quantisation point: z = r u
        e = m.vq_E[0][torch.arange(8)[None, None, :], a["vq_idx"][0]]
        want = 0.5 * z0 + 0.5 * a["vq_r"][0][..., None] * e
        got = a["xs"][0 if kind == "shared" else 1].detach().view(1, 32, 8, 8)
        if kind == "delta":
            got = got - a["xs"][0].detach().view(1, 32, 8, 8)
        assert torch.allclose(got, want, atol=1e-5)


def test_r18_funnel_layouts_and_mat():
    """Round 18 (funnel): block l writes exactly its groups -- every other coordinate is bit-identical after the block;
    narrow projections o: (w_l x d), w2: (w_l x ffn); funnel-narrow: groups 0..k are final after block k; static-36:
    36 writes, every group 4-5 times; mat: loss = CE + 0.2 CE(prefix head), the prefix head equals RMSNorm over the
    first k groups times the matching embedding columns, and its weights get a gradient."""
    from dd.model import lm_loss
    from dd.train import FUNNEL_NARROW, FUNNEL_WIDE, static36
    st = static36()
    assert sum(len(g) for g in st) == 36 and sorted(sum((list(g) for g in st), [])).count(0) in (4, 5)
    assert all(sum(g == x for gr in st for g in gr) in (4, 5) for x in range(8))
    assert sum(len(g) for g in FUNNEL_NARROW) == 36 and sum(len(g) for g in FUNNEL_WIDE) == 36
    torch.manual_seed(26)
    idx = torch.randint(2, 50, (2, 32))
    for layout in (FUNNEL_NARROW, FUNNEL_WIDE, st):
        cfg = Cfg(vocab=50, d=64, n_layers=8, n_heads=8, ffn_hidden=96, max_len=32, budget=False, depth_groups=8,
                  write_groups=layout)
        m = Model(cfg)
        for l, b in enumerate(m.blocks):
            assert b.o.weight.shape == (8 * len(layout[l]), 64) and b.w2.weight.shape == (8 * len(layout[l]), 96)
        _, a = m(idx, collect=True)
        for l in range(8):
            keep = torch.ones(64, dtype=torch.bool)
            for g in layout[l]:
                keep[8 * g:8 * (g + 1)] = False
            assert torch.equal(a["xs"][l][..., keep], a["xs"][l + 1][..., keep])
            assert not torch.equal(a["xs"][l][..., ~keep], a["xs"][l + 1][..., ~keep])
        if layout is FUNNEL_NARROW:
            for k in range(8):
                assert torch.equal(a["xs"][k + 1][..., :8 * (k + 1)], a["xs"][8][..., :8 * (k + 1)])
    cfg = Cfg(vocab=50, d=64, n_layers=8, n_heads=8, ffn_hidden=96, max_len=32, budget=False, depth_groups=8,
              write_groups=FUNNEL_NARROW, mat_weight=0.2)
    m = Model(cfg)
    with torch.no_grad():
        m.mat_w.copy_(torch.rand(8, 64) + 0.5)
    _, a = m(idx)
    x = a["x_final"]
    for k in (1, 3, 8):
        n = 8 * k
        xs = x[..., :n].float()
        want = (xs * torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + 1e-6) * m.mat_w[k - 1, :n]) @ m.emb.weight[:, :n].t()
        assert torch.allclose(m.prefix_logits(x, k), want, atol=1e-5)
    loss, ce, ce_k, _, _ = lm_loss(m, idx)
    assert torch.allclose(loss, ce + 0.2 * ce_k) and float(ce_k) > 0
    loss.backward()
    assert float(m.mat_w.grad.abs().sum()) > 0 and float(m.blocks[3].o.weight.grad.abs().sum()) > 0


def test_r20_mat_strict_head():
    """Round 20, mat-strict: no separate head parameters; the prefix head is RMSNorm over the first k groups with the
    main norm's gamma[:k d/G] times the matching embedding columns, and k = G gives the main head's logits."""
    from dd.model import lm_loss
    from dd.train import FUNNEL_NARROW
    torch.manual_seed(27)
    cfg = Cfg(vocab=50, d=64, n_layers=8, n_heads=8, ffn_hidden=96, max_len=32, budget=False, depth_groups=8,
              write_groups=FUNNEL_NARROW, mat_weight=0.2, mat_strict=True)
    m = Model(cfg)
    assert not hasattr(m, "mat_w")
    with torch.no_grad():
        m.nf.w.copy_(torch.rand(64) + 0.5)
    idx = torch.randint(2, 50, (2, 32))
    logits, a = m(idx)
    x = a["x_final"]
    assert torch.allclose(m.prefix_logits(x, 8), logits, atol=1e-5)
    for k in (1, 4):
        n = 8 * k
        xs = x[..., :n].float()
        want = (xs * torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + 1e-6) * m.nf.w[:n]) @ m.emb.weight[:, :n].t()
        assert torch.allclose(m.prefix_logits(x, k), want, atol=1e-5)
    loss, ce, ce_k, _, _ = lm_loss(m, idx)
    assert torch.allclose(loss, ce + 0.2 * ce_k)
    loss.backward()
    assert float(m.nf.w.grad.abs().sum()) > 0


def test_r20_a_mat_exit_loss():
    """Round 20, A-mat: loss = CE(main head on the final state) + w * CE(main head on the state after a random block k);
    the extra CE equals the logit-lens CE after one of the blocks; the model is plain A otherwise."""
    from dd.model import lm_loss
    torch.manual_seed(28)
    m = Model(small(budget=False, exit_weight=0.2))
    a = Model(small(budget=False))
    a.load_state_dict(m.state_dict())
    idx = torch.randint(2, 50, (2, 32))
    assert torch.equal(m(idx)[0], a(idx)[0])
    loss, ce, ce_k, _, _ = lm_loss(m, idx)
    assert torch.allclose(loss, ce + 0.2 * ce_k)
    _, aux = m(idx, collect=True)
    tgt = idx[:, 1:]
    lens = [F.cross_entropy((m.nf(x) @ m.emb.weight.t())[:, :-1].flatten(0, 1), tgt.flatten()) for x in aux["xs"][1:]]
    assert min(abs(float(ce_k) - float(c)) for c in lens) < 1e-5
    loss.backward()
    assert float(m.blocks[0].w1.weight.grad.abs().sum()) > 0
