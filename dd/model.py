"""Decoder transformer with an optional per-coordinate / per-group / per-token *depth budget*.

Normalised-state scheme (norm_state=True, the main one after the scale-loophole finding):
    x0 = normalize(emb)                              normalize = RMS 1 per token or per group, no weight
    t  = tanh(a + u)            (variant C: tanh(u))
    update="min" :  x = normalize(x + min(dd,1) * t)
    update="dd"  :  x = normalize(x + dd * t)
    update="pull":  x = normalize(x + min(dd,1) * (t - x))
    depth += dd   (always the full dd)
A_norm (budget=False, norm_state=True):  x = normalize(x + tanh(a + u)).

Legacy scheme (norm_state=False), kept to reproduce the loophole:

Budget block l (0-indexed), depth has shape (B, T, G), G in {d, n_heads, 1}:
    allowed = (l + 1) - depth
    a   = attention(rmsnorm(x) + depth_emb_a(depth))
    y   = x + a                                  # temporary, only feeds the FFN
    h   = silu(W1 z) * (W3 z),  z = rmsnorm(y) + depth_emb_f(depth)
    u   = W2 h
    dd  = allowed * sigmoid(W_depth h + b)
    depth = depth + dd
    x   = x + s(dd) * upd          upd = a + u (or u for variant C); s = dd (or min(dd,1), D)
    cost += mean(dd)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Cfg:
    vocab: int = 321
    d: int = 256
    n_layers: int = 8
    n_heads: int = 8
    ffn_hidden: int = 688
    max_len: int = 128
    budget: bool = False          # False -> plain residual (variant A)
    depth_groups: int = 256       # 256 = per coordinate, 8 = per head group, 1 = per token
    update: str = "full"          # legacy: full | ffn | clip ;  normalised: min | dd | pull
    update_src: str = "attn_ffn"  # normalised scheme: tanh(a+u) or "ffn" -> tanh(u) (variant C)
    norm_state: bool = False      # normalise the state after every update (RMS 1, no weight)
    norm_groups: int = 1          # 1 = per token, n_heads = per head group
    attn_depth_bias: bool = False  # variant G
    depth_rank: int = 32
    gate_bias_init: float = 0.0
    min_step: float = 0.0         # optional fix for dead gates: dd = allowed*(eps+(1-eps)*sig)
    lam: float = 0.0
    doc_mask: bool = True         # attention restricted to the current sentence (<bos> = id 1)
    anchor: bool = False          # fix for the scale loophole: RMS-1 embedding + RMS-normalised update
    logit_clamp: float = 6.0      # leaky gate: backward uses sigmoid'(clamp(z,-c,c)); 0 = off
    budget_global: bool = False   # H/I/J: dd in [0, R - depth], R = n_layers (no per-layer cap)
    rel_cost: bool = False        # I/J: cost = mean_i |dx_i| / s_i  (s_i = EMA RMS of coordinate i)
    state_noise: float = 0.0      # J: training noise on the state, sigma = state_noise * s_i
    gate_type: str = "sigmoid"    # "hc" = variant K: hard-concrete shift gate per group, decided before the FFN
    hc_beta: float = 2.0 / 3.0
    hc_gamma: float = -0.1
    hc_zeta: float = 1.1
    hc_bias_init: float = 2.0
    per_block_mu: bool = False    # K25L: one multiplier per block
    hc_tanh: bool = True          # K: bound the update with tanh (K25-free / K25-norm: False)
    # round 4
    quant: bool = False           # Q: state on a fixed grid delta*{-K..K}, no gates, no normalisation
    q_delta: float = 0.15
    q_levels: int = 8
    q_cost: str = "none"          # none | l1 (mean levels crossed) | l0 (share of coordinates that move)
    q_emb_init: float = 1.0       # embedding init std in units of delta (so the quantised embedding is not all 0)
    rel_scale: str = "rms"        # I: s_i = EMA RMS;  I': "median" = EMA of batch median |x_i|, floor 0.1*mean|x_i|
    med_floor: float = 0.1
    free_per_group: int = 0       # "+free": this many coordinates per norm group are excluded from normalisation
    # round 5
    qg: str = ""                  # QG: quantised state + per-group decision BEFORE the FFN: "gate" | "nogate"
    qg_split: bool = False        # QG-split: FFN cut into G blocks (block g writes only group g), head h -> group h
    # round 6: which parts qg_split cuts -- "both" (round 5) | "ffn" (W_O full) | "head" (FFN shared)
    qg_split_parts: str = "both"
    # round 6, part 2: grid step per state (index 0 = embedding, l + 1 = output of block l); empty = q_delta everywhere.
    # The levels -K..K are the same for every state.  emb_scale multiplies the embedding before quantisation
    # (input side only; the tied output layer uses the raw embedding).
    q_delta_layers: tuple = ()
    emb_scale: float = 1.0
    # round 7: gate controls for QG -- the open mask is not learned (no gate head, no cost multiplier):
    #   "random"   every (token, block, group) open with prob. qg_mask_p, redrawn at every forward pass
    #   "static"   round(qg_mask_p * G) fixed groups per block, the same on every token (each group open in the
    #              same number of blocks)
    #   "position" random with a probability that depends on the position in the story (since the last EOT):
    #              bins with lower edges qg_pos_edges -> qg_pos_probs; qg_pos_p0 where no EOT is visible yet
    qg_mask: str = ""
    qg_mask_p: float = 0.25
    qg_pos_edges: tuple = ()
    qg_pos_probs: tuple = ()
    qg_pos_p0: float = 0.25
    # round 7: what the learned QG gate sees -- "state" (QG: rmsnorm(x + a) + depth embedding, i.e. state and
    # context) | "token" (QG-tokengate: only the embedding of the current token, rmsnorm per block; same target)
    qg_gate_in: str = "state"
    # round 8: QG-staticinit -- gate bias +hc_bias_init on the static groups ((s*l + j) mod G, j < s, s = qg_mask_p*G)
    # and -hc_bias_init elsewhere, so training starts from the static mask; QG-hybrid -- the first qg_force_open of
    # those groups are always open (not decided by the gate), the gate decides the rest under the same total budget
    qg_static_init: bool = False
    qg_force_open: int = 0
    # round 8: QG-ctrl -- a separate control stream c (qg_ctrl dims, no grid) per token: c_0 = 0, in every block
    # c <- c + W_c zin (always), then the gate reads [zin, RMSNorm(c)]; c reaches nothing but the gates
    qg_ctrl: int = 0
    # round 9: round only the update (x + delta*round((a+u)/delta), clipped to +-K delta) instead of re-rounding x + a + u
    q_upd_round: bool = False
    # round 9, H-Net-style gate training: p = sigmoid(z), m = 1[p > 0.5] (no straight-through in the write path); an
    # open group writes Q(x + s(p) * (a + u)), so CE reaches the gate through s: "hnet" s = p, "dimmer" s = 2p - 1;
    # the cost keeps the sigmoid surrogate of QG25
    qg_smooth: str = ""
    # round 9 fix: a closed group (p <= 0.5) writes nothing forward but gets the QG25 straight-through gradient
    # through the sigmoid, as if opening wrote the full update (a + u); open groups keep the gradient through s(p)
    qg_smooth_closed_ste: bool = False
    # round 10, "nogrid": no quantisation anywhere -- the embedding and an open group's write x + a + u are neither
    # rounded nor clipped (the clipped straight-through becomes the identity); a closed group is still not written
    # (bit-identical).  Everything else (init, emb_scale, gate, cost, mu) as in the gridded variant.
    q_nogrid: bool = False
    # round 10, QG-ordered: the groups of block l form a queue static_groups(l, G) = (2l, 2l+1, ...) mod G; the gate
    # head gives ONE logit z per token and block, and the group at queue position j opens iff z > j, so the gate
    # chooses only k = #{j: z > j} (the first k groups of the queue).  zg_g = z - j(g); straight-through as in QG
    # (forward 1[zg > 0], backward d sigmoid(zg)); cost = share of open groups, as in QG.  Bias init (G - 1) +
    # hc_bias_init: everything open at the start, the last group of the queue with QG's margin.
    qg_ordered: bool = False
    # round 11: a separate Lagrange multiplier mu_l per block, each block with its own target share (QG-ordered-
    # perblock); cost = sum_l mu_l * paid_l / L (= mu * mean(paid) when all mu_l are equal)
    qg_mu_per_block: bool = False
    # round 11, static-randk (qg_mask="randk"): no gate; in block l the first k groups of the queue static_groups(l, G)
    # are open, k drawn for every (token, block) at every forward pass from the pmf qg_randk_probs over k = 0..G
    qg_randk_probs: tuple = ()
    # round 11, static-unequal (qg_mask="static"): block l opens the first qg_static_k[l] groups of its queue
    # instead of round(qg_mask_p * G)
    qg_static_k: tuple = ()
    # round 12 (size ladder): recompute the depth-embedding lookups in the backward pass instead of storing their
    # (B, T, G, d) fp32 intermediates (activation checkpointing; forward and gradients unchanged, memory only)
    depth_lookup_recompute: bool = False
    # round 13, QG-ordered with k limited to [kmin, kmax]: queue positions j < kmin always open, j >= kmax always
    # closed (both hard, no gradient), kmin <= j < kmax decided by z > j as in QG-ordered.  Bias init
    # (kmax - 1) + hc_bias_init: kmax groups open at the start, the last decidable one with QG's margin.
    qg_k_range: tuple = ()
    # round 14, VQ state on top of A (no grid, no gate, no cost): "shared" -- every group snaps its state (after the
    # embedding and after every block) to the nearest vector of ONE codebook per group (vq_k vectors, all depths);
    # "delta" -- every group and block has its own codebook of updates (vq_k vectors, vector 0 = the zero update,
    # fixed) and the state gets the nearest update.  Straight-through, commitment vq_beta * MSE(z, sg(q)), codebooks
    # by EMA (vq_decay, Laplace smoothing), a vector whose EMA count per step drops below vq_dead is restarted from
    # the batch.
    vq: str = ""
    vq_k: int = 1024
    vq_beta: float = 0.25
    vq_decay: float = 0.99
    vq_dead: float = 1.0
    # round 15, normalised VQ: codebook vectors are unit length, the symbol of a group is the codebook vector with the
    # largest cosine to the group's direction u = z / |z|; the length |z| is kept as a separate (continuous) scalar,
    # q = |z| e_k.  Gradient: exact through the length, straight-through through the direction.  Commitment
    # vq_beta * |u - sg(e_k)|^2 (scale-free); EMA on unit vectors, renormalised.  delta: no zero vector in the book
    # (a zero update is length -> 0).
    vq_norm: bool = False
    # round 17: gradual insertion of the book, output = (1 - vq_gamma) z + vq_gamma snap(z) (vq_gamma set by the trainer)
    vq_mix: bool = False
    # round 16, books-shared-attractor: one codebook of vq_k unit vectors per group, shared by all blocks (a learned
    # parameter, normalised on the fly).  After block l: x_hat_g = normalize(x_g + update_g), then soft attraction
    # x_g <- attract_scale[l] * C_g softmax(beta C_g^T x_hat_g); beta (buffer att_beta) is set by the trainer;
    # model.att_hard = True replaces the softmax by the arg max (hard snapping).  attract_scale[l] = A's group norm
    # at that depth (keeps the residual / update balance of the base model).
    attract: bool = False
    attract_scale: tuple = ()
    # round 18, funnel: block l writes only the groups write_groups[l] (depth_groups groups of d / G contiguous
    # coordinates).  Its output projections are narrow -- o: (w_l x d), w2: (w_l x ffn_hidden), w_l = |groups| d / G --
    # and one update a + u (attention and FFN summed as in QG; the FFN reads x + a) is added at those coordinates; all
    # other coordinates are bit-identical.  Attention, FFN and the output read the whole vector.
    write_groups: tuple = ()
    # round 18, funnel-narrow-mat: prefix heads logits_k = RMSNorm_k(x[:k d/G]) E[:, :k d/G]^T (E = the tied
    # embedding, one RMSNorm per k); loss = CE + mat_weight * CE(logits_k), k ~ U{1..G} once per batch
    mat_weight: float = 0.0
    # round 20, mat-strict: the prefix head is the main head cut to the first k groups -- RMSNorm over those k d/G
    # coordinates with the MAIN norm's weights gamma[:k d/G], times E[:, :k d/G]^T; k = G is exactly the main head;
    # no separate head parameters
    mat_strict: bool = False
    # round 20, A-mat: plain A with early-exit pressure by depth -- loss = CE(head(x_L)) + exit_weight * CE(head(x_k)),
    # x_k = the state after block k, k ~ U{1..L} once per batch, head = the main output head (final norm + E)
    exit_weight: float = 0.0

    def static_groups(self, l: int, k: int | None = None) -> list:
        s = int(round(self.qg_mask_p * self.depth_groups))
        return [(s * l + j) % self.depth_groups for j in range(s if k is None else k)]

    @property
    def split_ffn(self):
        return self.qg_split and self.qg_split_parts in ("both", "ffn")

    @property
    def split_head(self):
        return self.qg_split and self.qg_split_parts in ("both", "head")

    def qd(self, i: int) -> float:
        """grid step of state i (0 = embedding, l + 1 = output of block l)"""
        return float(self.q_delta_layers[i]) if self.q_delta_layers else self.q_delta

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.w


def rope_cache(T, hd, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2, device=device).float() / hd))
    t = torch.arange(T, device=device).float()
    f = torch.outer(t, inv)  # T, hd/2
    return f.cos(), f.sin()


def normalize(x, groups=1, eps=1e-12, floor=0.0, free=0):
    """RMS = 1 per token (groups=1) or per contiguous group of coordinates; no learnable weight.
    floor > 0: divide by max(RMS, floor) -- exact normalisation for ordinary states, but the gain
    (and the gradient) is bounded by 1/floor when an update nearly cancels the state.
    free > 0: the last `free` coordinates of every group are passed through unnormalised (and do not
    enter the group's RMS)."""
    sh = x.shape
    if free:
        xg = x.view(*sh[:-1], groups, sh[-1] // groups)
        a, b = xg[..., :-free], xg[..., -free:]
        r = torch.sqrt(a.pow(2).mean(-1, keepdim=True) + eps)
        if floor > 0:
            r = torch.clamp_min(r, floor)
        return torch.cat([a / r, b], -1).view(sh)
    xg = x if groups == 1 else x.view(*sh[:-1], groups, sh[-1] // groups)
    r = torch.sqrt(xg.pow(2).mean(-1, keepdim=True) + eps)
    if floor > 0:
        r = torch.clamp_min(r, floor)
    return (xg / r).view(sh)


def quantize(v, delta, K):
    """Q: forward delta * clamp(round(v / delta), -K, K) EXACTLY (bit-identical for equal levels);
    backward identity inside [-K-0.5, K+0.5]*delta, zero outside (clipped straight-through).
    Returns (x_q, level); level carries the same straight-through gradient (in level units)."""
    vl = v / delta
    r = torch.round(vl).clamp(-K, K)
    vlc = vl.clamp(-K - 0.5, K + 0.5)
    lvl = r + (vlc - vlc.detach())
    return delta * lvl, lvl


def quantize_update(x, upd, delta, K):
    """round only the update (round 9, "updround"): x_new = clamp(x + delta * round(upd / delta), -K*delta, K*delta).
    x itself is not re-rounded to the new grid; straight-through gradient for the rounding, zero outside the range.
    Returns (x_new, step) with step = round(upd / delta) in levels (straight-through)."""
    ul = upd / delta
    step = torch.round(ul) + (ul - ul.detach())
    lim = K * delta
    return (x + delta * step).clamp(-lim, lim), step


STATE_FLOOR = 0.1


def apply_rope(x, cos, sin):  # x: B,H,T,hd
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None].to(x.dtype), sin[None, None].to(x.dtype)
    out = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)
    return out.flatten(-2)


def sdpa(q, k, v, mask, scale):
    if mask is None:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


def doc_causal_mask(idx, bos=1):
    seg = torch.cumsum((idx == bos).int(), dim=1)
    T = idx.shape[1]
    causal = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
    return ((seg[:, :, None] == seg[:, None, :]) & causal)[:, None]   # B,1,T,T


class DepthEmb(nn.Module):
    """depth_emb(depth) = ((depth / L) @ P) @ Q, rank r, Q zero-initialised."""

    def __init__(self, G, d, L, r):
        super().__init__()
        self.L = L
        self.P = nn.Parameter(torch.randn(G, r) / math.sqrt(G))
        self.Q = nn.Parameter(torch.zeros(r, d))

    def forward(self, depth):
        return (((depth / self.L).to(self.P.dtype) @ self.P) @ self.Q)


class Block(nn.Module):
    def __init__(self, cfg: Cfg, l: int):
        super().__init__()
        self.cfg, self.l = cfg, l
        self.track = False          # set True (eager only) to record min pre-normalisation RMS
        self.pre_rms_min = float("nan")
        d, H = cfg.d, cfg.n_heads
        self.H, self.hd = H, d // H
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.w1 = nn.Linear(d, cfg.ffn_hidden, bias=False)
        self.w3 = nn.Linear(d, cfg.ffn_hidden, bias=False)
        self.w2 = nn.Linear(cfg.ffn_hidden, d, bias=False)
        self.hc_sample = False          # K: force sampled (training-mode) gates in eval
        self.o_mask = self.w2_mask = None
        if cfg.write_groups:            # round 18: narrow output projections, written columns
            G = cfg.depth_groups
            assert d % G == 0 and len(cfg.write_groups) == cfg.n_layers and not (cfg.budget or cfg.quant)
            gs = d // G
            grp = sorted(cfg.write_groups[l])
            assert grp and len(set(grp)) == len(grp) and all(0 <= g < G for g in grp)
            cols = torch.cat([torch.arange(g * gs, (g + 1) * gs) for g in grp])
            self.register_buffer("w_cols", cols, persistent=False)
            self.o = nn.Linear(d, len(cols), bias=False)
            self.w2 = nn.Linear(cfg.ffn_hidden, len(cols), bias=False)
        if cfg.qg_split:            # QG-split, and (round 7) A-split: the same cut without gate and grid
            G = cfg.depth_groups
            assert cfg.qg_split_parts in ("both", "ffn", "head") and d % G == 0
            gs, hb = d // G, cfg.ffn_hidden // G
            go = torch.arange(d) // gs                                       # group of each output coordinate
            if cfg.split_head:
                assert G == H
                del self.o_mask
                self.register_buffer("o_mask", (go[:, None] == torch.arange(d)[None] // self.hd).float(),
                                     persistent=False)
            if cfg.split_ffn:
                assert cfg.ffn_hidden % G == 0
                del self.w2_mask
                self.register_buffer("w2_mask", (go[:, None] == torch.arange(cfg.ffn_hidden)[None] // hb).float(),
                                     persistent=False)
        assert not cfg.q_nogrid or cfg.qg, "nogrid is implemented for the QG path only"
        if cfg.qg:
            G = cfg.depth_groups
            assert d % G == 0
            self.G = G
            assert not cfg.q_nogrid or not cfg.qg_smooth, "nogrid: QG / static path only"
            if cfg.qg == "gate" and not cfg.qg_mask:
                self.gate_head = nn.Linear(d + cfg.qg_ctrl, 1 if cfg.qg_ordered else G, bias=True)
                assert not cfg.qg_k_range or (cfg.qg_ordered and len(cfg.qg_k_range) == 2
                                              and 0 <= cfg.qg_k_range[0] < cfg.qg_k_range[1] <= G)
                if cfg.qg_ordered:                                       # queue position of every group
                    assert not (cfg.qg_smooth or cfg.qg_static_init or cfg.qg_force_open)
                    rk = torch.zeros(G)
                    rk[cfg.static_groups(l, G)] = torch.arange(G, dtype=torch.float32)
                    self.register_buffer("q_rank", rk, persistent=False)
                if cfg.qg_gate_in == "token":
                    self.ng = RMSNorm(d)
                if cfg.qg_ctrl:
                    self.wc = nn.Linear(d, cfg.qg_ctrl, bias=False)
                    self.nc = RMSNorm(cfg.qg_ctrl)
            if cfg.qg_mask == "static":
                sm = torch.zeros(G)
                sm[cfg.static_groups(l, cfg.qg_static_k[l] if cfg.qg_static_k else None)] = 1.0
                self.register_buffer("static_mask", sm, persistent=False)
            assert not cfg.qg_static_k or (cfg.qg_mask == "static" and len(cfg.qg_static_k) == cfg.n_layers)
            if cfg.qg_mask == "randk":                                   # round 11: random k, prefix of the queue
                assert len(cfg.qg_randk_probs) == G + 1 and abs(sum(cfg.qg_randk_probs) - 1) < 1e-6
                rk = torch.zeros(G)
                rk[cfg.static_groups(l, G)] = torch.arange(G, dtype=torch.float32)
                self.register_buffer("q_rank", rk, persistent=False)
                cdf = torch.cumsum(torch.tensor(cfg.qg_randk_probs, dtype=torch.float64), 0)[:-1]
                self.register_buffer("randk_cdf", cdf.float(), persistent=False)   # k = #{j < G: cdf_j <= u}
            if cfg.qg_force_open:
                fm = torch.zeros(G)
                fm[cfg.static_groups(l, cfg.qg_force_open)] = 1.0
                self.register_buffer("force_mask", fm, persistent=False)
            self.dt_a = nn.Parameter(torch.zeros(G, cfg.n_layers + 1, d))
            self.dt_f = nn.Parameter(torch.zeros(G, cfg.n_layers + 1, d))
        elif cfg.budget and cfg.gate_type in ("hc", "ste"):
            G = cfg.depth_groups
            self.G = G
            self.gate_head = nn.Linear(d, G, bias=True)
            # depth embedding by integer level, one table per group, linear interpolation for
            # fractional depth during training; zero-initialised like Q.
            self.dt_a = nn.Parameter(torch.zeros(G, cfg.n_layers + 1, d))
            self.dt_f = nn.Parameter(torch.zeros(G, cfg.n_layers + 1, d))
            if cfg.attn_depth_bias:
                assert G == H
                self.slope = nn.Parameter(torch.zeros(H))
        elif cfg.budget:
            G = cfg.depth_groups
            assert d % G == 0
            self.G = G
            self.de_a = DepthEmb(G, d, cfg.n_layers, cfg.depth_rank)
            self.de_f = DepthEmb(G, d, cfg.n_layers, cfg.depth_rank)
            self.wd = nn.Linear(cfg.ffn_hidden, G, bias=True)
            if cfg.attn_depth_bias:
                assert G == H, "attention depth bias needs one depth per head group"
                self.slope = nn.Parameter(torch.zeros(H))
            if cfg.rel_cost or cfg.state_noise > 0:
                self.register_buffer("s_ema", torch.ones(d))       # EMA of per-coordinate RMS (input state)
                self.register_buffer("s_init", torch.zeros(()))
                if cfg.rel_scale == "median":
                    self.register_buffer("s_med", torch.ones(d))   # I': EMA of batch median |x_i|
                    self.register_buffer("s_mabs", torch.ones(d))  # I': EMA of batch mean |x_i| (floor)

    # --- attention -----------------------------------------------------------------
    def attn(self, z, rope, depth=None, mask=None):
        B, T, d = z.shape
        H, hd = self.H, self.hd
        q, k, v = self.qkv(z).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        scale = 1.0 / math.sqrt(hd)
        if depth is not None and self.cfg.attn_depth_bias:
            # scores += slope_h * (depth_q - depth_k).  The depth_q term is constant along each
            # softmax row and cancels exactly, so we add only -slope_h * depth_k, implemented as
            # one extra q/k dimension (padded to a multiple of 8 for the fused kernels).
            dk = depth.transpose(1, 2).to(q.dtype)                       # B,H,T
            qx = (self.slope.to(q.dtype) / scale).view(1, H, 1, 1).expand(B, H, T, 1)
            kx = (-dk).unsqueeze(-1)
            pad = q.new_zeros(B, H, T, 7)
            q = torch.cat([q, qx, pad], -1)
            k = torch.cat([k, kx, pad], -1)
            v = torch.cat([v, v.new_zeros(B, H, T, 8)], -1)
            y = sdpa(q, k, v, mask, scale)[..., :hd]
        else:
            y = sdpa(q, k, v, mask, scale)
        y = y.transpose(1, 2).reshape(B, T, d)
        if self.o_mask is not None:                 # QG-split: head h writes only into group h
            return F.linear(y, self.o.weight * self.o_mask)
        return self.o(y)

    def expand(self, t):  # B,T,G -> B,T,d
        G, d = self.G, self.cfg.d
        if G == d:
            return t
        if G == 1:
            return t.expand(*t.shape[:-1], d)
        return t.repeat_interleave(d // G, dim=-1)

    def forward(self, x, depth, rope, ctl=None, mask=None):
        """ctl: optional dict for analysis: dd_zero_below (float) -> treat small dd as exact 0."""
        cfg = self.cfg
        if cfg.quant:
            return self._forward_q(x, depth, rope, ctl, mask)
        if not cfg.budget:
            if cfg.write_groups:                    # round 18: one write into the block's own groups
                a = self.attn(self.n1(x), rope, mask=mask)
                z = self.n2(x.index_add(-1, self.w_cols, a.to(x.dtype)))
                u = self.w2(F.silu(self.w1(z)) * self.w3(z))
                return x.index_add(-1, self.w_cols, (a + u).to(x.dtype)), depth, None, None
            a = self.attn(self.n1(x), rope, mask=mask)
            if cfg.norm_state:                      # A_norm: same normalised update, no gate
                z = self.n2(x + a)
                u = self.w2(F.silu(self.w1(z)) * self.w3(z))
                t = torch.tanh((a + u).float())
                if ctl is not None and ctl.get("free_fn") is not None:
                    t = ctl["free_fn"](self, t)
                return normalize(x + t, cfg.norm_groups, floor=STATE_FLOOR,
                                 free=cfg.free_per_group), depth, None, None
            x = x + a
            z = self.n2(x)
            h = F.silu(self.w1(z)) * self.w3(z)
            x = x + (F.linear(h, self.w2.weight * self.w2_mask) if self.w2_mask is not None else self.w2(h))
            return x, depth, None, None
        if cfg.gate_type in ("hc", "ste"):
            return self._forward_hc(x, depth, rope, ctl, mask)
        a = self.attn(self.n1(x) + self.de_a(depth).to(x.dtype), rope, depth, mask)
        y = x + a
        z = self.n2(y) + self.de_f(depth).to(x.dtype)
        h = F.silu(self.w1(z)) * self.w3(z)
        u = self.w2(h)
        R = float(cfg.n_layers if cfg.budget_global else self.l + 1)
        allowed = R - depth                                              # fp32
        zl = self.wd(h).float()
        g = torch.sigmoid(zl)
        if cfg.logit_clamp > 0:
            # Leaky gate (against dead logits).  Forward value is the exact sigmoid, so exact zeros
            # stay possible.  The gate's OWN parameters (W_depth, b) get the gradient evaluated at
            # max(z, -c), so a gate saturated near 0 still learns; the rest of the network gets the
            # true sigmoid'(z).  (Leaking into h as well made training diverge: W_depth grew without
            # bound and the fake gradient flowing back through it exploded.)
            zp = F.linear(h.detach(), self.wd.weight, self.wd.bias).float()
            zc = zp - (zp - zp.clamp_min(-cfg.logit_clamp)).detach()
            sc, sp = torch.sigmoid(zc), torch.sigmoid(zp)
            g = g + (sc - sc.detach()) - (sp - sp.detach())
        if cfg.min_step > 0:
            g = cfg.min_step + (1 - cfg.min_step) * g
        if ctl is not None and ctl.get("gate_fn") is not None:      # analysis: substitute gates
            g = ctl["gate_fn"](self, g, z, allowed)
        dd = allowed * g
        if ctl is not None and ctl.get("dd_zero_below") is not None:
            dd = torch.where(dd < ctl["dd_zero_below"], torch.zeros_like(dd), dd)
        depth0 = depth
        depth = torch.clamp_max(depth + dd, R)
        if cfg.update in ("min", "dd", "pull"):
            return self._tanh_update(x, a, u, dd, depth0, depth, R, ctl)
        if cfg.norm_state:
            t = torch.tanh((u if cfg.update_src == "ffn" else a + u).float())
            if cfg.update == "dd":
                x = x + self.expand(dd) * t
            elif cfg.update == "min":
                x = x + self.expand(torch.clamp_max(dd, 1.0)) * t
            elif cfg.update == "pull":
                x = x + self.expand(torch.clamp_max(dd, 1.0)) * (t - x)
            else:
                raise ValueError(cfg.update)
            self.pre_rms_min = float("nan") if not self.track else float(
                x.detach().view(*x.shape[:-1], cfg.norm_groups, -1).pow(2).mean(-1).sqrt().min())
            return normalize(x, cfg.norm_groups, floor=STATE_FLOOR), depth, dd, None
        s = dd if cfg.update != "clip" else torch.clamp_max(dd, 1.0)
        upd = u if cfg.update == "ffn" else a + u
        if cfg.anchor:
            uf = upd.float()
            upd = (uf * torch.rsqrt(uf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(upd.dtype)
        x = x + self.expand(s).to(x.dtype) * upd
        return x, depth, dd, None

    def depth_lookup(self, table, depth):
        """table: (G, L+1, d); depth: (B,T,G) in [0, L] -> sum_g table[g, depth_g] (linear interp.)"""
        if self.cfg.depth_lookup_recompute and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(self._depth_lookup, table, depth, use_reentrant=False,
                                                     preserve_rng_state=False)
        return self._depth_lookup(table, depth)

    def _depth_lookup(self, table, depth):
        L = self.cfg.n_layers
        dcl = depth.clamp(0, L)
        k = dcl.floor().clamp(max=L - 1)
        f = (dcl - k).unsqueeze(-1)
        k = k.long()
        gi = torch.arange(self.G, device=depth.device).expand_as(k)
        return ((1 - f) * table[gi, k] + f * table[gi, k + 1]).sum(2)

    def _forward_hc(self, x, depth, rope, ctl, mask):
        """Variant K: per group, decide BEFORE the FFN whether to shift (hard-concrete gate m in [0,1],
        hard 0/1 in eval); a shifted group gets x_g = groupnorm(x_g + m tanh(a_g + u_g)), a standing
        group is left bit-identical; depth_g += m (number of shifts)."""
        cfg = self.cfg
        a = self.attn(self.n1(x) + self.depth_lookup(self.dt_a, depth).to(x.dtype), rope, depth, mask)
        y = x + a
        zin = self.n2(y) + self.depth_lookup(self.dt_f, depth).to(x.dtype)
        zg = self.gate_head(zin).float()                                   # B,T,G logits (log alpha)
        beta, gam, zeta = cfg.hc_beta, cfg.hc_gamma, cfg.hc_zeta
        if cfg.gate_type == "ste":
            return self._forward_ste(x, depth, a, zin, zg, ctl)
        if self.training or self.hc_sample:
            uu = torch.rand_like(zg).clamp(1e-6, 1 - 1e-6)
            sg = torch.sigmoid((torch.log(uu) - torch.log1p(-uu) + zg) / beta)
            m = (sg * (zeta - gam) + gam).clamp(0.0, 1.0)
        else:
            m = ((torch.sigmoid(zg) * (zeta - gam) + gam) > 0.5).float()
        if ctl is not None and ctl.get("hc_gate_fn") is not None:
            m = ctl["hc_gate_fn"](self, m, zg)
        p_open = torch.sigmoid(zg - beta * math.log(-gam / zeta))            # P(m > 0), closed form
        h = F.silu(self.w1(zin)) * self.w3(zin)
        u = self.w2(h)
        t = torch.tanh((a + u).float()) if cfg.hc_tanh else (a + u).float()
        mf = self.expand(m)
        if cfg.norm_state:
            x_new = torch.where(mf > 0, normalize(x + mf * t, cfg.norm_groups, floor=STATE_FLOOR), x)
        else:                                       # K25-free: plain gated residual, no normalisation
            x_new = torch.where(mf > 0, x + mf * t, x)
        return x_new, depth + m, m, p_open

    def _forward_ste(self, x, depth, a, zin, zg, ctl):
        """K-STE: forward m = 1[z > 0] (exact 0/1, identical in training and eval), backward d sigmoid(z).
        x_new = m * groupnorm(x + tanh(a + u)) + (1 - m) * x  -> a standing group is bit-identical and
        its gate still receives the gradient (groupnorm(x + t) - x) . dL/dx_new  (Skip-RNN style)."""
        cfg = self.cfg
        p = torch.sigmoid(zg)
        m = (zg > 0).float() + (p - p.detach())
        if ctl is not None and ctl.get("hc_gate_fn") is not None:
            m = ctl["hc_gate_fn"](self, m, zg)
        h = F.silu(self.w1(zin)) * self.w3(zin)
        u = self.w2(h)
        t = torch.tanh((a + u).float())
        mf = self.expand(m)
        xn = normalize(x + t, cfg.norm_groups, floor=STATE_FLOOR)
        x_new = mf * xn + (1 - mf) * x
        return x_new, depth + m, m, m          # 4th output = cost term: forward hard share, grad mean sigmoid

    def _forward_q(self, x, depth, rope, ctl, mask):
        """Q: y = x + a (temporary); u = FFN(rmsnorm(y)); x_new = delta*clamp(round((x + a + u)/delta), -K, K).
        Returns n = |level_new - level_old| (levels crossed, straight-through gradient) as `dd` and the
        L0 indicator 1[n > 0] with the gradient of min(|a+u|/delta, 1) as `extra`."""
        cfg = self.cfg
        if cfg.qg:
            return self._forward_qg(x, depth, rope, ctl, mask)
        D, K = cfg.qd(self.l + 1), cfg.q_levels
        a = self.attn(self.n1(x), rope, mask=mask)
        z = self.n2(x + a)
        h = F.silu(self.w1(z)) * self.w3(z)
        u = F.linear(h, self.w2.weight * self.w2_mask) if self.w2_mask is not None else self.w2(h)   # r7: Q0-split
        upd = (a + u).float()
        if ctl is not None and ctl.get("q_upd_fn") is not None:
            upd = ctl["q_upd_fn"](self, upd, x)
        if cfg.q_upd_round:
            x_new, step = quantize_update(x, upd, D, K)
            n = step.abs()
        else:
            lo = x / D
            lvl_old = torch.round(lo) + (lo - lo.detach())
            x_new, lvl_new = quantize(x + upd, D, K)
            n = (lvl_new - lvl_old).abs()
        soft = torch.clamp_max(upd.abs() / D, 1.0)
        n0 = (n.detach() > 0).float() + (soft - soft.detach())
        return x_new, depth, n, n0

    def _forward_qg(self, x, depth, rope, ctl, mask):
        """QG: quantised state (grid as in Q) + per-group decision before the FFN.
            a   = attention(rmsnorm(x) + depth_emb_a(depth))
            zin = rmsnorm(x + a) + depth_emb_f(depth)           (FFN input and gate input)
            gate:   m_g = 1[z_g > 0], backward d sigmoid(z_g);  z = gate_head(zin)
                    x_g <- m_g * Q(x_g + a_g + u_g) + (1 - m_g) * x_g      (standing group bit-identical)
                    depth_g += m_g;  returns (m, real) with real = 1[some coordinate of g changed]
            nogate: x <- Q(x + a + u);  moved_g = 1[some coordinate of g changed]  (forward)
                    cost surrogate: max_{i in g} min(|a_i + u_i| / delta, 1)        (backward)
                    depth_g += moved_g;  returns (moved with the surrogate gradient, moved)
        QG-split: o and w2 are block-diagonal (masks): u_g depends only on FFN block g, a_g only on head g."""
        cfg = self.cfg
        D, K, G = cfg.qd(self.l + 1), cfg.q_levels, self.G
        B, T, d = x.shape
        gs = d // G
        a = self.attn(self.n1(x) + self.depth_lookup(self.dt_a, depth).to(x.dtype), rope, mask=mask)
        zin = self.n2(x + a) + self.depth_lookup(self.dt_f, depth).to(x.dtype)
        h = F.silu(self.w1(zin)) * self.w3(zin)
        u = F.linear(h, self.w2.weight * self.w2_mask) if self.w2_mask is not None else self.w2(h)
        upd = (a + u).float()
        if ctl is not None and ctl.get("q_upd_fn") is not None:
            upd = ctl["q_upd_fn"](self, upd, x)
        if cfg.q_nogrid:                                                 # round 10: no rounding, no clipping
            x_c = x + upd
        elif cfg.q_upd_round:
            x_c = quantize_update(x, upd, D, K)[0]
        else:
            x_c = quantize(x + upd, D, K)[0]
        if cfg.qg == "gate":
            if cfg.qg_mask == "static":                                  # round 7 controls: mask not learned
                zg = None
                m = self.static_mask.expand(B, T, G)
            elif cfg.qg_mask == "randk":                                 # round 11: k ~ pmf per (token, block)
                zg = None
                u = torch.rand(B, T, 1, device=x.device)
                k = (u >= self.randk_cdf).sum(-1, keepdim=True).float()  # B, T, 1 in 0..G
                m = (self.q_rank < k).float()
            elif cfg.qg_mask:
                zg = None
                pm = ctl["mask_p"] if ctl is not None and ctl.get("mask_p") is not None else cfg.qg_mask_p
                m = (torch.rand(B, T, G, device=x.device) < pm).float()
            else:
                gin = self.ng(ctl["gate_tok"]) if cfg.qg_gate_in == "token" else zin
                if cfg.qg_ctrl:                                          # QG-ctrl: control stream, gates only
                    c = ctl["ctrl"] + self.wc(zin).float()
                    ctl["ctrl"] = c
                    gin = torch.cat([gin, self.nc(c).to(gin.dtype)], -1)
                zg = self.gate_head(gin).float()
                if cfg.qg_ordered:                                       # one z; group at queue position j: z - j
                    zg = zg - self.q_rank
                p = torch.sigmoid(zg)
                if cfg.qg_smooth:                                        # round 9: H-Net-style write
                    mh = (zg > 0).float()
                    if ctl is not None and ctl.get("hc_gate_fn") is not None:
                        mh = ctl["hc_gate_fn"](self, mh, zg).detach()
                    sg = p if cfg.qg_smooth == "hnet" else 2 * p - 1
                    if ctl is not None and ctl.get("smooth_s_fn") is not None:
                        sg = ctl["smooth_s_fn"](self, sg)
                    x_full = x_c                                         # Q(x + a + u): the QG25 write
                    x_c = quantize(x + self.expand(sg) * upd, D, K)[0]
                    mf = self.expand(mh)
                    x_new = mf * x_c + (1 - mf) * x
                    if cfg.qg_smooth_closed_ste:                         # closed groups: value 0, QG25 gradient
                        ste = (1 - mh) * (p - p.detach())
                        x_new = x_new + self.expand(ste) * (x_full - x)
                    real = (x_new.detach() != x.detach()).view(B, T, G, gs).any(-1).float()
                    return x_new, depth + mh, mh + (p - p.detach()), real
                m = (zg > 0).float() + (p - p.detach())
                if cfg.qg_k_range:                                       # round 13: k in [kmin, kmax]
                    kmin, kmax = cfg.qg_k_range
                    m = torch.where(self.q_rank < kmin, torch.ones_like(m),
                                    torch.where(self.q_rank < kmax, m, torch.zeros_like(m)))
                if cfg.qg_force_open:                                    # QG-hybrid: forced groups, no gate gradient
                    m = torch.where(self.force_mask.bool().expand_as(m), torch.ones_like(m), m)
            if ctl is not None and ctl.get("hc_gate_fn") is not None:
                m = ctl["hc_gate_fn"](self, m, zg)
            mf = self.expand(m)
            x_new = mf * x_c + (1 - mf) * x
            real = (x_new.detach() != x.detach()).view(B, T, G, gs).any(-1).float()
            return x_new, depth + m, m, real
        moved = (x_c.detach() != x.detach()).view(B, T, G, gs).any(-1).float()
        soft = torch.clamp_max(upd.abs() / D, 1.0).view(B, T, G, gs).amax(-1)
        return x_c, depth + moved, moved + (soft - soft.detach()), moved

    def s_floor(self):
        if self.cfg.rel_scale == "median":
            return torch.maximum(self.s_med, self.cfg.med_floor * self.s_mabs)
        s = self.s_ema
        return torch.maximum(s, 1e-3 * s.mean())

    def _tanh_update(self, x, a, u, dd, depth0, depth, R, ctl):
        """Round-3 update (E/G/H normalised per group; I/J unnormalised with relative-change cost)."""
        cfg = self.cfg
        t = torch.tanh((u if cfg.update_src == "ffn" else a + u).float())
        sc = dd if cfg.update == "dd" else torch.clamp_max(dd, 1.0)
        delta = self.expand(sc) * ((t - x) if cfg.update == "pull" else t)
        if ctl is not None and ctl.get("zero_own_below") is not None:
            # analysis: zero coordinates whose change is small relative to the token's own value
            sref = self.s_floor() if hasattr(self, "s_ema") else x.detach().pow(2).mean((0, 1)).sqrt()
            r = delta.abs() / (x.abs() + 0.1 * sref)
            delta = torch.where(r < ctl["zero_own_below"], torch.zeros_like(delta), delta)
        x_new = x + delta
        if cfg.norm_state:
            if self.track:
                self.pre_rms_min = float(x_new.detach().view(*x.shape[:-1], cfg.norm_groups, -1)
                                         .pow(2).mean(-1).sqrt().min())
            if ctl is not None and ctl.get("free_fn") is not None:
                x_new = x + ctl["free_fn"](self, x_new - x)
            x_new = normalize(x_new, cfg.norm_groups, floor=STATE_FLOOR, free=cfg.free_per_group)
        if ctl is not None and ctl.get("zero_rel_below") is not None:
            # analysis: groups whose real relative change is below thr keep their old state and depth
            B, T, d = x.shape
            Gq = self.G
            dx = (x_new - x).view(B, T, Gq, d // Gq).pow(2).mean(-1).sqrt()
            xr = x.view(B, T, Gq, d // Gq).pow(2).mean(-1).sqrt().clamp_min(1e-12)
            st = dx / xr < ctl["zero_rel_below"]
            x_new = torch.where(self.expand(st.float()).bool(), x, x_new)
            dd = torch.where(st, torch.zeros_like(dd), dd)
            depth = torch.clamp_max(depth0 + dd, R)
        extra = None
        if cfg.rel_cost:
            extra = (delta.abs() / self.s_floor()).mean(-1)           # B,T: mean_i |dx_i| / s_i
        if cfg.state_noise > 0 and self.training:
            x_new = x_new + cfg.state_noise * self.s_ema * torch.randn_like(x_new)
        return x_new, depth, dd, extra


class Model(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d)
        self.blocks = nn.ModuleList([Block(cfg, l) for l in range(cfg.n_layers)])
        self.nf = RMSNorm(cfg.d)
        self.register_buffer("rope_cos", torch.empty(0), persistent=False)
        self.register_buffer("rope_sin", torch.empty(0), persistent=False)
        if cfg.gate_type in ("hc", "ste") or cfg.q_cost != "none":
            self.register_buffer("mu", torch.zeros(cfg.n_layers))           # K / Q: Lagrange multiplier(s)
        if cfg.vq:                                                          # round 14: codebooks (buffers, EMA)
            assert cfg.vq in ("shared", "delta") and not (cfg.budget or cfg.quant or cfg.norm_state)
            G = cfg.depth_groups
            assert cfg.d % G == 0
            nb = 1 if cfg.vq == "shared" else cfg.n_layers
            E = torch.randn(nb, G, cfg.vq_k, cfg.d // G) * 0.02
            if cfg.vq_norm:
                E = F.normalize(E, dim=-1)
            elif cfg.vq == "delta":
                E[:, :, 0] = 0.0
            self.register_buffer("vq_E", E)
            self.register_buffer("vq_N", torch.ones(nb, G, cfg.vq_k))
            self.register_buffer("vq_M", E.clone())
            self.register_buffer("vq_ready", torch.zeros((), dtype=torch.bool))   # data init done
            # round 17: gradual insertion, output = (1 - g) z + g snap(z); set by the trainer, 1 = pure VQ
            self.register_buffer("vq_gamma", torch.tensor(1.0), persistent=False)
        if cfg.mat_weight > 0 and not cfg.mat_strict:                      # round 18: RMSNorm_k weights
            assert cfg.write_groups or not cfg.budget
            self.mat_w = nn.Parameter(torch.ones(cfg.depth_groups, cfg.d))
        if cfg.attract:                                                     # round 16: shared attractor book
            assert not (cfg.vq or cfg.budget or cfg.quant or cfg.norm_state) and len(cfg.attract_scale) == cfg.n_layers
            G = cfg.depth_groups
            self.att_C = nn.Parameter(F.normalize(torch.randn(G, cfg.vq_k, cfg.d // G), dim=-1))
            self.register_buffer("att_beta", torch.tensor(1.0))
            self.att_hard = False
        if cfg.qg_mask == "position":
            assert len(cfg.qg_pos_edges) == len(cfg.qg_pos_probs) and cfg.qg_pos_edges[0] == 0
            self.register_buffer("pos_edges", torch.tensor(cfg.qg_pos_edges[1:], dtype=torch.long), persistent=False)
            self.register_buffer("pos_probs", torch.tensor(cfg.qg_pos_probs, dtype=torch.float32), persistent=False)
        self._init()

    def pos_prob(self, idx, bos=1):
        """round 7, QG-position: open probability of every token from its position in the story (since the last
        EOT, EOT itself = 0; the same segments as doc_causal_mask); qg_pos_p0 before the first EOT of the window"""
        B, T = idx.shape
        ar = torch.arange(T, device=idx.device).expand(B, T)
        is_bos = idx == bos
        pos = ar - torch.cummax(torch.where(is_bos, ar, torch.zeros_like(ar)), dim=1).values
        seen = torch.cumsum(is_bos.int(), dim=1) > 0
        p = self.pos_probs[torch.bucketize(pos, self.pos_edges, right=True)]
        return torch.where(seen, p, torch.full_like(p, self.cfg.qg_pos_p0)).unsqueeze(-1)    # B,T,1

    def _init(self):
        std = 0.02
        for n, p in self.named_parameters():
            if p.dim() == 2 and not (n.endswith(".P") or n.endswith(".Q")):
                nn.init.normal_(p, 0.0, std)
            if n.endswith("o.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, 0.0, std / math.sqrt(2 * self.cfg.n_layers))
        if self.cfg.quant:
            nn.init.normal_(self.emb.weight, 0.0, self.cfg.qd(0) * self.cfg.q_emb_init / self.cfg.emb_scale)
        for b in self.blocks:
            with torch.no_grad():            # masked fan-in is G times smaller: keep the output scale at init
                if self.cfg.split_head:
                    b.o.weight.mul_(math.sqrt(self.cfg.n_heads))
                if self.cfg.split_ffn:
                    b.w2.weight.mul_(math.sqrt(self.cfg.depth_groups))
            if self.cfg.qg == "gate" and not self.cfg.qg_mask:
                nn.init.normal_(b.gate_head.weight, 0.0, 0.02)
                kq = (self.cfg.qg_k_range[1] if self.cfg.qg_k_range else self.cfg.depth_groups) - 1
                nn.init.constant_(b.gate_head.bias, self.cfg.hc_bias_init + (kq if self.cfg.qg_ordered else 0))
                if self.cfg.qg_static_init:                              # start from the static mask
                    with torch.no_grad():
                        b.gate_head.bias.fill_(-abs(self.cfg.hc_bias_init))
                        b.gate_head.bias[self.cfg.static_groups(b.l)] = abs(self.cfg.hc_bias_init)
            elif self.cfg.budget and self.cfg.gate_type in ("hc", "ste"):
                nn.init.normal_(b.gate_head.weight, 0.0, 0.02)
                nn.init.constant_(b.gate_head.bias, self.cfg.hc_bias_init)
            elif self.cfg.budget:
                nn.init.constant_(b.wd.bias, self.cfg.gate_bias_init)

    def rope(self, T, device):
        if self.rope_cos.numel() == 0 or self.rope_cos.shape[0] < T or self.rope_cos.device != device:
            c, s = rope_cache(max(T, self.cfg.max_len), self.cfg.d // self.cfg.n_heads, device)
            self.rope_cos, self.rope_sin = c, s
        return self.rope_cos[:T], self.rope_sin[:T]

    def vq(self, z, b, rec, off=False):
        """round 14: nearest vector of codebook b per group (exact fp32 distances, no gradient), straight-through;
        rec collects (book, idx, z, commitment) for the EMA update and the analysis.  round 17: off=True returns z
        unchanged (base model) and records only z (or its direction); vq_gamma < 1 mixes (1 - g) z + g snap(z)"""
        B, T, d = z.shape
        G = self.cfg.depth_groups
        zg = z.float().view(B, T, G, d // G)
        E = self.vq_E[b]                                                    # G, K, gs
        if off:
            r = zg.norm(dim=-1, keepdim=True)
            rec.append((b, None, (zg / r.clamp_min(1e-6) if self.cfg.vq_norm else zg).detach(), None, r.detach()[..., 0]))
            return z
        if self.cfg.vq_norm:                                                # round 15: cosine, separate length
            r = zg.norm(dim=-1, keepdim=True)
            u = zg / r.clamp_min(1e-6)
            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                idx = torch.einsum("btgs,gks->btgk", u, E).argmax(-1)       # B, T, G
            e = E[torch.arange(G, device=z.device)[None, None, :], idx]
            commit = (u - e).pow(2).sum(-1).mean()
            rec.append((b, idx, u.detach(), commit, r.detach()[..., 0]))
            out = r * (e + (u - u.detach()))                                # value |z| e_k
            if self.cfg.vq_mix:
                out = (1 - self.vq_gamma) * zg + self.vq_gamma * out
            return out.reshape(B, T, d)
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            d2 = (zg.pow(2).sum(-1, keepdim=True) - 2 * torch.einsum("btgs,gks->btgk", zg, E)
                  + E.pow(2).sum(-1)[None, None])
            idx = d2.argmin(-1)                                             # B, T, G
        q = E[torch.arange(G, device=z.device)[None, None, :], idx]         # B, T, G, gs
        commit = (zg - q).pow(2).mean()
        rec.append((b, idx, zg.detach(), commit, None))
        out = q + (zg - zg.detach())
        if self.cfg.vq_mix:
            out = (1 - self.vq_gamma) * zg + self.vq_gamma * out
        return out.reshape(B, T, d)                     # value exactly q, gradient identity

    def attract(self, x, l, rec, off=False):
        """round 16: x_hat = normalize per group; soft (softmax(beta cos)) or hard (arg max) attraction to the book;
        rec collects (block, symbol = arg max cos, x_hat, max softmax weight)"""
        B, T, d = x.shape
        G = self.cfg.depth_groups
        xh = F.normalize(x.float().view(B, T, G, d // G), dim=-1)
        if off:                                                             # analysis / init: A unchanged
            rec.append((l, None, xh.detach(), None))
            return x
        Cn = F.normalize(self.att_C, dim=-1)
        with torch.autocast("cuda", enabled=False):
            sims = torch.einsum("btgs,gks->btgk", xh, Cn)
            idx = sims.detach().argmax(-1)
            if self.att_hard:
                att, wmax = Cn[torch.arange(G, device=x.device)[None, None, :], idx], None
            else:
                w = torch.softmax(self.att_beta * sims, -1)
                att, wmax = torch.einsum("btgk,gks->btgs", w, Cn), w.detach().max(-1).values
        rec.append((l, idx, xh.detach(), wmax))
        return (float(self.cfg.attract_scale[l]) * att).reshape(B, T, d)

    def prefix_logits(self, x, k, w=None, eps=1e-6):
        """round 18: logits from the first k groups of the final state: RMSNorm over those k d/G coordinates (weights
        w, default the trained mat_w[k-1]) times the matching columns of the tied embedding.  k may be a tensor."""
        d, G = self.cfg.d, self.cfg.depth_groups
        n = k * (d // G)
        keep = (torch.arange(d, device=x.device) < n).to(torch.float32)
        xm = x.float() * keep
        xn = xm * torch.rsqrt(xm.pow(2).sum(-1, keepdim=True) / n + eps)
        if w is None:
            w = self.nf.w if self.cfg.mat_strict else self.mat_w[k - 1]   # strict: the main norm's gamma
        return (xn * w * keep).to(x.dtype) @ self.emb.weight.t().to(x.dtype)

    def forward(self, idx, collect=False, skip=(), ctl=None):
        """Returns logits, aux.  aux['dd'] = list of per-layer dd (B,T,G) (budget only).
        collect=True also returns residual states after the embedding and every block,
        and the depth after every block.  skip: block indices to bypass entirely."""
        B, T = idx.shape
        rope = self.rope(T, idx.device)
        mask = doc_causal_mask(idx) if self.cfg.doc_mask else None
        if self.cfg.qg_mask == "position":
            ctl = dict(ctl or {}, mask_p=self.pos_prob(idx))
        x = self.emb(idx)
        if self.cfg.qg_gate_in == "token":                  # QG-tokengate: the gate sees the token embedding only
            ctl = dict(ctl or {}, gate_tok=x)
        if self.cfg.qg_ctrl:                                 # QG-ctrl: c_0 = 0
            ctl = dict(ctl or {}, ctrl=torch.zeros(B, T, self.cfg.qg_ctrl, device=idx.device))
        if self.cfg.emb_scale != 1.0:
            x = x * self.cfg.emb_scale
        if self.cfg.quant:
            x = x.float() if self.cfg.q_nogrid else quantize(x.float(), self.cfg.qd(0), self.cfg.q_levels)[0]
        if self.cfg.norm_state:
            x = normalize(x.float(), self.cfg.norm_groups, free=self.cfg.free_per_group)
        if self.cfg.anchor:
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        vq_rec, att_rec, x_blocks = [], [], []
        vq_off = bool(ctl is not None and ctl.get("vq_off"))
        if self.cfg.vq == "shared":                         # round 14: the embedding state is a codebook vector too
            x = self.vq(x.float(), 0, vq_rec, vq_off)
        G = self.cfg.depth_groups if (self.cfg.budget or self.cfg.qg) else 1
        depth = torch.zeros(B, T, G, device=idx.device, dtype=torch.float32)
        dds, xs, depths, rcs, xin_ms, xin = [], [x] if collect else None, [], [], [], []
        need_ms = self.cfg.rel_cost or self.cfg.state_noise > 0
        need_med = need_ms and self.cfg.rel_scale == "median"
        tm = (idx != 0).float()[..., None] if need_ms else None
        for l, blk in enumerate(self.blocks):
            if need_ms:
                xin_ms.append((x.detach().float().pow(2) * tm).sum((0, 1)) / tm.sum().clamp_min(1))
            if need_med:
                xin.append(x.detach().float())
            if l in skip:
                if self.cfg.budget or self.cfg.quant:
                    dds.append(torch.zeros_like(depth))
            else:
                x_in = x
                x, depth, dd, extra = blk(x, depth, rope, ctl, mask)
                if self.cfg.exit_weight > 0 and not collect:
                    x_blocks.append(x)
                if self.cfg.vq == "shared":
                    x = self.vq(x, 0, vq_rec, vq_off)
                elif self.cfg.vq == "delta":
                    x = x_in.float() + self.vq(x.float() - x_in.float(), l, vq_rec, vq_off)
                if self.cfg.attract:
                    x = self.attract(x, l, att_rec, off=bool(ctl is not None and ctl.get("attract_off")))
                if dd is not None:
                    dds.append(dd)
                if extra is not None:
                    rcs.append(extra)
            if collect:
                xs.append(x)
                depths.append(depth)
        logits = self.nf(x) @ self.emb.weight.t().to(x.dtype)
        aux = {"dd": dds, "depth": depth, "rc": rcs, "xin_ms": xin_ms, "x_final": x}
        if self.cfg.exit_weight > 0:                        # round 20: states after every block for the exit loss
            aux["xs_blocks"] = xs[1:] if collect else x_blocks
        if att_rec:
            aux["att_idx"] = [r[1] for r in att_rec]
            aux["att_xh"] = [r[2] for r in att_rec]
            aux["att_wmax"] = [r[3] for r in att_rec]
        if vq_rec:
            aux["vq_book"] = [r[0] for r in vq_rec]
            aux["vq_idx"] = [r[1] for r in vq_rec]
            aux["vq_z"] = [r[2] for r in vq_rec]
            aux["vq_commit"] = (torch.stack([r[3] for r in vq_rec]).mean() if vq_rec[0][3] is not None
                                else torch.zeros((), device=idx.device))
            if self.cfg.vq_norm:
                aux["vq_r"] = [r[4] for r in vq_rec]
        if need_med:
            aux["xin"], aux["tmask"] = xin, idx != 0
        if collect:
            aux["xs"], aux["depths"] = xs, depths
        return logits, aux


def lm_loss(model, idx, lam=None):
    """CE over non-pad next tokens + lam * sum_l mean(dd_l) (mean over real tokens)."""
    logits, aux = model(idx)
    tgt = idx[:, 1:]
    m = tgt != 0
    ce = F.cross_entropy(logits[:, :-1][m].float(), tgt[m].long())
    lam = model.cfg.lam if lam is None else lam
    cost = torch.zeros((), device=idx.device)
    per_layer = []
    if model.cfg.exit_weight > 0:                       # round 20: + w * CE of the main head after a random block
        k = torch.randint(1, model.cfg.n_layers + 1, (), device=idx.device)
        xk = torch.stack(aux["xs_blocks"])[k - 1]
        lk = model.nf(xk) @ model.emb.weight.t().to(xk.dtype)
        ce_k = F.cross_entropy(lk[:, :-1][m].float(), tgt[m].long())
        return ce + model.cfg.exit_weight * ce_k, ce, ce_k, per_layer, aux
    if model.cfg.mat_weight > 0:                        # round 18: + w * CE of a random prefix head
        k = torch.randint(1, model.cfg.depth_groups + 1, (), device=idx.device)
        lk = model.prefix_logits(aux["x_final"], k)
        ce_k = F.cross_entropy(lk[:, :-1][m].float(), tgt[m].long())
        return ce + model.cfg.mat_weight * ce_k, ce, ce_k, per_layer, aux
    if model.cfg.vq:                                    # round 14: CE + beta * commitment (cost = commitment)
        cm = aux["vq_commit"]
        return ce + model.cfg.vq_beta * cm, ce, cm, per_layer, aux
    if model.cfg.qg:
        # QG: cost = share of (token, group) allowed to change (gate: forward hard, gradient mean sigmoid;
        # nogate: share that changed, gradient of the max-surrogate).  q_l0_layer = paid share (drives mu),
        # q_l1_layer = share of groups whose state really changed.
        tm = (idx != 0).float().unsqueeze(-1)
        den = tm.sum() * model.cfg.depth_groups
        paid = [(mm * tm).sum() / den for mm in aux["dd"]]
        real = [(r * tm).sum() / den for r in aux["rc"]]
        aux["q_l0_layer"], aux["q_l1_layer"] = paid, real
        cost = torch.stack(paid).mean()
        if model.cfg.qg_mu_per_block:                   # round 11: sum_l mu_l * paid_l / L
            return ce + (model.mu * torch.stack(paid)).mean(), ce, cost, paid, aux
        return ce + model.mu[0] * cost, ce, cost, paid, aux
    if model.cfg.quant:
        tm = (idx != 0).float().unsqueeze(-1)
        denom = tm.sum() * model.cfg.d
        l1 = [(n * tm).sum() / denom for n in aux["dd"]]
        l0 = [(n0 * tm).sum() / denom for n0 in aux["rc"]]
        aux["q_l1_layer"], aux["q_l0_layer"] = l1, l0
        per_layer = l1
        if model.cfg.q_cost == "l1":
            cost = torch.stack(l1).mean()
        elif model.cfg.q_cost == "l0":
            cost = torch.stack(l0).mean()
        loss = ce + model.mu[0] * cost if model.cfg.q_cost != "none" else ce
        return loss, ce, cost, per_layer, aux
    if model.cfg.budget:
        tm = (idx != 0).float().unsqueeze(-1)
        denom = tm.sum() * aux["dd"][0].shape[-1]
        for dd in aux["dd"]:
            c = (dd * tm).sum() / denom
            per_layer.append(c)
        if model.cfg.gate_type in ("hc", "ste"):
            # aux["rc"] holds P(m > 0) per block for K (K-STE: hard share with the gradient of mean sigmoid);
            # cost = sum_l mu_l * frac_l / L
            tm2 = tm[..., 0]
            frac_l = [(pr.mean(-1) * tm2).sum() / tm2.sum() for pr in aux["rc"]]
            aux["frac_layer"] = frac_l
            cost = (torch.stack(frac_l) * model.mu).sum() / len(frac_l)
        elif model.cfg.rel_cost:
            tm2 = tm[..., 0]
            rc_layer = [(rc * tm2).sum() / tm2.sum() for rc in aux["rc"]]
            aux["rc_layer"] = rc_layer
            cost = torch.stack(rc_layer).sum()
        else:
            cost = torch.stack(per_layer).sum()
    loss = ce + (cost if model.cfg.gate_type in ("hc", "ste") else lam * cost)
    return loss, ce, cost, per_layer, aux


@torch.no_grad()
def update_scale_ema(model, aux, decay=0.99):
    """I/J: EMA of the per-coordinate RMS of each block's input state (no gradient)."""
    if not aux.get("xin_ms"):
        return
    med = model.cfg.rel_scale == "median"
    for l, (blk, ms) in enumerate(zip(model.blocks, aux["xin_ms"])):
        r = ms.float().sqrt()
        if med:
            ax = aux["xin"][l][aux["tmask"]].abs()               # real tokens x d
            bmed, bmean = ax.median(0).values, ax.mean(0)
        if float(blk.s_init) == 0:
            blk.s_ema.copy_(r)
            if med:
                blk.s_med.copy_(bmed)
                blk.s_mabs.copy_(bmean)
            blk.s_init.fill_(1)
        else:
            blk.s_ema.mul_(decay).add_(r, alpha=1 - decay)
            if med:
                blk.s_med.mul_(decay).add_(bmed, alpha=1 - decay)
                blk.s_mabs.mul_(decay).add_(bmean, alpha=1 - decay)


@torch.no_grad()
def vq_update(model, aux, eps=1e-5):
    """round 14: EMA codebook update after the optimizer step.  Counts n_k and sums m_k of the vectors assigned to
    each codebook vector (shared: all depths together), N <- decay N + (1 - decay) n, M <- decay M + (1 - decay) m,
    E = M / N (Laplace-smoothed N).  First call: codebooks initialised from random batch vectors.  Vectors whose
    EMA count N < vq_dead are restarted from random batch vectors (N = 2 vq_dead).  delta: vector 0 stays 0.
    Returns per book: (vectors used this step, vectors restarted)."""
    cfg = model.cfg
    G, K, dec = cfg.depth_groups, cfg.vq_k, cfg.vq_decay
    gs = cfg.d // G
    nb = model.vq_E.shape[0]
    lo = 1 if cfg.vq == "delta" and not cfg.vq_norm else 0
    stats = []
    garange = torch.arange(G, device=model.vq_E.device)
    for b in range(nb):
        Z = torch.cat([z.reshape(-1, G, gs) for bk, z in zip(aux["vq_book"], aux["vq_z"]) if bk == b])     # N, G, gs
        I = torch.cat([i.reshape(-1, G) for bk, i in zip(aux["vq_book"], aux["vq_idx"]) if bk == b])      # N, G
        n_vec = Z.shape[0]
        if not bool(model.vq_ready):
            r = torch.stack([torch.randperm(n_vec, device=Z.device)[:K - lo] for _ in range(G)], 1)         # K-lo, G
            model.vq_E[b, :, lo:] = Z[r, garange[None, :]].transpose(0, 1)
            model.vq_N[b] = n_vec / K
            model.vq_M[b] = model.vq_E[b] * model.vq_N[b][..., None]
            stats.append((K, 0))
            continue
        flat = (I + garange[None, :] * K).reshape(-1)
        n = torch.bincount(flat, minlength=G * K).view(G, K).float()
        m = torch.zeros(G * K, gs, device=Z.device).index_add_(0, flat, Z.reshape(-1, gs)).view(G, K, gs)
        model.vq_N[b].mul_(dec).add_(n, alpha=1 - dec)
        model.vq_M[b].mul_(dec).add_(m, alpha=1 - dec)
        Nb = model.vq_N[b]
        tot = Nb.sum(-1, keepdim=True)
        Ns = (Nb + eps) / (tot + K * eps) * tot
        E = model.vq_M[b] / Ns[..., None]
        if cfg.vq_norm:
            E = F.normalize(E, dim=-1)
        dead = Nb < cfg.vq_dead
        dead[:, :lo] = False
        nd = int(dead.sum())
        if nd:
            cand = Z[torch.randint(0, n_vec, (G, K), device=Z.device), garange[:, None]]                    # G, K, gs
            E = torch.where(dead[..., None], cand, E)
            model.vq_N[b] = torch.where(dead, torch.full_like(Nb, 2 * cfg.vq_dead), Nb)
            model.vq_M[b] = torch.where(dead[..., None], cand * 2 * cfg.vq_dead, model.vq_M[b])
        if lo:
            E[:, 0] = 0.0
            model.vq_M[b, :, 0] = 0.0
        model.vq_E[b] = E
        stats.append((int((n > 0).sum()), nd))
    model.vq_ready.fill_(True)
    return stats


@torch.no_grad()
def spherical_kmeans(X, K, iters=20, seed=0, chunk=1 << 17):
    """round 17: k-means on unit vectors by cosine (X: N, s unit rows) -> unit centres (K, s), cluster sizes (K);
    k-means++ start (distance 1 - cos, on a random subsample of up to 64 K rows), empty clusters re-seeded"""
    g = torch.Generator(device=X.device).manual_seed(seed)
    N = X.shape[0]
    sub = X[torch.randperm(N, generator=g, device=X.device)[:min(N, 64 * K)]]
    C = torch.empty(K, X.shape[1], device=X.device, dtype=X.dtype)
    C[0] = sub[torch.randint(0, sub.shape[0], (1,), generator=g, device=X.device)]
    dmin = (1 - sub @ C[0]).clamp_min(0)
    for k in range(1, K):
        p = dmin.pow(2)
        j = torch.multinomial(p / p.sum(), 1, generator=g) if float(p.sum()) > 0 else             torch.randint(0, sub.shape[0], (1,), generator=g, device=X.device)
        C[k] = sub[j[0]]
        dmin = torch.minimum(dmin, (1 - sub @ C[k]).clamp_min(0))
    for _ in range(iters):
        idx = torch.cat([(X[i:i + chunk] @ C.t()).argmax(1) for i in range(0, N, chunk)])
        S = torch.zeros_like(C).index_add_(0, idx, X)
        cnt = torch.bincount(idx, minlength=K)
        C = F.normalize(S, dim=-1)
        empty = cnt == 0
        if bool(empty.any()):
            C[empty] = X[torch.randint(0, N, (int(empty.sum()),), generator=g, device=X.device)]
    idx = torch.cat([(X[i:i + chunk] @ C.t()).argmax(1) for i in range(0, N, chunk)])
    return C, torch.bincount(idx, minlength=K).float()


@torch.no_grad()
def vq_kmeans_init(model, Z_per_book, per_step, iters=20):
    """round 17: codebooks from k-means on the base model's directions (normalised VQ only).  Z_per_book[b]: (N, G, gs)
    unit vectors; per_step[b]: vectors per training step for that book (EMA counts are put on that scale)."""
    cfg = model.cfg
    assert cfg.vq_norm
    for b, Z in enumerate(Z_per_book):
        for g in range(cfg.depth_groups):
            C, cnt = spherical_kmeans(Z[:, g].contiguous(), cfg.vq_k, iters=iters, seed=1000 * b + g)
            model.vq_E[b, g] = C
            model.vq_N[b, g] = cnt / cnt.sum() * per_step[b]
            model.vq_M[b, g] = C * model.vq_N[b, g][:, None]
    model.vq_ready.fill_(True)
