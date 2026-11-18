"""Round 21, «пробелы в коде» (dd.r21_ws): annotation of Python lines, the position-wise switch-off, attention
weights, probe helpers."""
import os

import numpy as np
import pytest
import torch

from dd import r21_moe as R
from dd import r21_ws as W
from tests.test_r21 import ROOT, fake_books, idx_batch, tiny

SNIP = ("class A:\n    def f(self, x):\n        if x:\n            return 1\n\n        for i in range(3):\n"
        "            y = (1,\n                 2)\n        return 2\n")


def tok_and_bytes():
    from tokenizers import Tokenizer
    path = os.path.join(ROOT, "data_mix", "tokenizer.json")
    if not os.path.exists(path):
        pytest.skip("no GPT-2 tokenizer (built by dd.mix_data)")
    tok = Tokenizer.from_file(path)
    return tok, W.token_bytes(tok)


def test_token_bytes_match_decode():
    tok, tb = tok_and_bytes()
    ids = tok.encode(SNIP + "é ü → 中").ids
    assert b"".join(tb[i] for i in ids).decode("utf-8") == tok.decode(ids)


def test_annotate_doc_lines():
    tok, tb = tok_and_bytes()
    sp = W.space_ids(tb)
    ids = tok.encode(SNIP).ids
    r = W.annotate_doc(ids, tb, sp)
    s = [tok.decode([i]) for i in ids]
    assert r["err"] is None
    # the indent of 'return 1' (row 3): 11 single spaces + ' return'
    ret1 = [k for k in range(len(s)) if s[k] == " return" and r["row"][k] == 3][0]
    lead = [k for k in range(len(s)) if r["row"][k] == 3 and r["lead"][k]]
    assert len(lead) == 11 and lead[-1] == ret1 - 1 and all(s[k] == " " for k in lead)
    assert r["first"][ret1] and not r["first"][lead[0]]
    assert r["run_end"][ret1 - 1] and not r["run_end"][lead[0]]
    assert list(r["cnt"][lead]) == list(range(1, 12))
    assert r["depth"][lead[0]] == 3 and r["hdr"][lead[0]] == 3          # inside `if x:`
    assert r["line_ret"][lead[0]] == 1 and r["same_prev"][lead[0]] == 0 and r["same_next"][lead[0]] == 0
    # 'y = (1,' (row 6) inside `for`; the continuation row 7 is not a logical line
    row6 = [k for k in range(len(s)) if r["row"][k] == 6 and r["lead"][k]]
    assert r["hdr"][row6[0]] == 2 and r["depth"][row6[0]] == 3 and r["same_next"][row6[0]] == 0
    row7 = [k for k in range(len(s)) if r["row"][k] == 7 and r["lead"][k]]
    assert row7 and not r["logical"][row7[0]] and r["kind"][row7[0]] == 1 and r["kind"][row6[0]] == 0
    # the final `return 2` (row 8) is in `def f` (depth 2), same indent as the `for` line two logical lines up? no:
    # its previous logical line is `y = (1, 2)` (12 spaces) -> not the same
    row8 = [k for k in range(len(s)) if r["row"][k] == 8 and r["lead"][k]]
    assert r["hdr"][row8[0]] == 0 and r["depth"][row8[0]] == 2 and r["same_prev"][row8[0]] == 0
    assert r["same_next"][row8[0]] == -1
    # the def line (row 1) is in `class A`
    row1 = [k for k in range(len(s)) if r["row"][k] == 1 and r["lead"][k]]
    assert r["hdr"][row1[0]] == 1 and r["depth"][row1[0]] == 1
    # newline tokens are not spaces
    assert not sp[tok.encode("\n").ids[0]] and sp[tok.encode(" ").ids[0]]


def test_row_kinds():
    tok, tb = tok_and_bytes()
    src = 'def f():\n    """doc\n    more\n    """\n    # note\n    x = g(1,\n          2)\n    return x\n'
    ids = tok.encode(src).ids
    r = W.annotate_doc(ids, tb, W.space_ids(tb))
    kind = {int(r["row"][k]): int(r["kind"][k]) for k in range(len(ids)) if r["lead"][k]}
    assert kind == {1: 0, 2: 2, 3: 2, 4: 3, 5: 0, 6: 1, 7: 0}


def test_annotate_survives_a_tokenize_error():
    tok, tb = tok_and_bytes()
    ids = tok.encode("def f():\n    x = 1\n  y = 2\n    z = 3\n").ids
    r = W.annotate_doc(ids, tb, W.space_ids(tb))
    assert r["err"] is not None
    s = [tok.decode([i]) for i in ids]
    row1 = [k for k in range(len(s)) if r["row"][k] == 1 and r["lead"][k]]
    row3 = [k for k in range(len(s)) if r["row"][k] == 3 and r["lead"][k]]
    assert r["logical"][row1[0]] and not r["logical"][row3[0]] and r["kind"][row3[0]] == -1


def test_off_where_is_positionwise():
    base = tiny()
    m = R.MoE(base, "perlayer", 2, books=fake_books(base, 2)).eval()
    idx = idx_batch()
    with torch.no_grad():
        a, aux = m(idx)
        all_e = list(range(R.E))
        full = torch.ones_like(idx, dtype=torch.bool)
        none = torch.zeros_like(full)
        b, _ = m(idx, {"off_where": {0: (none, all_e)}})
        assert torch.equal(a, b)
        e = int(aux["eid"][0][1, 20])
        c, _ = m(idx, {"off": (0, e)})
        d, _ = m(idx, {"off_where": {0: (full, [e])}})
        assert torch.allclose(c, d)
        one = none.clone()
        one[1, 20] = True
        f, aux2 = m(idx, {"off_where": {0: (one, [e])}, "ffn_io": True, "x_in": True})
        assert torch.equal(f[:, :20], a[:, :20]) and torch.equal(f[0], a[0]) and torch.equal(f[2], a[2])
        assert not torch.allclose(f[1, 20], a[1, 20])
        assert len(aux2["x_in"]) == 4 and len(aux2["uf"]) == 2 and aux2["z"][0].shape == (3, 33, 64)


def test_attn_weights_reproduce_the_block():
    base = tiny()
    idx = idx_batch()
    blk = base.blocks[1]
    with torch.no_grad():
        x = base.emb(idx)
        rope = base.rope(idx.shape[1], idx.device)
        mask = R.doc_causal_mask(idx)
        A = W.attn_weights(blk, x, rope, mask)
        B, T, d = x.shape
        v = blk.qkv(blk.n1(x)).view(B, T, 3, blk.H, blk.hd).permute(2, 0, 3, 1, 4)[2]
        y = blk.o((A @ v).transpose(1, 2).reshape(B, T, d))
        ref = blk.attn(blk.n1(x), rope, mask=mask)
    assert torch.allclose(A.sum(-1), torch.ones_like(A.sum(-1)), atol=1e-5)
    assert torch.allclose(y, ref, atol=1e-5)
    assert float(A[0, :, 13, :12].abs().max()) == 0.0              # the document boundary at 12


def test_probe_helpers():
    cm = np.array([[[8, 2], [1, 1]]])
    assert np.isclose(W.balanced_acc(cm)[0], (0.8 + 0.5) / 2)
    g = torch.Generator().manual_seed(0)
    X = torch.randn(2000, 8, generator=g)
    y = (X[:, 0] + 0.1 * torch.randn(2000, generator=g) > 1.0).long()   # rare class, separable
    Wb = W.fit_logreg(X, y, 2, 1e-4)
    p = W.predict(X, Wb).numpy()
    ba = W.balanced_acc(W.doc_cm(np.zeros(2000, np.int64), y.numpy(), p, 2, 1))[0]
    assert ba > 0.9
    g_tr, y_tr = np.array([0, 0, 1, 1, 1]), np.array([0, 0, 1, 1, 0])
    assert list(W.lookup_pred(g_tr, y_tr, np.array([0, 1]), 2, 2)) == [0, 1]


def test_space_term_splits_the_ce():
    g = torch.Generator().manual_seed(0)
    lg = torch.randn(2, 6, 20, generator=g)
    idx = torch.randint(0, 20, (2, 6), generator=g)
    sp_ids = torch.tensor([3, 7])
    sp_next = torch.isin(idx[:, 1:], sp_ids)
    ce = W.ce_rows(lg, idx)
    t1 = W.space_term(lg, idx, sp_ids, sp_next)
    p = torch.softmax(lg[:, :-1], -1)
    psp = p[..., sp_ids].sum(-1)
    ref = -torch.where(sp_next, psp.log(), (1 - psp).log())
    assert torch.allclose(t1, ref, atol=1e-5) and bool((t1 <= ce + 1e-5).all())
