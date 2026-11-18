"""Round 21: experts without a trained gate (dd.r21_moe) and the mixed data (dd.mix_data)."""
import json
import os

import numpy as np
import pytest
import torch

from dd.model import Cfg, Model
from dd import r21_moe as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tiny(vocab=300, L=4):
    torch.manual_seed(0)
    cfg = Cfg(vocab=vocab, d=64, n_layers=L, n_heads=4, ffn_hidden=96, max_len=33, budget=False)
    return Model(cfg).eval()


def fake_books(base, N, seed=0):
    g = torch.Generator().manual_seed(seed)
    M, d, V = base.cfg.n_layers - N, base.cfg.d, base.cfg.vocab
    return dict(N=N, M=M, rvq=torch.randn(M, R.K, d, generator=g), rvq_sem=torch.randint(0, R.E, (M, R.K), generator=g),
                rvq_semw=torch.randint(0, R.E, (M, R.K), generator=g),
                rvq_rand=torch.randint(0, R.E, (M, R.K), generator=g), pl=torch.randn(M, R.K, d, generator=g),
                pl_sem=torch.randint(0, R.E, (M, R.K), generator=g), hash=torch.randint(0, R.E, (M, V), generator=g))


def idx_batch(vocab=300, B=3, T=33, seed=1):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(2, vocab, (B, T), generator=g)
    idx[:, 0] = 1
    idx[0, 12] = 1                                             # a document boundary inside a window
    return idx


@pytest.mark.parametrize("variant", list(R.ROUTE))
def test_moe_starts_as_the_dense_model(variant):
    """experts are copies of the A FFN (shared variant: the two halves), the learned gate is 1 in the forward:
    every variant computes exactly the A function at step 0"""
    base = tiny()
    idx = idx_batch()
    with torch.no_grad():
        ref, _ = base(idx)
        m = R.MoE(base, variant, 2, books=fake_books(base, 2)).eval()
        out, aux = m(idx, {"codes": True})
    assert torch.allclose(out, ref, atol=2e-5, rtol=1e-5), (out - ref).abs().max()
    assert aux["codes"].shape == (3, 33, 2)
    if R.ROUTE[variant]:
        assert len(aux["eid"]) == 2 and all(e.shape == (3, 33) for e in aux["eid"])


def test_routes_follow_the_tables():
    base = tiny()
    books = fake_books(base, 2)
    idx = idx_batch()
    with torch.no_grad():
        m = R.MoE(base, "rvq_sem", 2, books=books).eval()
        _, aux = m(idx)
        c = aux["codes"]
        for j in range(2):
            assert torch.equal(aux["eid"][j], books["rvq_sem"][j][c[..., j]])
        h = R.MoE(tiny(), "hash", 2, books=books).eval()
        _, ah = h(idx)
        for j in range(2):
            assert torch.equal(ah["eid"][j], books["hash"][j][idx])
        p = R.MoE(tiny(), "perlayer", 2, books=books).eval()
        _, ap = p(idx)
        for j in range(2):
            assert torch.equal(ap["eid"][j], books["pl_sem"][j][ap["pl_codes"][j]])


def test_rvq_encode_matches_brute_force_and_penalty():
    g = torch.Generator().manual_seed(0)
    u = torch.randn(200, 16, generator=g)
    books = torch.randn(3, 10, 16, generator=g)
    codes, rec = R.rvq_encode(u, books)
    r = u.clone()
    for j in range(3):
        d = ((r[:, None] - books[j][None]) ** 2).sum(-1)
        c = d.argmin(1)
        assert torch.equal(codes[:, j], c)
        r = r - books[j][c]
    assert torch.allclose(rec, u - r, atol=1e-5)
    base = tiny()
    m = R.MoE(base, "rvq_sem", 2, books=fake_books(base, 2))
    _, aux = m(idx_batch())
    assert aux["pen"].requires_grad and aux["pen"] > 0


def test_dispatch_gradient_equals_dense_gradient():
    """identical experts, one per token: the sum over experts of the expert gradients = the dense FFN gradient"""
    base = tiny()
    idx = idx_batch()
    tgt = torch.randint(2, 300, idx.shape)
    lg, _ = base(idx)
    torch.nn.functional.cross_entropy(lg.flatten(0, 1), tgt.flatten()).backward()
    gd = base.blocks[3].w1.weight.grad.clone()
    base.zero_grad()
    m = R.MoE(base, "hash", 2, books=fake_books(base, 2))
    lg2, _ = m(idx)
    torch.nn.functional.cross_entropy(lg2.flatten(0, 1), tgt.flatten()).backward()
    assert torch.allclose(m.experts[1].w1.grad.sum(0), gd, atol=1e-6)


def test_ablation_zeroes_one_expert():
    base = tiny()
    books = fake_books(base, 2)
    m = R.MoE(base, "rvq_sem", 2, books=books).eval()
    idx = idx_batch()
    with torch.no_grad():
        a, aux = m(idx)
        e = int(aux["eid"][0].flatten()[0])
        b, _ = m(idx, {"off": (0, e)})
        unused = [k for k in range(R.E) if not (aux["eid"][0] == k).any()]
    assert not torch.allclose(a, b)
    if unused:
        with torch.no_grad():
            c, _ = m(idx, {"off": (0, unused[0])})
        assert torch.allclose(a, c)


def test_learned_router_gets_a_gradient():
    base = tiny()
    m = R.MoE(base, "learned", 2, books=fake_books(base, 2))
    loss, ce, aux = R.moe_loss(m, idx_batch(), alpha=0.01)
    loss.backward()
    assert all(r.weight.grad is not None and r.weight.grad.abs().sum() > 0 for r in m.router)
    assert 0.9 < float(aux["bal"]) / 2 < 2.5                 # E * sum f P per block ~ 1 near balance


def test_save_load_roundtrip(tmp_path):
    base = tiny()
    m = R.MoE(base, "rvq_shared", 2, books=fake_books(base, 2)).eval()
    torch.save(dict(cfg=m.base.cfg.to_dict(), variant="rvq_shared", N=2, state=m.state_dict()), tmp_path / "model.pt")
    m2 = R.load_moe(str(tmp_path), dev="cpu")
    idx = idx_batch()
    with torch.no_grad():
        assert torch.allclose(m(idx)[0], m2(idx)[0])


def test_nmi():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 5, 10000)
    assert R.nmi(a, a) == pytest.approx(1.0)
    assert R.nmi(a, (a + 3) % 5) == pytest.approx(1.0)
    assert R.nmi(a, rng.integers(0, 5, 10000)) < 0.01
    # hand example: x = [0,0,1,1], y = [0,0,0,1]
    x, y = np.array([0, 0, 1, 1]), np.array([0, 0, 0, 1])
    p = np.array([[.5, 0], [.25, .25]])
    pa, pb = p.sum(1), p.sum(0)
    mi = sum(p[i, j] * np.log(p[i, j] / pa[i] / pb[j]) for i in range(2) for j in range(2) if p[i, j] > 0)
    h = lambda q: -(q[q > 0] * np.log(q[q > 0])).sum()          # noqa: E731
    assert R.nmi(x, y) == pytest.approx(2 * mi / (h(pa) + h(pb)))


def test_balanced_tables():
    freq = 1.0 / np.arange(1, 50001) ** 1.1                  # Zipf
    t = R.hash_table(freq, seed=0)
    load = np.bincount(t, weights=freq, minlength=R.E) / freq.sum()
    assert load.max() * R.E - 1 < 0.6 and len(np.unique(t)) == R.E
    ft = np.random.default_rng(1).gamma(2.0, size=64)
    tab = R.random_balanced_table(ft, tries=500, seed=0)
    ld = np.bincount(tab, weights=ft, minlength=R.E) / ft.sum()
    assert ld.max() * R.E - 1 < 0.1 and len(np.unique(tab)) == R.E
    assert not np.array_equal(tab, R.random_balanced_table(ft, tries=500, seed=1))


def test_kmeans_and_superclusters():
    g = torch.Generator().manual_seed(0)
    centres = torch.randn(4, 8, generator=g) * 10
    X = (centres[torch.arange(4).repeat_interleave(250)] + torch.randn(1000, 8, generator=g))
    C, cnt, inert = R.kmeans(X, 4, seed=0)
    assert sorted(cnt.tolist()) == [250.0] * 4 and inert < 12
    pts = torch.cat([centres[i] + 0.1 * torch.randn(16, 8, generator=g) for i in range(4)])
    lab = R.superclusters(pts, n=4, restarts=5)
    assert all(len(set(lab[16 * i:16 * (i + 1)].tolist())) == 1 for i in range(4))
    assert len(set(lab.tolist())) == 4


def test_lru2():
    e = np.array([[0, 1, 0, 2, 1, 1]])
    # 0 miss, 1 miss, 0 hit, 2 miss (evicts 1), 1 miss, 1 hit
    assert R.lru2_hits(e) == pytest.approx(2 / 6)


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data_wiki", "tokenizer.json")), reason="no GPT-2 tokenizer")
def test_token_types():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(ROOT, "data_wiki", "tokenizer.json"))
    P = R.vocab_props(os.path.join(ROOT, "data_wiki", "tokenizer.json"))
    ids = [1] + tok.encode("The dog met Alice in 1999, zxqvblorptic.\n").ids
    prev, cur = np.array([1] + ids[:-1]), np.array(ids)
    t = [R.TYPES[k] for k in R.token_types(P, prev, cur)]
    s = [tok.decode([i]) if i > 1 else "<eot>" for i in ids]
    d = dict(zip(s, t))
    assert d["The"] == "служебные" and d[" dog"] == "начало слова" and d[" Alice"] == "имена"
    assert d[" 1999"] == "числа" and d[","] == "пунктуация" and d["\n"] == "пробелы" and d[" in"] == "служебные"
    assert any(x == "продолжение слова" for x in t)          # " zxqvblorptic" is cut into pieces


@pytest.mark.skipif(not all(os.path.exists(os.path.join(ROOT, p)) for p in
                            ("data_wiki/val.bin", "data_wiki/tokenizer.json", "data_ts/raw/TinyStories-valid.txt")),
                    reason="local test sources missing")
def test_mix_data_layout(tmp_path):
    from dd import mix_data
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        out = str(tmp_path / "mix")
        mix_data.main(out, 20000, 3000, test=True)
    finally:
        os.chdir(cwd)
    meta = json.load(open(os.path.join(out, "meta.json")))
    tr = np.fromfile(os.path.join(out, "train.bin"), dtype=np.uint16)
    va = np.fromfile(os.path.join(out, "val.bin"), dtype=np.uint16)
    vs = np.fromfile(os.path.join(out, "val_src.bin"), dtype=np.uint8)
    assert meta["train_bounds"] == [0, 20000, 40000, 60000] and meta["val_bounds"] == [0, 3000, 6000, 9000]
    assert tr.size == 60000 and va.size == 9000 and vs.size == 9000
    for s in range(3):                                        # every source starts with a document start
        assert tr[meta["train_bounds"][s]] == 1 and va[meta["val_bounds"][s]] == 1
        assert (vs[meta["val_bounds"][s]:meta["val_bounds"][s + 1]] == s).all()
    assert (tr != 0).all() and (va != 0).all()
    assert np.array_equal(mix_data.source_of(meta, "val", np.arange(9000)), vs)
    freq = np.load(os.path.join(out, "freq.npy"))
    assert freq.sum() == tr.size


def test_mix_sources_parse_hf_formats(tmp_path, monkeypatch):
    """wiki_docs / code_docs / ts_docs on small files in the HF formats (parquet "text", json.gz "content", txt)"""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:                                       # not in the local venv; installed on the machines
        pa = pq = None
    import gzip
    import huggingface_hub
    from dd import mix_data
    made = []

    def fake_download(repo, fn, repo_type=None, local_dir=None):
        path = os.path.join(local_dir, fn.replace("/", "_"))
        os.makedirs(local_dir, exist_ok=True)
        if fn.endswith(".parquet"):
            shard = fn.split("train-")[1][1:5]                # "train-00000-of-00041.parquet" -> "0000"
            pq.write_table(pa.table({"text": [f"Article {shard} {i}." for i in range(3)] + ["  "]}), path)
        elif fn.endswith(".json.gz"):
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                for i in range(3):
                    fh.write(json.dumps({"repo_name": "r", "content": f"def f{i}():\n    return {i}\n"}) + "\n")
        else:
            open(path, "w", encoding="utf-8").write("Once upon a time.\n<|endoftext|>\nThe end.\n<|endoftext|>\n")
        made.append(fn)
        return path
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.chdir(tmp_path)                               # no data_ts/raw here: ts_docs downloads
    if pa is not None:
        w = mix_data.wiki_docs(str(tmp_path / "raw"))
        got = [next(w) for _ in range(5)]
        assert got[:3] == ["Article 0000 0.", "Article 0000 1.", "Article 0000 2."] and got[3] == "Article 0001 0."
        assert made[0] == "20231101.en/train-00000-of-00041.parquet"
    c = mix_data.code_docs(str(tmp_path / "raw"))
    assert [next(c) for _ in range(4)][3] == "def f0():\n    return 0\n"
    assert list(mix_data.ts_docs(str(tmp_path / "raw"))) == ["Once upon a time.", "The end."]
    assert "file-000000000001.json.gz" in made
    assert "file-000000000002.json.gz" in made and "TinyStories-train.txt" in made


def test_mix_encoder_ignores_special_strings_in_text():
    from tokenizers import Tokenizer
    from dd import mix_data
    path = os.path.join(ROOT, "data_wiki", "tokenizer.json")
    if not os.path.exists(path):
        pytest.skip("no GPT-2 tokenizer")
    tok = Tokenizer.from_file(path)
    enc = mix_data.plain_encoder(tok)
    s = 'pad = "<pad>"\neos = "<|endoftext|>"\nThe Battle of Hastings, 1066.'
    ids = enc.encode(s).ids
    assert 0 not in ids and 1 not in ids and tok.decode(ids) == s
    plain = "The Battle of Hastings was fought on 14 October 1066."
    assert enc.encode(plain).ids == tok.encode(plain).ids


def test_weighted_superclusters_follow_the_load():
    """a dense core with most tokens and a few far, rare centres: unweighted superclusters isolate the rare
    centres (the core = one expert); weighted ones split the core"""
    g = torch.Generator().manual_seed(0)
    core = torch.randn(56, 8, generator=g)
    far = torch.randn(8, 8, generator=g) * 0.3 + 12 * F_unit(8, g)
    C = torch.cat([core, far])
    w = np.r_[np.full(56, 0.99 / 56), np.full(8, 0.01 / 8)]
    lab_u = R.superclusters(C, restarts=10).numpy()
    lab_w = R.weighted_superclusters(C.numpy(), w, restarts=10).numpy()
    load_u = np.bincount(lab_u, weights=w, minlength=R.E)
    load_w = np.bincount(lab_w, weights=w, minlength=R.E)
    assert load_u.max() > 0.9 and load_w.max() < 0.5 and len(np.unique(lab_w)) == R.E


def F_unit(n, g):
    v = torch.randn(n, n, generator=g)
    return v / v.norm(dim=1, keepdim=True)


def test_rvq_residual_diagnostics_and_drift_record():
    g = torch.Generator().manual_seed(0)
    u = torch.randn(300, 16, generator=g)
    books = torch.randn(3, 10, 16, generator=g)
    codes, _ = R.rvq_encode(u, books)
    rn, mg = R.rvq_residuals(u, codes, books)
    r = u.clone()
    for j in range(3):
        d = ((r[:, None] - books[j][None]) ** 2).sum(-1).sqrt()
        assert torch.allclose(rn[:, j], r.norm(dim=1), atol=1e-5)
        srt = d.sort(1).values
        assert torch.allclose(mg[:, j], srt[:, 1] - srt[:, 0], atol=1e-4)     # the chosen code is the nearest
        r = r - books[j][codes[:, j]]
    assert torch.allclose(rn[:, 3], r.norm(dim=1), atol=1e-5)
    sp = R.book_spacing(books)
    C = books[0]
    assert abs(sp[0] - float(torch.pdist(C).mean())) < 1e-4

    class FakeD:
        dev = torch.device("cpu")

        def win(self, offs):
            return idx_batch(B=offs.numel(), seed=int(offs[0]) + 5)

    base = tiny()
    m = R.MoE(base, "rvq_sem", 2, books=fake_books(base, 2))
    s = R.snapshot(m, FakeD(), torch.arange(4), bs=2)
    rec = R.drift_rec(1000, s, s, s)
    M = m.M
    assert len(rec["resid_norm"]) == M + 1 and len(rec["margin"]) == M and len(rec["code_entropy_bits"]) == M
    assert abs(rec["resid_norm"][0] - 64 ** 0.5) < 1e-2                     # stage 1 input = RMS-normalised state
    assert all(x >= 0 for x in rec["margin"]) and rec["codes_per_vsprev"] == [0.0] * M
    assert "rn_any_vs0" not in rec and rec["codes_any_vs0"] == 0.0
