"""Round 21, «пробелы в коде»: the experts of MoE-perlayer that take the space / tab tokens of Python (home, no
training, Python val only: 1.5M tokens, 239 files, the 2929 val windows lying wholly in the Python part).

In the GPT-2 BPE every space of an indent is its own token ' ' (the last one is glued to the next word:
'\\n', ' ', ' ', ' ', ' def'), so 36% of the Python val tokens are ' '. «Пробелы» here = tokens made only of spaces /
tabs (' ', '\\t'); newline tokens are another expert's and are not counted. The experts «пробелы в коде» of a block:
those whose val tokens are >= 50% Python space tokens (from the r21_viz routes cache).

  python -m dd.r21_ws annot    # every val Python file through Python's tokenize: logical lines, nesting depth, the
                               #   enclosing construct, indents of the previous / next logical line, first word;
                               #   per token -> runs_home/r21_ws/annot.npz, sets.json (the experts per block)
  python -m dd.r21_ws ablate   # 1: the experts switched off (output 0) ONLY at space positions, block 5 / 6 / 7 / 8 /
                               #   all four; CE rise (a) at those positions, (b) on the first token of their line
                               #   and further along the line, (c) elsewhere -> ablate.json, ablate_win.npz
  python -m dd.r21_ws attn     # 2: share of attention the non-space tokens give to space positions, every block and
                               #   head, against the share of spaces among the keys they see -> attn.json
  python -m dd.r21_ws probe    # 3: linear probes at the indent spaces the experts take: on the expert input z, its
                               #   output, and the stream after the block -> probe.json
  python -m dd.r21_ws report   # -> report/r21_ws.md, report/r21_ws.json (the card on the experts page)

Spread: perlayer has one seed, so the bar is measurement noise, not seed spread: bootstrap over the val windows
(1, 2) or over the Python files (3); a difference counts only when it is > 2 SE."""
import argparse
import io
import json
import os
import time
import tokenize

import numpy as np
import torch
import torch.nn.functional as F

from .model import apply_rope, doc_causal_mask
from .r21_moe import CTX, E, Data, load_moe

RUN = os.path.join("runs_vast", "mix_perlayer")
OUT = os.path.join("runs_home", "r21_ws")
ROUTES = os.path.join("runs_home", "r21_viz", "routes.npz")
MD = os.path.join("report", "r21_ws.md")
JS = os.path.join("report", "r21_ws.json")
CODE = 1
BS = 16
HDR = ["def", "class", "for", "if", "while", "другое"]
TARGETS = [  # name, classes, Russian name; all but `kind` only at logical line starts
    ("kind", 4, "тип строки (логическая / продолжение в скобках / внутри многострочной строки / комментарий)"),
    ("depth", 5, "глубина вложенности строки (1 … ≥5)"),
    ("cnt", 16, "сколько пробелов отступа уже набрано (1 … ≥16)"),
    ("hdr", 6, "охватывающая конструкция (def / class / for / if / while / другое)"),
    ("same_prev", 2, "отступ этой строки = отступу предыдущей логической строки"),
    ("same_next", 2, "отступ следующей логической строки = отступу этой"),
    ("line_ret", 2, "строка начнётся с return"),
    ("run_end", 2, "отступ кончается здесь (следующий токен — первое слово строки)"),
    ("next_ret", 2, "следующий токен — return"),
]
REPS = [("z", "вход эксперта (z = RMSNorm потока)"), ("uf", "выход эксперта"), ("x", "поток после блока")]
RUNS = [[0], [1], [2], [3], [0, 1, 2, 3]]
BUCKETS = ["a_last", "a_mid", "a_inline", "b_first", "b_rest", "c"]
BUCKET_RU = {"a_last": "(а) последний пробел отступа", "a_mid": "(а) пробел внутри отступа",
             "a_inline": "(а) пробел внутри строки", "b_first": "(б) первый токен строки",
             "b_rest": "(б) дальше по строке", "c": "остальные позиции кода"}


# ------------------------------------------------------------------------------------------------ annotation
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def token_bytes(tok):
    inv = {c: b for b, c in bytes_to_unicode().items()}
    out = [b""] * tok.get_vocab_size()
    for i in range(2, len(out)):
        out[i] = bytes(inv[ch] for ch in tok.id_to_token(i))
    return out


def space_ids(tb):
    """tokens made only of spaces / tabs"""
    return np.array([bool(b) and set(b) <= {32, 9} for b in tb])


def header_class(first, second):
    if first == "async":
        first = second
    return {"def": 0, "class": 1, "for": 2, "if": 3, "elif": 3, "while": 4}.get(first, 5)


def logical_lines(text, extra=False):
    """Python tokenize of a file: [(row0, col, depth, first, second)] for every logical line start; stops quietly at a
    tokenize error (the rows after it get no logical lines) -> (lines, error row or None); extra=True adds the rows
    that hold only a comment and the rows inside a multi-line token (docstrings, multi-line strings)"""
    out, depth, at_start, nth = [], 0, True, 0
    com, inside = [], []
    try:
        for tk in tokenize.generate_tokens(io.StringIO(text).readline):
            ty = tk.type
            if tk.end[0] > tk.start[0] and ty not in (tokenize.NEWLINE, tokenize.NL):
                inside.extend(range(tk.start[0], tk.end[0]))          # 0-based rows start+1 .. end
            if ty == tokenize.COMMENT and at_start:
                com.append(tk.start[0] - 1)
            if ty == tokenize.INDENT:
                depth += 1
            elif ty == tokenize.DEDENT:
                depth -= 1
            elif ty == tokenize.NEWLINE:
                at_start = True
            elif ty in (tokenize.NL, tokenize.COMMENT, tokenize.ENDMARKER, tokenize.ENCODING):
                pass
            elif at_start:
                out.append([tk.start[0] - 1, tk.start[1], depth, tk.string, ""])
                at_start, nth = False, 1
            elif nth == 1:
                out[-1][4] = tk.string
                nth = 2
        res = (out, None)
    except Exception:                                        # noqa: BLE001  (IndentationError, TokenError, ...)
        res = (out, (out[-1][0] if out else 0))
    return res + (com, inside) if extra else res


def annotate_doc(ids, tb, sp):
    """per token of one file (no eot inside): row, lead (a space token inside the leading indent of its row), first
    (the token holding the row's first non-blank byte), cnt (indent bytes up to the token end), run_end (lead and the
    next token is the row's first token), and the row targets (-1 = unknown / not a logical line start)"""
    n = len(ids)
    bl = [tb[i] for i in ids]
    ln = np.array([len(b) for b in bl], np.int64)
    end = np.cumsum(ln)
    start = end - ln
    text_b = b"".join(bl)
    arr = np.frombuffer(text_b, np.uint8)
    nl = np.flatnonzero(arr == 10)
    row = np.searchsorted(nl, start, side="left")              # newlines strictly before the token start
    row_start = np.concatenate([[0], nl + 1])
    nrows = row_start.size
    row_end = np.concatenate([nl, [arr.size]])
    fw = np.empty(nrows, np.int64)                               # first non-blank byte of the row (= row_end if blank)
    for r in range(nrows):
        k = row_start[r]
        while k < row_end[r] and arr[k] in (32, 9, 12):
            k += 1
        fw[r] = k
    is_sp = sp[np.asarray(ids)]
    lead = is_sp & (end <= fw[row])
    blank = fw[row] >= row_end[row]
    first = (~blank) & (start <= fw[row]) & (fw[row] < end)
    cnt = np.where(lead, end - row_start[row], 0)
    nxt_first = np.zeros(n, bool)
    nxt_first[:-1] = first[1:] & (row[1:] == row[:-1])
    run_end = lead & nxt_first
    lines, err, com, inside = logical_lines(text_b.decode("utf-8", errors="replace"), extra=True)
    # row targets
    R = {k: np.full(nrows, -1, np.int64) for k in ("depth", "hdr", "same_prev", "same_next", "line_ret", "col", "kind")}
    # row kind: 0 logical line, 1 continuation (in brackets / after a backslash), 2 inside a multi-line string, 3 comment
    known = np.arange(nrows) < (err if err is not None else nrows)
    R["kind"][known & (fw < row_end)] = 1
    R["kind"][[r for r in inside if r < nrows]] = 2
    R["kind"][com] = 3
    R["kind"][[x[0] for x in lines]] = 0
    last_at = {}
    for k, (r, c, d, f, s) in enumerate(lines):
        R["depth"][r], R["col"][r] = d, c
        R["hdr"][r] = header_class(*lines[last_at[d - 1]][3:5]) if d >= 1 and (d - 1) in last_at else -1
        last_at[d] = k
        for dd in [x for x in last_at if x > d]:
            del last_at[dd]
        R["same_prev"][r] = int(c == lines[k - 1][1]) if k > 0 else -1
        R["same_next"][r] = int(lines[k + 1][1] == c) if k + 1 < len(lines) else -1
        R["line_ret"][r] = int(f == "return")
    out = dict(row=row, lead=lead, first=first, cnt=cnt, run_end=run_end, sp=is_sp,
               nlc=np.array([(10 in b) or (13 in b) for b in bl]),
               logical=R["depth"][row] >= 0)
    for k in ("depth", "hdr", "same_prev", "same_next", "line_ret", "kind"):
        out[k] = R[k][row]
    out["n_logical"], out["nrows"], out["err"] = len(lines), nrows, err
    return out


def annot_cmd():
    from .r21_viz import tokenizer
    tok = tokenizer()
    tb = token_bytes(tok)
    sp = space_ids(tb)
    meta = json.load(open(os.path.join("data_mix", "meta.json")))
    lo, hi = meta["val_bounds"][CODE], meta["val_bounds"][CODE + 1]
    ids_all = np.fromfile(os.path.join("data_mix", "val.bin"), dtype=np.uint16).astype(np.int64)
    ids = ids_all[lo:hi]
    n = ids.size
    ret = np.array([b.strip() == b"return" for b in tb])
    A = {k: np.zeros(n, bool) for k in ("sp", "lead", "first", "run_end", "nlc", "logical", "next_ret")}
    A.update({k: np.full(n, -1, np.int16) for k in ("depth", "hdr", "same_prev", "same_next", "line_ret", "cnt",
                                                       "kind")})
    A["line"] = np.full(n, -1, np.int64)
    A["doc"] = np.full(n, -1, np.int32)
    eot = np.flatnonzero(ids == 1)
    bounds = np.concatenate([[-1], eot, [n]])
    row_base, nd, bad, t0 = 0, 0, 0, time.time()
    for a, b in zip(bounds[:-1] + 1, bounds[1:]):
        if b - a < 1:
            continue
        r = annotate_doc(ids[a:b].tolist(), tb, sp)
        s = slice(a, b)
        for k in ("sp", "lead", "first", "run_end", "nlc", "logical"):
            A[k][s] = r[k]
        for k in ("depth", "hdr", "same_prev", "same_next", "line_ret", "cnt", "kind"):
            A[k][s] = r[k]
        A["line"][s] = r["row"] + row_base
        A["doc"][s] = nd
        row_base += r["nrows"]
        nd += 1
        bad += r["err"] is not None
    A["next_ret"][:-1] = ret[ids[1:]]
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "annot.npz"), lo=lo, **A)
    lead = A["lead"]
    print(f"{nd} files ({bad} with a tokenize error: rows after it have no logical lines), {n} tokens, "
          f"{row_base} rows ({time.time() - t0:.0f}s)", flush=True)
    kinds = np.bincount(A["kind"][lead] + 1, minlength=5) / lead.sum()
    print(f"space tokens {A['sp'].mean():.3f} of Python tokens; in the leading indent {lead.sum() / A['sp'].sum():.3f}; "
          f"rows of the indent spaces: unknown {kinds[0]:.3f}, logical {kinds[1]:.3f}, continuation {kinds[2]:.3f}, "
          f"in a string {kinds[3]:.3f}, comment {kinds[4]:.3f}", flush=True)
    # experts per block from the r21_viz routes cache
    z = np.load(ROUTES)
    pos, codes = z["pos"], z["codes"]
    model_table = torch.load(os.path.join(RUN, "model_bf16.pt"), map_location="cpu", weights_only=False)["state"]
    table = model_table["pl_table"].numpy()
    src_all = np.fromfile(os.path.join("data_mix", "val_src.bin"), dtype=np.uint8)
    is_sp_all = sp[ids_all[pos]] & (src_all[pos] == CODE)
    sets = []
    for j in range(table.shape[0]):
        ex = table[j][codes[:, j]]
        share = [float(is_sp_all[ex == e].mean()) if (ex == e).any() else 0.0 for e in range(E)]
        S = [e for e in range(E) if share[e] >= 0.5]
        cover = float(np.isin(ex[is_sp_all], S).mean())
        sets.append(dict(j=j, experts=S, share_sp=share, cover=cover))
        print(f"block {5 + j}: experts {S} (space share " + ", ".join(f"E{e} {share[e]:.2f}" for e in S) +
              f"); they take {cover:.3f} of the Python space tokens", flush=True)
    json.dump(dict(sets=sets, files=nd, files_tokenize_error=bad, tokens=int(n), rows=int(row_base),
                   sp_share=float(A["sp"].mean()), lead_of_sp=float(lead.sum() / A["sp"].sum()),
                   logical_of_lead=float((lead & A["logical"]).sum() / lead.sum()),
                   kind_of_lead=kinds[1:].tolist(), kind_unknown=float(kinds[0])),
              open(os.path.join(OUT, "sets.json"), "w"), indent=1)


def load_annot(dev, n_total):
    """annotation arrays over the whole val stream (zeros / -1 outside the Python part), on dev"""
    z = np.load(os.path.join(OUT, "annot.npz"))
    lo = int(z["lo"])
    out = {}
    for k in z.files:
        if k == "lo":
            continue
        a = z[k]
        full = np.full(n_total, -1 if a.dtype.kind == "i" else 0, a.dtype)
        full[lo:lo + a.size] = a
        out[k] = torch.from_numpy(full.astype(np.int64) if a.dtype.kind == "i" else full).to(dev)
    return out, lo, lo + z["sp"].size


def model_and_data():
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    model = load_moe(RUN, None, dev, fname="model_bf16.pt")
    assert model.variant == "perlayer"
    D = Data("data_mix", dev)
    return model, D, dev


def code_windows(D, lo, hi):
    s = D.starts()
    return s[(s >= lo) & (s + CTX + 1 <= hi)]


def ws_sets(dev):
    sets = json.load(open(os.path.join(OUT, "sets.json")))["sets"]
    return [s["experts"] for s in sets]


def space_term(logits, idx, sp_ids, sp_next):
    """the part of the CE that only says «space or not» for the next token: -log P(space) if the next token is a space,
    else -log(1 - P(space)); the rest of the CE is the choice inside that class"""
    lg = logits[:, :-1].float()
    lsp = torch.logsumexp(lg[..., sp_ids], -1) - torch.logsumexp(lg, -1)
    lnot = torch.log1p(-torch.exp(lsp).clamp_max(1 - 1e-7))
    return -torch.where(sp_next, lsp, lnot)


def ce_rows(logits, idx):
    tgt = idx[:, 1:]
    return F.cross_entropy(logits[:, :-1].float().flatten(0, 1), tgt.flatten(), reduction="none").view(tgt.shape)


# ------------------------------------------------------------------------------------------------ 1. ablation
def ablated_positions(eids, sp_tok, sets, blocks):
    """positions where a «spaces» expert of one of `blocks` was really switched off (live routing)"""
    a = torch.zeros_like(sp_tok)
    for j in blocks:
        a |= sp_tok & torch.isin(eids[j], torch.tensor(sets[j], device=sp_tok.device))
    return a


def buckets(A, an, P, T):
    """bucket masks over the T scored positions of every window: (а) ablated positions split into last indent space /
    indent space / in-line space, (б) first token and the rest of the line of a line whose indent has an ablated
    position, (c) every other Python position"""
    pos = P[:, :T]
    a = A[:, :T]
    lead, run_end = an["lead"][pos], an["run_end"][pos]
    line = an["line"][P]
    al = A & an["lead"][P]                                      # ablated indent spaces (all T+1 positions)
    same = (line[:, :T, None] == line[:, None, :]) & (line[:, :T, None] >= 0)
    in_l = (same & al[:, None, :]).any(-1)
    nonws = ~an["sp"][pos] & ~an["nlc"][pos] & (line[:, :T] >= 0)
    first = an["first"][pos]
    m = dict(a_last=a & lead & run_end, a_mid=a & lead & ~run_end, a_inline=a & ~lead,
             b_first=~a & in_l & first & nonws, b_rest=~a & in_l & ~first & nonws)
    rest = torch.ones_like(a)
    for v in m.values():
        rest &= ~v
    m["c"] = rest & (line[:, :T] >= 0)
    return m


@torch.no_grad()
def ablate_cmd():
    model, D, dev = model_and_data()
    an, lo, hi = load_annot(dev, D.val.numel())
    sets = ws_sets(dev)
    offs = code_windows(D, lo, hi)
    nw, M = offs.numel(), model.M
    nb = len(BUCKETS)
    S = np.zeros((len(RUNS), nw, nb))       # sum of the CE rise per window and bucket
    C = np.zeros((len(RUNS), nw, nb))       # counts
    B0 = np.zeros((len(RUNS), nw, nb))      # base CE sums
    CH = np.zeros((len(RUNS), nw))          # positions whose expert changed in a later block
    S1 = np.zeros((len(RUNS), nw, nb))      # rise of the «space or not» part of the CE
    from .r21_viz import tokenizer
    sp_ids = torch.from_numpy(np.flatnonzero(space_ids(token_bytes(tokenizer())))).to(dev)
    ar = torch.arange(CTX + 1, device=dev)
    t0 = time.time()
    for i in range(0, nw, BS):
        o = offs[i:i + BS]
        idx = D.win(o)
        P = o[:, None] + ar[None]
        sp_tok = an["sp"][P]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, aux0 = model(idx)
        ce0 = ce_rows(lg, idx)
        sp_next = sp_tok[:, 1:]
        t0_ = space_term(lg, idx, sp_ids, sp_next)
        for r, blocks in enumerate(RUNS):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg1, aux1 = model(idx, {"off_where": {j: (sp_tok, sets[j]) for j in blocks}})
            d = ce_rows(lg1, idx) - ce0
            d1 = space_term(lg1, idx, sp_ids, sp_next) - t0_
            A = ablated_positions(aux1["eid"], sp_tok, sets, blocks)
            m = buckets(A, an, P, CTX)
            for k, name in enumerate(BUCKETS):
                mk = m[name].double()
                S[r, i:i + BS, k] = (d.double() * mk).sum(1).cpu().numpy()
                S1[r, i:i + BS, k] = (d1.double() * mk).sum(1).cpu().numpy()
                B0[r, i:i + BS, k] = (ce0.double() * mk).sum(1).cpu().numpy()
                C[r, i:i + BS, k] = mk.sum(1).cpu().numpy()
            later = [j for j in range(M) if j > min(blocks)]
            if later:
                ch = torch.zeros_like(sp_tok)
                for j in later:
                    ch |= aux1["eid"][j] != aux0["eid"][j]
                CH[r, i:i + BS] = ch[:, :CTX].sum(1).cpu().numpy()
        if i % (BS * 40) == 0:
            print(f"{i + o.numel()}/{nw} windows ({time.time() - t0:.0f}s)", flush=True)
    np.savez(os.path.join(OUT, "ablate_win.npz"), S=S, C=C, B0=B0, CH=CH, S1=S1)
    res = dict(windows=int(nw), tokens=int(nw * CTX), sets=sets, runs=[])
    rng = np.random.default_rng(0)
    boot = rng.integers(0, nw, (1000, nw))
    for r, blocks in enumerate(RUNS):
        rr = dict(blocks=[5 + j for j in blocks], buckets={}, routes_changed=float(CH[r].sum() / (nw * CTX)))
        for k, name in enumerate(BUCKETS):
            s, c, b0 = S[r, :, k], C[r, :, k], B0[r, :, k]
            per = s.sum() / max(c.sum(), 1)
            bs_per = s[boot].sum(1) / np.maximum(c[boot].sum(1), 1)
            tot = s.sum() / (nw * CTX)
            bs_tot = s[boot].sum(1) / (nw * CTX)
            s1 = S1[r, :, k]
            rr["buckets"][name] = dict(n=int(c.sum()), share=float(c.sum() / (nw * CTX)), per_token=float(per),
                                       per_token_se=float(bs_per.std()), total=float(tot), total_se=float(bs_tot.std()),
                                       base_ce=float(b0.sum() / max(c.sum(), 1)),
                                       space_part=float(s1.sum() / max(c.sum(), 1)),
                                       space_part_se=float((s1[boot].sum(1) / np.maximum(c[boot].sum(1), 1)).std()))
        # (а) and (б) as wholes, and the paired difference (б) - (а) per token
        ka = [BUCKETS.index(x) for x in ("a_last", "a_mid", "a_inline")]
        kb = [BUCKETS.index(x) for x in ("b_first", "b_rest")]
        for nm, ks in (("a", ka), ("b", kb)):
            s, c = S[r][:, ks].sum(1), C[r][:, ks].sum(1)
            rr[nm] = dict(n=int(c.sum()), per_token=float(s.sum() / max(c.sum(), 1)),
                          per_token_se=float((s[boot].sum(1) / np.maximum(c[boot].sum(1), 1)).std()),
                          total=float(s.sum() / (nw * CTX)), total_se=float((s[boot].sum(1) / (nw * CTX)).std()))
        sa, ca = S[r][:, ka].sum(1), C[r][:, ka].sum(1)
        sb, cb = S[r][:, kb].sum(1), C[r][:, kb].sum(1)
        diff = sb.sum() / max(cb.sum(), 1) - sa.sum() / max(ca.sum(), 1)
        bd = sb[boot].sum(1) / np.maximum(cb[boot].sum(1), 1) - sa[boot].sum(1) / np.maximum(ca[boot].sum(1), 1)
        rr["b_minus_a"] = dict(per_token=float(diff), se=float(bd.std()))
        dt = sb.sum() - sa.sum()
        rr["b_minus_a_total"] = dict(total=float(dt / (nw * CTX)), se=float(((sb[boot].sum(1) - sa[boot].sum(1)) /
                                                                            (nw * CTX)).std()))
        allc = S[r].sum(1)
        rr["all"] = dict(total=float(allc.sum() / (nw * CTX)), se=float((allc[boot].sum(1) / (nw * CTX)).std()))
        res["runs"].append(rr)
        print(f"blocks {rr['blocks']}: (a) {rr['a']['per_token']:+.4f}/tok (n {rr['a']['n']}), (b) "
              f"{rr['b']['per_token']:+.4f}/tok (n {rr['b']['n']}), b-a {diff:+.4f} ± {bd.std():.4f}; Python CE "
              f"+{rr['all']['total']:.4f}; routes changed {rr['routes_changed']:.3f}", flush=True)
    json.dump(res, open(os.path.join(OUT, "ablate.json"), "w"), indent=1)


# ------------------------------------------------------------------------------------------------ 2. attention
def attn_weights(blk, x, rope, mask):
    """attention probabilities of a block (B, H, T, T), fp32, from its input x (the residual before the block)"""
    B, T, d = x.shape
    H, hd = blk.H, blk.hd
    q, k, _ = blk.qkv(blk.n1(x)).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
    q, k = apply_rope(q, *rope), apply_rope(k, *rope)
    s = (q.float() @ k.float().transpose(-1, -2)) / hd ** 0.5
    s = s.masked_fill(~mask, float("-inf"))
    return torch.softmax(s, -1)


LAG = 32
ATT_STATS = ["sp", "sp_same", "sp_prev", "base_sp", "base_same", "sink", "n"]
QCLS = ["первый токен строки с отступом", "прочие не-пробельные"]


@torch.no_grad()
def attn_cmd():
    model, D, dev = model_and_data()
    an, lo, hi = load_annot(dev, D.val.numel())
    offs = code_windows(D, lo, hi)
    nw = offs.numel()
    L, H = len(model.base.blocks), model.base.cfg.n_heads
    acc = np.zeros((nw, L, H, len(QCLS), len(ATT_STATS)))
    lagm = np.zeros((L, len(QCLS), LAG + 1))                         # head-mean attention on the key at t - lag
    khist = np.zeros(LAG + 1)                                       # first-of-line queries: spaces of their indent seen
    ar = torch.arange(CTX + 1, device=dev)
    t0 = time.time()
    for i in range(0, nw, BS):
        o = offs[i:i + BS]
        idx = D.win(o)
        P = o[:, None] + ar[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, aux = model(idx, {"x_in": True})
        T = idx.shape[1]
        rope = model.base.rope(T, dev)
        mask = doc_causal_mask(idx)                                 # B,1,T,T
        sp_k = an["sp"][P]
        line = an["line"][P]
        same = (line[:, :, None] == line[:, None, :]) & (line[:, :, None] >= 0)
        vis = mask[:, 0]
        nvis = vis.sum(-1).double()
        base_sp = (vis & sp_k[:, None, :]).sum(-1).double() / nvis
        base_same = (vis & same & sp_k[:, None, :]).sum(-1).double() / nvis
        sinkpos = vis.float().argmax(-1)                            # first visible key (window / document start)
        nonws = ~an["sp"][P] & ~an["nlc"][P] & (line >= 0) & (idx != 1)
        # first token of a row whose leading indent has space tokens in this window
        lead_rows = (same & an["lead"][P][:, None, :] & vis).any(-1)
        indented_first = an["first"][P] & lead_rows & nonws
        qc = [indented_first, nonws & ~indented_first]
        kq = (vis & same & sp_k[:, None, :]).sum(-1).clamp_max(LAG)
        khist += torch.bincount(kq[indented_first], minlength=LAG + 1).cpu().numpy()
        for l in range(L):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                Aw = attn_weights(model.base.blocks[l], aux["x_in"][l], rope, mask)   # B,H,T,T
            s_sp = (Aw * sp_k[:, None, None, :]).sum(-1).double()
            s_same = (Aw * (same & sp_k[:, None, :])[:, None]).sum(-1).double()
            s_sink = Aw.gather(-1, sinkpos[:, None, :, None].expand(-1, Aw.shape[1], -1, 1))[..., 0].double()
            for c, q in enumerate(qc):
                qd = q.double()[:, None]                            # B,1,T
                vals = [s_sp, s_same, s_sp - s_same, base_sp[:, None].expand_as(s_sp),
                        base_same[:, None].expand_as(s_sp), s_sink, torch.ones_like(s_sp)]
                for k, v in enumerate(vals):
                    acc[i:i + BS, l, :, c, k] = (v * qd).sum(-1).cpu().numpy()
                for lag in range(1, LAG + 1):
                    dg = torch.diagonal(Aw, offset=-lag, dim1=-2, dim2=-1).mean(1)      # B, T - lag
                    lagm[l, c, lag] += float((dg.double() * q[:, lag:].double()).sum())
        if i % (BS * 40) == 0:
            print(f"{i + o.numel()}/{nw} windows ({time.time() - t0:.0f}s)", flush=True)
    np.savez(os.path.join(OUT, "attn_win.npz"), acc=acc, lagm=lagm, khist=khist)
    rng = np.random.default_rng(0)
    boot = rng.integers(0, nw, (500, nw))
    res = dict(windows=int(nw), stats=ATT_STATS, qcls=QCLS, layers=[])
    for l in range(L):
        lay = dict(block=l + 1, classes=[])
        for c in range(len(QCLS)):
            a = acc[:, l, :, c, :]                                  # nw, H, stats
            n = a[:, 0, -1]
            tot = a.sum(0)
            heads = [dict(sp=float(tot[h, 0] / tot[h, -1]), sp_same=float(tot[h, 1] / tot[h, -1]),
                          sp_prev=float(tot[h, 2] / tot[h, -1]), sink=float(tot[h, 5] / tot[h, -1]))
                     for h in range(H)]
            mean_sp = a[:, :, 0].sum(1) / H                          # per window, head mean
            mean_same = a[:, :, 1].sum(1) / H
            bsp, bsame = a[:, 0, 3], a[:, 0, 4]
            r = lambda num: (num[boot].sum(1) / np.maximum(n[boot].sum(1), 1))   # noqa: E731
            lay["classes"].append(dict(
                n=int(n.sum()), sp=float(mean_sp.sum() / n.sum()), sp_se=float(r(mean_sp).std()),
                sp_same=float(mean_same.sum() / n.sum()), sp_same_se=float(r(mean_same).std()),
                base_sp=float(bsp.sum() / n.sum()), base_same=float(bsame.sum() / n.sum()),
                ratio=float(mean_sp.sum() / bsp.sum()), ratio_se=float((mean_sp[boot].sum(1) / bsp[boot].sum(1)).std()),
                ratio_same=float(mean_same.sum() / max(bsame.sum(), 1e-9)),
                ratio_same_se=float((mean_same[boot].sum(1) / np.maximum(bsame[boot].sum(1), 1e-9)).std()),
                sink=float(a[:, :, 5].sum() / (H * n.sum())), heads=heads,
                head_ratio=[float(tot[h, 0] / tot[h, 3]) for h in range(H)]))
        # recency control: what the other non-space queries give to their own k nearest previous keys, k distributed
        # as the indent spaces seen by the first-of-line queries
        nq = [acc[:, l, 0, c, -1].sum() for c in range(len(QCLS))]
        cum_other = np.cumsum(lagm[l, 1] / nq[1])
        cum_first = np.cumsum(lagm[l, 0] / nq[0])
        lay["recency"] = dict(other_on_k_nearest=float((khist * cum_other).sum() / khist.sum()),
                              first_on_k_nearest=float((khist * cum_first).sum() / khist.sum()),
                              lag_other=(lagm[l, 1, 1:] / nq[1]).tolist(), lag_first=(lagm[l, 0, 1:] / nq[0]).tolist())
        res["layers"].append(lay)
        c0, c1 = lay["classes"]
        print(f"block {l + 1}: first-of-line sp {c0['sp']:.3f} vs base {c0['base_sp']:.3f} (x{c0['ratio']:.2f}; same "
              f"line x{c0['ratio_same']:.2f}) | other sp {c1['sp']:.3f} vs {c1['base_sp']:.3f} (x{c1['ratio']:.2f}); "
              f"max head x{max(c1['head_ratio']):.2f}; recency control: other queries on their k nearest "
              f"{lay['recency']['other_on_k_nearest']:.3f}", flush=True)
    res["khist"] = khist.tolist()
    json.dump(res, open(os.path.join(OUT, "attn.json"), "w"), indent=1)


# ------------------------------------------------------------------------------------------------ 3. probes
def balanced_acc(cm):
    """cm (..., K, K) rows = true class: mean recall over the classes present"""
    tot = cm.sum(-1)
    rec = np.diagonal(cm, axis1=-2, axis2=-1) / np.maximum(tot, 1)
    pres = tot > 0
    return (rec * pres).sum(-1) / np.maximum(pres.sum(-1), 1)


def fit_logreg(X, y, K, lam, iters=150):
    """multinomial logistic regression, class-balanced weights, L2 lam on standardised features (LBFGS, fp32)"""
    d = X.shape[1]
    W = torch.zeros(d, K, device=X.device, requires_grad=True)
    b = torch.zeros(K, device=X.device, requires_grad=True)
    cnt = torch.bincount(y, minlength=K).float()
    w = torch.where(cnt > 0, cnt.sum() / (cnt.clamp_min(1) * (cnt > 0).sum()), torch.zeros_like(cnt))
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(X @ W + b, y, weight=w) + lam * W.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach(), b.detach()


def predict(X, Wb):
    return (X @ Wb[0] + Wb[1]).argmax(1)


def doc_cm(doc, y, p, K, nd):
    cm = np.zeros((nd, K, K))
    np.add.at(cm, (doc, y, p), 1)
    return cm


def lookup_pred(g_tr, y_tr, g_te, K, G):
    """class-balanced Bayes rule from a single discrete feature g: argmax_c P(c | g) / P(c)"""
    t = np.zeros((G, K))
    np.add.at(t, (g_tr, y_tr), 1)
    prior = np.bincount(y_tr, minlength=K) / y_tr.size
    score = t / np.maximum(t.sum(1, keepdims=True), 1) / np.maximum(prior, 1e-12)
    return score.argmax(1)[g_te]


@torch.no_grad()
def collect_probe(model, D, an, sets, offs, cap, dev):
    """inputs z, outputs and the stream after the block at the indent spaces of logical lines the «spaces» experts
    take (baseline routing), a random subset of about `cap` positions per block"""
    M = model.M
    ar = torch.arange(CTX + 1, device=dev)
    elig = an["lead"].clone()
    g = torch.Generator(device=dev).manual_seed(0)
    u = torch.rand(elig.numel(), device=dev, generator=g)
    p_keep = min(1.0, cap / max(1, int(elig.sum())) * 1.15)
    keep = elig & (u < p_keep)
    out = [{k: [] for k in ("z", "uf", "x", "pos")} for _ in range(M)]
    t0 = time.time()
    for i in range(0, offs.numel(), BS):
        o = offs[i:i + BS]
        idx = D.win(o)
        P = o[:, None] + ar[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, aux = model(idx, {"ffn_io": True})
        kp = keep[P]
        kp[:, CTX] = False                                     # targets of the last position may need the next token
        for j in range(M):
            m = kp & torch.isin(aux["eid"][j], torch.tensor(sets[j], device=dev))
            out[j]["z"].append(aux["z"][j][m].half().cpu())
            out[j]["uf"].append(aux["uf"][j][m].half().cpu())
            out[j]["x"].append((aux["y"][j][m].float() + aux["uf"][j][m].float()).half().cpu())
            out[j]["pos"].append(P[m].cpu())
        if i % (BS * 40) == 0:
            print(f"collect {i + o.numel()}/{offs.numel()} ({time.time() - t0:.0f}s)", flush=True)
    return [{k: torch.cat(v) for k, v in o_.items()} for o_ in out]


def probe_cmd(cap=160000, folds=4, lams=(1e-5, 1e-4, 1e-3, 1e-2), n_boot=500):
    model, D, dev = model_and_data()
    an, lo, hi = load_annot(dev, D.val.numel())
    sets = ws_sets(dev)
    offs = code_windows(D, lo, hi)
    data = collect_probe(model, D, an, sets, offs, cap, dev)
    del model
    torch.cuda.empty_cache()
    z = np.load(os.path.join(OUT, "annot.npz"))
    tg = {k: z[k] for k in ("depth", "cnt", "hdr", "same_prev", "same_next", "line_ret", "run_end", "next_ret", "doc",
                            "kind", "logical")}
    nd = int(tg["doc"].max()) + 1
    rng = np.random.default_rng(0)
    fold_of_doc = np.empty(nd, np.int64)
    fold_of_doc[rng.permutation(nd)] = np.arange(nd) % folds
    res = dict(cap=cap, folds=folds, lams=list(lams), blocks=[])
    t0 = time.time()
    for j, dj in enumerate(data):
        pos = dj["pos"].numpy() - lo
        doc = tg["doc"][pos]
        fold = fold_of_doc[doc]
        cnt_raw = tg["cnt"][pos].astype(np.int64)
        lg = tg["logical"][pos]
        y_all = dict(depth=np.minimum(tg["depth"][pos], 5) - 1, cnt=np.minimum(cnt_raw, 16) - 1, hdr=tg["hdr"][pos],
                     same_prev=tg["same_prev"][pos], same_next=tg["same_next"][pos], line_ret=tg["line_ret"][pos],
                     run_end=tg["run_end"][pos].astype(np.int64), next_ret=tg["next_ret"][pos].astype(np.int64))
        y_all = {k: np.where(lg, v, -1) for k, v in y_all.items()}
        y_all["kind"] = tg["kind"][pos].astype(np.int64)
        blk = dict(block=5 + j, n=int(pos.size), docs=int(np.unique(doc).size), targets={})
        Xs = {r: dj[r].to(dev).float() for r, _ in REPS}
        for name, K, _ in TARGETS:
            y = y_all[name].astype(np.int64)
            ok = (y >= 0) & (y < K)
            yo, fo, do = y[ok], fold[ok], doc[ok]
            g = np.minimum(cnt_raw[ok], 16)
            tr = dict(n=int(ok.sum()), freq=(np.bincount(yo, minlength=K) / max(1, yo.size)).tolist(), reps={})
            # baseline: the count of indent spaces typed so far, alone
            pred = np.zeros_like(yo)
            for f in range(folds):
                te_ = fo == f
                pred[te_] = lookup_pred(g[~te_], yo[~te_], g[te_], K, 17)
            cms = {"count": doc_cm(do, yo, pred, K, nd)}
            yt = torch.from_numpy(yo).to(dev)
            for r, _ in REPS:
                X = Xs[r][torch.from_numpy(ok).to(dev)]
                # lambda: inner split of fold 0's training files
                tr0 = np.flatnonzero(fo != 0)
                inner = (do[tr0] % 4) == 0
                a_i, b_i = torch.from_numpy(tr0[~inner]).to(dev), torch.from_numpy(tr0[inner]).to(dev)
                mu, sd = X[a_i].mean(0), X[a_i].std(0) + 1e-6
                best = None
                for lam in lams:
                    Wb = fit_logreg((X[a_i] - mu) / sd, yt[a_i], K, lam)
                    p = predict((X[b_i] - mu) / sd, Wb).cpu().numpy()
                    ba = float(balanced_acc(doc_cm(np.zeros(b_i.numel(), np.int64), yo[tr0[inner]], p, K, 1))[0])
                    if best is None or ba > best[0]:
                        best = (ba, lam)
                pred = np.zeros_like(yo)
                for f in range(folds):
                    tr_i = torch.from_numpy(np.flatnonzero(fo != f)).to(dev)
                    te_np = np.flatnonzero(fo == f)
                    te_i = torch.from_numpy(te_np).to(dev)
                    mu, sd = X[tr_i].mean(0), X[tr_i].std(0) + 1e-6
                    Wb = fit_logreg((X[tr_i] - mu) / sd, yt[tr_i], K, best[1])
                    pred[te_np] = predict((X[te_i] - mu) / sd, Wb).cpu().numpy()
                cms[r] = doc_cm(do, yo, pred, K, nd)
                tr["reps"][r] = dict(lam=best[1])
            # balanced accuracy and bootstrap over files
            docs_present = np.unique(do)
            bidx = rng.choice(docs_present, (n_boot, docs_present.size))
            bt = {}
            for k, cm in cms.items():
                tr.setdefault("bacc", {})[k] = float(balanced_acc(cm.sum(0)))
                bt[k] = balanced_acc(cm[bidx].sum(1))
                tr.setdefault("bacc_se", {})[k] = float(bt[k].std())
            for a_, b_ in (("uf", "z"), ("x", "z"), ("z", "count"), ("uf", "count")):
                tr.setdefault("diff", {})[f"{a_}-{b_}"] = dict(d=tr["bacc"][a_] - tr["bacc"][b_],
                                                               se=float((bt[a_] - bt[b_]).std()))
            tr["chance"] = 1.0 / max(1, int((np.bincount(yo, minlength=K) > 0).sum()))
            blk["targets"][name] = tr
            print(f"block {5 + j} {name}: count {tr['bacc']['count']:.3f} | z {tr['bacc']['z']:.3f} | out "
                  f"{tr['bacc']['uf']:.3f} | after {tr['bacc']['x']:.3f} (out-in {tr['diff']['uf-z']['d']:+.3f} ± "
                  f"{tr['diff']['uf-z']['se']:.3f}) n {tr['n']} ({time.time() - t0:.0f}s)", flush=True)
        res["blocks"].append(blk)
        del Xs
        torch.cuda.empty_cache()
    json.dump(res, open(os.path.join(OUT, "probe.json"), "w"), indent=1)


# ------------------------------------------------------------------------------------------------ report
def pm(v, se, d=4, sign=True):
    return f"{v:+.{d}f} ± {se:.{d}f}" if sign else f"{v:.{d}f} ± {se:.{d}f}"


def passes(v, se):
    return abs(v) > 2 * se


def report_cmd():
    sets = json.load(open(os.path.join(OUT, "sets.json")))
    ab = json.load(open(os.path.join(OUT, "ablate.json")))
    at = json.load(open(os.path.join(OUT, "attn.json")))
    pr = json.load(open(os.path.join(OUT, "probe.json")))
    yes = lambda v, se: "да" if passes(v, se) else "нет"                      # noqa: E731
    L = []
    L.append("# r21: эксперты «пробелы в коде» у MoE-perlayer (val, только Python)\n")
    L.append(f"Дома, без обучения; модель `runs_vast/mix_perlayer` (сид 0, второго сида нет). Python-часть val: "
             f"{sets['tokens'] / 1e6:.1f}M токенов, {sets['files']} файлов, {ab['windows']} окон по 512 токенов целиком "
             f"внутри неё. Пробел = токен только из пробелов / табуляций; в GPT-2 BPE каждый пробел отступа — отдельный "
             f"токен ' ', последний приклеен к слову (' def'). Пробелы — {100 * sets['sp_share']:.1f}% токенов Python, "
             f"{100 * sets['lead_of_sp']:.1f}% из них стоят в отступе в начале строки. Строки этих пробелов отступа: "
             + ", ".join(f"{n} {100 * v:.1f}%" for n, v in zip(("логические", "продолжения в скобках",
                                                                 "внутри многострочных строк", "комментарии"),
                                                                sets["kind_of_lead"]))
             + f" (неизвестно — после ошибки разбора в {sets['files_tokenize_error']} файлах — "
             f"{100 * sets['kind_unknown']:.1f}%).\n")
    L.append("Эксперты «пробелы в коде» блока — те, у кого ≥ 50% токенов val — пробелы Python:\n")
    L.append("| блок | эксперты (доля пробелов Python среди их токенов) | доля пробелов Python у них |\n|---|---|---|")
    for s in sets["sets"]:
        L.append(f"| {5 + s['j']} | " + ", ".join(f"E{e} ({100 * s['share_sp'][e]:.0f}%)" for e in s["experts"]) +
                 f" | {100 * s['cover']:.1f}% |")
    L.append("\nРазброс: у perlayer один сид, поэтому планка — шум измерения, а не разброс по сидам: бутстрэп по окнам "
             "val (пункты 1–2) или по файлам (пункт 3), ± = 1 SE; «проходит» = разница > 2 SE.\n")
    L.append("## Итог\n")
    L.append(CONCLUSIONS["summary"] + "\n")
    for k, name in (("ablate", "Кому нужен"), ("attn", "Читают ли"), ("probe", "Что записывает")):
        L.append(f"- **{name}.** {CONCLUSIONS[k]}")
    L.append("")
    # 1
    L.append("## 1. Кому нужен эксперт: выключение только на позициях пробелов\n")
    L.append("Выход эксперта(ов) «пробелы» = 0 только там, где токен — пробел и маршрут ведёт в этого эксперта "
             "(маршруты следующих блоков perlayer пересчитываются по изменённому состоянию). CE позиции t — потеря "
             "предсказания токена t+1 из позиции t. (а) — сами выключенные позиции; (б) — не-пробельные токены строки, "
             "в отступе которой были выключенные пробелы: первый токен строки и дальше по строке; остальное — все "
             "прочие позиции Python. Рост CE, нат на токен.\n")
    L.append("| выключены блоки | позиций (а) | (а) на токен | (б) на токен | (б) − (а) | > 2 SE | вклад (а) в CE Python "
             "| вклад (б) | вклад остального | весь рост CE Python | маршрут сменился |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for r in ab["runs"]:
        c = r["buckets"]["c"]
        L.append(f"| {', '.join(map(str, r['blocks']))} | {r['a']['n']} | {pm(r['a']['per_token'], r['a']['per_token_se'])} "
                 f"| {pm(r['b']['per_token'], r['b']['per_token_se'])} | {pm(r['b_minus_a']['per_token'], r['b_minus_a']['se'])} "
                 f"| {yes(r['b_minus_a']['per_token'], r['b_minus_a']['se'])} | {pm(r['a']['total'], r['a']['total_se'])} "
                 f"| {pm(r['b']['total'], r['b']['total_se'])} | {pm(c['total'], c['total_se'])} "
                 f"| {pm(r['all']['total'], r['all']['se'])} | {100 * r['routes_changed']:.1f}% |")
    L.append("\nПо частям (рост CE на токен ± SE; в скобках — CE этих позиций без выключения и их доля среди позиций Python):\n")
    L.append("| выключены блоки | " + " | ".join(BUCKET_RU[b] for b in BUCKETS) + " |\n|---|" + "---|" * len(BUCKETS))
    for r in ab["runs"]:
        L.append(f"| {', '.join(map(str, r['blocks']))} | " + " | ".join(
            f"{pm(x['per_token'], x['per_token_se'])} ({x['base_ce']:.2f}; {100 * x['share']:.1f}%)"
            for x in (r["buckets"][b] for b in BUCKETS)) + " |")
    L.append("\nСколько из роста CE приходится на решение «следующий токен — пробел или нет» (−log P(пробел) или "
             "−log(1 − P(пробел))); остальное — выбор конкретного токена внутри класса:\n")
    L.append("| выключены блоки | " + " | ".join(BUCKET_RU[b] for b in BUCKETS) + " |\n|---|" + "---|" * len(BUCKETS))
    for r in ab["runs"]:
        L.append(f"| {', '.join(map(str, r['blocks']))} | " + " | ".join(
            f"{pm(x['space_part'], x['space_part_se'])} из {x['per_token']:+.4f}"
            for x in (r["buckets"][b] for b in BUCKETS)) + " |")
    # 2
    L.append("\n## 2. Читают ли его: внимание не-пробельных токенов к позициям пробелов\n")
    L.append("Для каждого блока — средняя по 8 головам доля внимания, которую не-пробельные токены Python (запросы) отдают "
             "ключам-пробелам, и для сравнения доля пробелов среди ключей, которые этот запрос видит (столько дало бы "
             "равномерное внимание). «Своя строка» — только пробелы отступа той же строки. Запросы двух видов: первый "
             "токен строки с отступом и все прочие не-пробельные. Контроль близости: пробелы отступа стоят прямо перед "
             "первым токеном строки, а ближайшим ключам внимания и так достаётся больше; поэтому для первых токенов "
             "рядом — сколько прочие не-пробельные токены отдают такому же числу своих ближайших предыдущих ключей "
             "(число k распределено как длина видимого отступа у первых токенов, в среднем "
             f"{sum(i * v for i, v in enumerate(at['khist'])) / sum(at['khist']):.1f}).\n")
    L.append("| блок | запросы | доля внимания на пробелы | доля пробелов среди видимых | отношение | на пробелы своей строки "
             "| их доля среди видимых | отношение | контроль близости: прочие токены на k ближайших | лучшая голова: "
             "отношение | на первый токен окна |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for lay in at["layers"]:
        for k, (c, cl) in enumerate(zip(lay["classes"], at["qcls"])):
            rc = f"{lay['recency']['other_on_k_nearest']:.3f}" if k == 0 else ""
            L.append(f"| {lay['block']} | {cl} | {pm(c['sp'], c['sp_se'], 3, False)} | {c['base_sp']:.3f} "
                     f"| {pm(c['ratio'], c['ratio_se'], 2, False)} | {pm(c['sp_same'], c['sp_same_se'], 3, False)} "
                     f"| {c['base_same']:.3f} | {pm(c['ratio_same'], c['ratio_same_se'], 2, False)} | {rc} "
                     f"| {max(c['head_ratio']):.2f} | {c['sink']:.3f} |")
    # 3
    L.append("\n## 3. Что он записывает: линейные пробы на входе и на выходе\n")
    L.append(f"Позиции — пробелы отступа, которые взял эксперт «пробелы» блока (случайная выборка ~{pr['cap'] // 1000}k на "
             f"блок; все признаки, кроме типа строки, — только на логических строках). Проба — многоклассовая "
             f"логистическая регрессия с весами классов, признаки стандартизованы, L2 подобран на внутреннем "
             f"разбиении; {pr['folds']} фолда по файлам, метрика — сбалансированная точность (средний recall по "
             f"классам; случайное угадывание = 1/число классов). «По счётчику» — то же правило Байеса только по числу "
             f"уже набранных пробелов отступа. Вход — z = RMSNorm(поток) перед экспертом; выход — то, что эксперт "
             f"прибавляет к потоку; поток после — сумма.\n")
    for b in pr["blocks"]:
        L.append(f"\n### Блок {b['block']} ({b['n']} позиций, {b['docs']} файлов)\n")
        L.append("| признак | позиций | случайно | по счётчику | вход z | выход | поток после | выход − вход | > 2 SE "
                 "| поток после − вход | > 2 SE |\n|---|---|---|---|---|---|---|---|---|---|---|")
        for name, K, ru in TARGETS:
            t = b["targets"][name]
            ba, se = t["bacc"], t["bacc_se"]
            d1, d2 = t["diff"]["uf-z"], t["diff"]["x-z"]
            L.append(f"| {ru} | {t['n']} | {t['chance']:.3f} | {pm(ba['count'], se['count'], 3, False)} "
                     f"| {pm(ba['z'], se['z'], 3, False)} | {pm(ba['uf'], se['uf'], 3, False)} "
                     f"| {pm(ba['x'], se['x'], 3, False)} | {pm(d1['d'], d1['se'], 3)} | {yes(d1['d'], d1['se'])} "
                     f"| {pm(d2['d'], d2['se'], 3)} | {yes(d2['d'], d2['se'])} |")
    open(MD, "w", encoding="utf-8").write("\n".join(L) + "\n")
    # compact data for the card on the experts page
    card = dict(sets=sets["sets"], tokens=sets["tokens"], files=sets["files"], sp_share=sets["sp_share"],
                lead_of_sp=sets["lead_of_sp"], kind_of_lead=sets["kind_of_lead"], windows=ab["windows"],
                ablate=[dict(blocks=r["blocks"], n=r["a"]["n"], a=r["a"], b=r["b"], b_minus_a=r["b_minus_a"],
                             all=r["all"], routes_changed=r["routes_changed"],
                             parts={k: dict(per_token=v["per_token"], se=v["per_token_se"], share=v["share"],
                                            base_ce=v["base_ce"], total=v["total"], space_part=v["space_part"],
                                            space_part_se=v["space_part_se"])
                                    for k, v in r["buckets"].items()}) for r in ab["runs"]],
                bucket_names=BUCKET_RU,
                attn=[dict(block=l_["block"], cls=[dict(sp=c["sp"], sp_se=c["sp_se"], base=c["base_sp"],
                                                         ratio=c["ratio"], ratio_se=c["ratio_se"], same=c["sp_same"],
                                                         base_same=c["base_same"], ratio_same=c["ratio_same"],
                                                         ratio_same_se=c["ratio_same_se"],
                                                         best_head=max(c["head_ratio"]), sink=c["sink"])
                                                    for c in l_["classes"]],
                          recency=l_["recency"]["other_on_k_nearest"]) for l_ in at["layers"]],
                k_mean=sum(i * v for i, v in enumerate(at["khist"])) / sum(at["khist"]),
                qcls=at["qcls"],
                probe=[dict(block=b["block"], n=b["n"],
                            t=[dict(name=name, ru=ru, n=b["targets"][name]["n"], chance=b["targets"][name]["chance"],
                                    bacc=b["targets"][name]["bacc"], se=b["targets"][name]["bacc_se"],
                                    diff=b["targets"][name]["diff"]) for name, _, ru in TARGETS])
                       for b in pr["blocks"]],
                conclusions=CONCLUSIONS)
    json.dump(card, open(JS, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"-> {MD}, {JS}")


CONCLUSIONS = {  # written after the measurements of 2026-11-11 (the numbers are in report/r21_ws.md)
    "summary": "Эксперты «пробелы» работают в основном на свою позицию, а не на будущее. Почти весь их вклад приходится "
               "на последний пробел отступа: отсюда модель предсказывает первое слово строки. Не-пробельные токены "
               "читают пробелы реже, чем те встречаются среди видимых ключей. Признаки отступа уже читаются на входе "
               "эксперта, выход их не улучшает; единственное исключение — «следующий токен — return».",
    "ablate": "На токен (а) больше (б) во всех блоках, разница проходит 2 SE: блок 5 — +0.0144 против +0.0049 нат, "
              "6 — +0.0132 против +0.0024, 7 — +0.0340 против +0.0014, 8 — +0.142 против 0. В блоке 8 выход позиции "
              "никто дальше не читает, поэтому (б) там должен быть нулём; это проверка постановки. По критерию «(б) "
              "сильнее (а)» эксперт на будущее не работает. Почти весь рост (а) — на последнем пробеле отступа, откуда "
              "предсказывается первое слово строки: +0.11 / +0.09 / +0.34 / +1.18 нат, все четыре блока вместе — +2.21. "
              "Из этого на решение «пробел или нет» приходится 29–43%, остальное — на выбор слова. На пробелах внутри "
              "отступа (30% позиций Python) следующий пробел и так почти известен (CE 0.02), рост там не больше "
              "+0.03. Но (б) охватывает 49% позиций, поэтому в сумме он заметен: 31% всего роста CE Python в блоке 5, "
              "20% в блоке 6, 5% в блоке 7. Польза «на будущее» есть, но на токен она в 3–24 раза слабее, чем на самом "
              "пробеле.",
    "attn": "В основном не читают. В блоках 5–8 не-пробельные токены отдают пробелам 7–22% внимания, хотя среди "
            "видимых им ключей пробелов 30% (отношение 0.24 / 0.73 / 0.51 / 0.43). Из 32 голов блоков 5–8 выше доли "
            "пробелов только две: голова 5 блока 6 (×1.33 ± 0.01) и голова 1 блока 7 (×1.31 ± 0.01). Первый токен "
            "строки смотрит на свой отступ в 1.9–4 раза больше доли отступа среди видимых ключей. Но это меньше, чем "
            "обычный токен отдаёт такому же числу ближайших соседей: 0.28 против 0.54 в блоке 5, 0.21 / 0.28, "
            "0.26 / 0.41 и 0.13 / 0.26 в блоках 6–8. Значит, это эффект близости, а не особый интерес к пробелам "
            "(SE контроля не считал; разрыв 0.07–0.26 при SE долей ≤ 0.002).",
    "probe": "Все признаки уже линейно читаются на входе эксперта, и намного лучше, чем по одному счётчику пробелов: "
             "глубина — 0.80–0.81 (по счётчику 0.50), охватывающая конструкция — 0.74–0.76 (0.31), тип строки — "
             "0.81–0.82 (0.36), отступ как у предыдущей строки — 0.89 (0.57), строка начнётся с return — 0.72–0.76 "
             "(0.54). Сам счётчик набранных пробелов читается на 0.985–0.999. Выход эксперта читается лучше входа "
             "только для «следующий токен — return» в блоке 5 (+0.013 ± 0.005); в остальных случаях он хуже на "
             "0.004–0.095 или разница не проходит 2 SE. По критерию задания эксперт эти признаки не вычисляет: они "
             "посчитаны раньше, вниманием нижних блоков. После прибавки выхода поток читается почти как вход "
             "(разницы в пределах ±0.01). Устойчиво лучше становится только «следующий токен — return» (+0.008…"
             "+0.014 во всех четырёх блоках) и «строка начнётся с return» в блоках 5–6 (+0.006 и +0.008). Хуже всего "
             "читается, совпадёт ли отступ следующей строки: 0.67–0.68 против 0.57 по счётчику.",
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["annot", "ablate", "attn", "probe", "report"])
    a = ap.parse_args()
    dict(annot=annot_cmd, ablate=ablate_cmd, attn=attn_cmd, probe=probe_cmd, report=report_cmd)[a.cmd]()
