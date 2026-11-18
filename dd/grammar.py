"""Synthetic language: a probabilistic CFG with nesting, long-distance agreement,
typed bracket closure and name copying.  Every token carries structural annotations.

Sentence shape:
    <bos> NAME said : CLAUSE [CODA] .
    CODA  -> ; MARK NAME_copy V_intr      (NAME_copy == the NAME from the intro)

    CLAUSE[e]  -> NP_subj[e,num] ADV? VP[e,num]
    NP[e,num]  -> DET[num] ADJ{0,2} N[num] MOD?   |  NAME (sg)
    MOD        -> PP[e+1] | RC[e+1]
    PP[e]      -> P NP[e,any]
    RC[e]      -> ( who|which VP[e,num_head] )          subject relative
                | ( whom NP[e,any] V_tr[num_of_that_NP] ) object relative
    VP[e,num]  -> V_intr[num] PP?
                | V_tr[num] NP[e,any]
                | V_cc[num] [ that CLAUSE[e+1] ]        complement clause

`e` is the embedding depth (number of enclosing PP / RC / CP), capped at MAX_EMB.

Checks recorded per token:
  agree  : a finite verb whose number is fixed by its subject (counterpart = other number form)
  copy   : the repeated name in the coda
  close  : a closing bracket ')' or ']' (counterpart = the other closer)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

MAX_EMB = 5

# ----------------------------------------------------------------------------- lexicon
NOUNS = ["key", "cabinet", "door", "table", "window", "letter", "box", "lamp", "garden",
         "river", "bridge", "tower", "road", "car", "ship", "book", "chair", "wall", "stone",
         "tree", "house", "cat", "dog", "bird", "horse", "teacher", "doctor", "farmer",
         "singer", "child", "king", "soldier", "student", "painter", "pilot", "baker",
         "lawyer", "queen", "friend", "neighbor"]
ADJS = ["old", "new", "red", "blue", "green", "small", "big", "dark", "bright", "quiet",
        "loud", "heavy", "light", "strange", "happy", "sad", "tall", "short", "warm", "cold",
        "young", "clever", "lazy", "brave", "gentle", "wild", "rich", "poor", "famous", "tiny"]
V_INTR = ["sleep", "lie", "fall", "wait", "laugh", "shine", "vanish", "rest", "stay", "smile",
          "cry", "run", "swim", "dance", "sing", "glow", "work", "rise", "sit", "breathe"]
V_TR = ["see", "find", "hold", "open", "break", "follow", "like", "watch", "carry", "paint",
        "move", "touch", "keep", "hide", "push", "clean", "build", "visit", "call", "help"]
V_CC = ["think", "say", "know", "believe", "hope", "feel", "hear", "doubt"]
PREPS = ["of", "near", "under", "behind", "with", "from", "beside", "above", "inside", "without"]
ADVS = ["often", "rarely", "always", "never", "quickly", "slowly", "surely", "maybe",
        "still", "soon", "again", "already", "almost", "quietly", "suddenly"]
NAMES = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi", "ivan", "judy",
         "mallory", "nina", "oscar", "peggy", "quinn", "rupert", "sybil", "trent", "ursula",
         "victor", "walter", "xena", "yusuf", "zoe", "amir", "bianca", "cyrus", "dora",
         "elias", "fiona", "gideon", "hana", "igor", "jade", "kofi", "lena", "milo", "nora",
         "omar", "pia", "rafael", "sara", "tomas", "uma", "vera", "wim", "yara", "zane",
         "anya", "boris", "clara", "dmitri", "edith", "felix", "greta", "hugo", "iris",
         "jonas", "kira", "leo"]
DET_SG = ["a", "this", "every", "one"]
DET_PL = ["these", "many", "some", "few", "two"]
DET_ANY = ["the", "my", "our"]
CODA_MARKS = ["then", "so", "later", "and"]

SPECIALS = ["<pad>", "<bos>"]
BRACKETS = [("[", "]"), ("{", "}"), ("<", ">")]
FUNCTION = ["said", ":", ";", ".", "[", "]", "{", "}", "<", ">", "who", "which", "whom", "that"]


def _plural(w: str) -> str:
    if w == "child":
        return "children"
    if w.endswith(("s", "x", "sh", "ch")):
        return w + "es"
    if w.endswith("y") and w[-2] not in "aeiou":
        return w[:-1] + "ies"
    return w + "s"


def _verb_sg(w: str) -> str:  # 3rd person singular present
    if w.endswith(("s", "x", "sh", "ch")):
        return w + "es"
    if w.endswith("y") and w[-2] not in "aeiou":
        return w[:-1] + "ies"
    return w + "s"


@dataclass
class Vocab:
    itos: list[str]
    stoi: dict[str, int]
    cls: dict[str, str]          # word -> word class
    flip: dict[int, int]         # number counterpart for nouns/verbs, closer counterpart

    def __len__(self):
        return len(self.itos)


def build_vocab() -> Vocab:
    itos, cls = [], {}

    def add(w, c):
        if w not in cls:
            itos.append(w)
            cls[w] = c

    for w in SPECIALS:
        add(w, "special")
    for w in FUNCTION:
        add(w, {"[": "open", "{": "open", "<": "open", "]": "close", "}": "close", ">": "close",
                "who": "rel", "which": "rel", "whom": "rel", "that": "comp"}.get(w, "func"))
    for w in DET_SG + DET_PL + DET_ANY:
        add(w, "det")
    for n in NOUNS:
        add(n, "noun")
        add(_plural(n), "noun")
    for a in ADJS:
        add(a, "adj")
    for v in V_INTR + V_TR + V_CC:
        add(_verb_sg(v), "verb")
        add(v, "verb")
    for p in PREPS:
        add(p, "prep")
    for a in ADVS:
        add(a, "adv")
    for n in NAMES:
        add(n, "name")
    for m in CODA_MARKS:
        add(m, "func")
    stoi = {w: i for i, w in enumerate(itos)}
    flip = {}
    for n in NOUNS:
        a, b = stoi[n], stoi[_plural(n)]
        flip[a], flip[b] = b, a
    for v in V_INTR + V_TR + V_CC:
        a, b = stoi[_verb_sg(v)], stoi[v]
        flip[a], flip[b] = b, a
    return Vocab(itos, stoi, cls, flip)


# ----------------------------------------------------------------------------- tree
@dataclass
class Node:
    label: str
    children: list = field(default_factory=list)   # Node or Leaf
    head: "Leaf | None" = None


@dataclass
class Leaf:
    word: str
    # annotations filled by the generator
    agree: int = 0          # 1 = verb whose number is fixed by its subject
    attractor: int = 0      # 1 = a noun of opposite number intervenes between subject head and verb
    agree_dist: int = 0     # distance (tokens) from subject head to verb
    copy: int = 0
    close: int = 0
    kind: int = -1          # agreement kind: 0 bare subj, 1 subj+PP, 2 subj+RC, 3 subj-RC verb, 4 obj-RC verb
    ev: int = 0             # story: index (1..m) of the swap event this token belongs to
    ansq: int = 0           # story: 1 on the token whose next-token prediction is the answer
    part: int = 0           # 0 intro/bos/punct, 1 main clause (syntax), 2 story/query
    qm: int = -1            # v5: number of swaps before this question (on question ORD and answer NAME)
    qi: int = -1            # v5: index of the question in the sentence


class Generator:
    """Probabilistic generator; probabilities of embedding decay with depth.
    Tracks self.lp = log-probability of the derivation (the grammar is unambiguous, so this is
    the log-probability of the sentence and -mean(lp)/tokens is the entropy floor of the LM)."""

    def __init__(self, seed: int, p_mod_subj=0.75, p_mod_obj=0.40, p_cc=0.38, p_coda=0.85,
                 decay=0.90):
        self.r = random.Random(seed)
        self.p_mod_subj, self.p_mod_obj, self.p_cc = p_mod_subj, p_mod_obj, p_cc
        self.p_coda, self.decay = p_coda, decay
        self.lp = 0.0

    # helpers (every random decision goes through these and is scored)
    def ch(self, xs):
        self.lp -= math.log(len(xs))
        return self.r.choice(xs)

    def flip(self, p):
        x = self.r.random() < p
        if p > 0:
            self.lp += math.log(p if x else 1.0 - p)
        return x

    def cat(self, items, probs):
        i = self.r.choices(range(len(items)), probs)[0]
        self.lp += math.log(probs[i] / sum(probs))
        return items[i]

    def num(self):
        return self.ch(("sg", "pl"))

    def emb_scale(self, e):
        return self.decay ** e

    # --- phrases
    def NP(self, e, num, role, allow_name=True):
        if allow_name and self.flip(0.12):
            leaf = Leaf(self.ch(NAMES))
            n = Node("NP", [leaf], head=leaf)
            n.num, n.head_leaf, n.mod = "sg", leaf, 0
            return n
        if num is None:
            num = self.num()
        det = self.ch((DET_SG if num == "sg" else DET_PL) + DET_ANY)
        kids = [Leaf(det)]
        n_adj = self.cat((0, 1, 2), (0.55, 0.33, 0.12))
        for _ in range(n_adj):
            kids.append(Leaf(self.ch(ADJS)))
        base = self.ch(NOUNS)
        head = Leaf(base if num == "sg" else _plural(base))
        kids.append(head)
        node = Node("NP", kids, head=head)
        node.num, node.head_leaf, node.mod = num, head, 0
        p_mod = (self.p_mod_subj if role == "subj" else self.p_mod_obj) * self.emb_scale(e)
        if e < MAX_EMB and self.flip(p_mod):
            if self.flip(0.5):
                node.children.append(self.PP(e + 1))
                node.mod = 1
            else:
                node.children.append(self.RC(e + 1, num, head))
                node.mod = 2
        return node

    def PP(self, e):
        p = Leaf(self.ch(PREPS))
        return Node("PP", [p, self.NP(e, None, "obj")], head=p)

    def RC(self, e, head_num, head_leaf):
        """Relative clause WITHOUT brackets: its end is implicit (as in natural language)."""
        if self.flip(0.5):
            rel = Leaf(self.ch(("who", "which")))
            vp = self.VP(e, head_num, subj_head=head_leaf, kind=3)
            kids = [rel, vp]
        else:
            rel = Leaf("whom")
            subj = self.NP(e, None, "subj_rc", allow_name=True)
            v = self.ch(V_TR)
            vleaf = Leaf(_verb_sg(v) if subj.num == "sg" else v)
            vleaf.agree, vleaf.kind, vleaf._subj = 1, 4, subj.head_leaf
            vp = Node("VP", [vleaf], head=vleaf)
            kids = [rel, subj, vp]
        return Node("RC", kids, head=rel)

    def VP(self, e, num, subj_head, kind):
        p_cc = self.p_cc * self.emb_scale(e) if e < MAX_EMB else 0.0
        choice = self.cat(("cc", "intr", "tr"), (p_cc, (1 - p_cc) * 0.45, (1 - p_cc) * 0.55))
        if choice == "cc":
            v = self.ch(V_CC)
            vleaf = Leaf(_verb_sg(v) if num == "sg" else v)
            o, c = self.ch(BRACKETS)          # bracket kind is random: closing must match it
            op, cl = Leaf(o), Leaf(c)
            cl.close = 1
            comp = Leaf("that")
            cp = Node("CP", [op, comp, self.CLAUSE(e + 1), cl], head=comp)
            kids = [vleaf, cp]
        elif choice == "intr":
            v = self.ch(V_INTR)
            vleaf = Leaf(_verb_sg(v) if num == "sg" else v)
            kids = [vleaf]
            if e < MAX_EMB and self.flip(0.35 * self.emb_scale(e)):
                kids.append(self.PP(e + 1))
        else:
            v = self.ch(V_TR)
            vleaf = Leaf(_verb_sg(v) if num == "sg" else v)
            kids = [vleaf, self.NP(e, None, "obj")]
        vleaf.agree, vleaf.kind, vleaf._subj = 1, kind, subj_head
        return Node("VP", kids, head=vleaf)

    def CLAUSE(self, e):
        subj = self.NP(e, None, "subj")
        kids = [subj]
        if self.flip(0.2):
            kids.append(Leaf(self.ch(ADVS)))
        vp = self.VP(e, subj.num, subj_head=subj.head_leaf, kind=subj.mod)
        kids.append(vp)
        return Node("CL", kids, head=vp.head)

    def sentence(self):
        self.lp = 0.0
        name = self.ch(NAMES)
        nleaf = Leaf(name)
        intro = Node("INTRO", [Node("NP", [nleaf], head=nleaf), Leaf("said"), Leaf(":")])
        kids = [Leaf("<bos>"), intro, self.CLAUSE(0)]
        if self.flip(self.p_coda):
            c = Leaf(name)
            c.copy = 1
            v = Leaf(_verb_sg(self.ch(V_INTR)))
            kids.append(Node("CODA", [Leaf(";"), Leaf(self.ch(CODA_MARKS)),
                                      Node("NP", [c], head=c), Node("VP", [v], head=v)]))
        kids.append(Leaf("."))
        return Node("S", kids)


# ----------------------------------------------------------------------------- flatten
PHRASAL = {"NP", "VP", "PP", "RC", "CP", "CL", "INTRO", "CODA"}
EMBEDDING = {"PP", "RC", "CP"}


def flatten(root: Node, vocab: Vocab, sg_nouns=None):
    """Returns dict of per-token lists + list of spans (start, end_inclusive, label).
    sg_nouns: set of singular noun forms (default: this module's NOUNS)."""
    sg_nouns = set(NOUNS) if sg_nouns is None else sg_nouns
    toks, tree_depth, emb_depth, is_head, opens, closes = [], [], [], [], [], []
    leaves, spans = [], []

    def rec(node, d, e):
        if isinstance(node, Leaf):
            leaves.append(node)
            toks.append(vocab.stoi[node.word])
            tree_depth.append(d)
            emb_depth.append(e)
            is_head.append(0)
            opens.append(0)
            closes.append(0)
            return
        start = len(toks)
        e2 = e + (1 if node.label in EMBEDDING else 0)
        for c in node.children:
            rec(c, d + 1, e2)
        end = len(toks) - 1
        if node.label in PHRASAL:
            spans.append((start, end, node.label))
            opens[start] += 1
            closes[end] += 1
            if node.head is not None:
                idx = next(i for i in range(start, end + 1) if leaves[i] is node.head)
                is_head[idx] = 1

    rec(root, -1, 0)   # root S has depth -1 so its children are depth 0
    # agreement metadata
    pos = {id(l): i for i, l in enumerate(leaves)}
    agree, attractor, agree_dist, copy, close, flip, kind = [], [], [], [], [], [], []
    ev = [l.ev for l in leaves]
    ansq = [l.ansq for l in leaves]
    part = [l.part for l in leaves]
    qm = [l.qm for l in leaves]
    qi = [l.qi for l in leaves]
    for i, l in enumerate(leaves):
        a = l.agree
        att, dist = 0, 0
        if a:
            j = pos[id(l._subj)]
            dist = i - j
            # number of the subject head: name -> sg, noun -> by form
            sw = vocab.itos[toks[j]]
            s_pl = (vocab.cls[sw] == "noun" and sw not in sg_nouns)
            for k in range(j + 1, i):
                w = vocab.itos[toks[k]]
                if vocab.cls[w] == "noun" and ((w not in sg_nouns) != s_pl):
                    att = 1
                    break
        agree.append(a)
        attractor.append(att)
        agree_dist.append(dist)
        copy.append(l.copy)
        close.append(l.close)
        kind.append(l.kind if a else -1)
        flip.append(vocab.flip.get(toks[i], -1) if a else -1)
    return dict(tok=toks, tree_depth=tree_depth, emb_depth=emb_depth, is_head=is_head,
                opens=opens, closes=closes, agree=agree, attractor=attractor,
                agree_dist=agree_dist, copy=copy, close=close, flip=flip, kind=kind,
                ev=ev, ansq=ansq, part=part, qm=qm, qi=qi), spans


def bracketed(root: Node) -> str:
    if isinstance(root, Leaf):
        return root.word
    inner = " ".join(bracketed(c) for c in root.children)
    return f"({root.label} {inner})" if root.label in PHRASAL else inner
