"""Language v3 (harder): center-embedded object relatives, nesting up to MAX_EMB, ordinal name copy,
larger vocabulary.  Reuses Node/Leaf/flatten/bracketed from grammar.py.

    <bos> NAME_1 [, NAME_2 [, NAME_3]] said : CLAUSE [; MARK ORD NAME_ORD V_intr] .
    CLAUSE[e] -> NP_subj ADV? VP
    NP        -> DET ADJ{0,2} N MOD? | NAME
    MOD       -> PP | RC          (RC much more frequent than in v2)
    RC        -> who|which VP  |  whom NP_rcsubj V_tr        (no brackets; object RCs dominate,
                 NP_rcsubj itself often carries an RC  ->  center embedding: N whom N whom N V V V)
    VP        -> V_intr PP? | V_tr NP | V_cc [k that CLAUSE ]k     k in 4 random bracket kinds
"""
from __future__ import annotations

import math
import random

from .grammar import Leaf, Node, Vocab, _plural, _verb_sg, flatten as _flatten, bracketed  # noqa

MAX_EMB = 8

NOUNS = ["key", "cabinet", "door", "table", "window", "letter", "box", "lamp", "garden", "river",
         "bridge", "tower", "road", "car", "ship", "book", "chair", "wall", "stone", "tree", "house",
         "cat", "dog", "bird", "horse", "teacher", "doctor", "farmer", "singer", "child", "king",
         "soldier", "student", "painter", "pilot", "baker", "lawyer", "queen", "friend", "neighbor",
         "boat", "cup", "coat", "hat", "bell", "clock", "flower", "map", "ring", "shoe", "sword",
         "drum", "rope", "nurse", "judge", "thief", "guard", "poet", "sailor", "wolf"]
ADJS = ["old", "new", "red", "blue", "green", "small", "big", "dark", "bright", "quiet", "loud",
        "heavy", "light", "strange", "happy", "sad", "tall", "short", "warm", "cold", "young",
        "clever", "lazy", "brave", "gentle", "wild", "rich", "poor", "famous", "tiny", "proud",
        "calm", "eager", "fierce", "humble", "kind", "odd", "pale", "quick", "rough"]
V_INTR = ["sleep", "lie", "fall", "wait", "laugh", "shine", "vanish", "rest", "stay", "smile", "cry",
          "run", "swim", "dance", "sing", "glow", "work", "rise", "sit", "breathe", "shout", "pray",
          "dream", "listen", "travel", "whisper", "tremble", "wander", "yawn", "grow"]
V_TR = ["see", "find", "hold", "open", "break", "follow", "like", "watch", "carry", "paint", "move",
        "touch", "keep", "hide", "push", "clean", "build", "visit", "call", "help", "chase", "greet",
        "fear", "love", "pull", "lift", "draw", "guard", "praise", "blame"]
V_CC = ["think", "say", "know", "believe", "hope", "feel", "hear", "doubt", "claim", "suspect"]
PREPS = ["of", "near", "under", "behind", "with", "from", "beside", "above", "inside", "without",
         "below", "around"]
ADVS = ["often", "rarely", "always", "never", "quickly", "slowly", "surely", "maybe", "still",
        "soon", "again", "already", "almost", "quietly", "suddenly"]
NAMES = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi", "ivan", "judy",
         "mallory", "nina", "oscar", "peggy", "quinn", "rupert", "sybil", "trent", "ursula", "victor",
         "walter", "xena", "yusuf", "zoe", "amir", "bianca", "cyrus", "dora", "elias", "fiona",
         "gideon", "hana", "igor", "jade", "kofi", "lena", "milo", "nora", "omar", "pia", "rafael",
         "sara", "tomas", "uma", "vera", "wim", "yara", "zane", "anya", "boris", "clara", "dmitri",
         "edith", "felix", "greta", "hugo", "iris", "jonas", "kira", "leo", "mara", "nils", "olga",
         "pavel", "rosa", "sven", "tara", "ulf", "vika", "willa", "ada", "basil", "celia", "dario",
         "esme", "fritz", "gemma", "hal", "ines", "joel"]
DET_SG = ["a", "this", "every", "one"]
DET_PL = ["these", "many", "some", "few", "two"]
DET_ANY = ["the", "my", "our"]
CODA_MARKS = ["then", "so", "later", "and"]
ORDINALS = ["first", "second", "third", "fourth", "fifth"]
BRACKETS = [("[", "]"), ("{", "}"), ("<", ">"), ("|", "!")]
CLOSERS = [c for _, c in BRACKETS]
SG_NOUNS = set(NOUNS)


def build_vocab() -> Vocab:
    itos, cls = [], {}

    def add(w, c):
        if w not in cls:
            itos.append(w)
            cls[w] = c

    for w in ["<pad>", "<bos>"]:
        add(w, "special")
    for w in ["said", ":", ";", ".", ","]:
        add(w, "func")
    for o, c in BRACKETS:
        add(o, "open")
        add(c, "close")
    for w in ["who", "which", "whom"]:
        add(w, "rel")
    add("that", "comp")
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
    for m in CODA_MARKS + ORDINALS + ["and", "swap"]:
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


def flatten(root, vocab):
    return _flatten(root, vocab, sg_nouns=SG_NOUNS)


class Generator:
    """All random decisions are scored into self.lp (unambiguous grammar -> log p(sentence))."""

    def __init__(self, seed, p_mod_subj=0.8, p_mod_obj=0.45, p_mod_rcsubj=0.55, p_rc=0.65,
                 p_objrc=0.65, p_cc=0.35, p_coda=0.9, decay=0.93, max_emb=MAX_EMB):
        self.r = random.Random(seed)
        self.p_mod_subj, self.p_mod_obj, self.p_mod_rcsubj = p_mod_subj, p_mod_obj, p_mod_rcsubj
        self.p_rc, self.p_objrc, self.p_cc, self.p_coda = p_rc, p_objrc, p_cc, p_coda
        self.decay, self.max_emb = decay, max_emb
        self.lp = 0.0

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

    def sc(self, e):
        return self.decay ** e

    def NP(self, e, role, allow_name=True):
        if allow_name and self.flip(0.12):
            leaf = Leaf(self.ch(NAMES))
            n = Node("NP", [leaf], head=leaf)
            n.num, n.head_leaf, n.mod = "sg", leaf, 0
            return n
        num = self.ch(("sg", "pl"))
        kids = [Leaf(self.ch((DET_SG if num == "sg" else DET_PL) + DET_ANY))]
        for _ in range(self.cat((0, 1, 2), (0.55, 0.33, 0.12))):
            kids.append(Leaf(self.ch(ADJS)))
        base = self.ch(NOUNS)
        head = Leaf(base if num == "sg" else _plural(base))
        kids.append(head)
        node = Node("NP", kids, head=head)
        node.num, node.head_leaf, node.mod = num, head, 0
        p = {"subj": self.p_mod_subj, "rcsubj": self.p_mod_rcsubj}.get(role, self.p_mod_obj)
        if e < self.max_emb and self.flip(p * self.sc(e)):
            if self.flip(self.p_rc):
                node.children.append(self.RC(e + 1, num, head))
                node.mod = 2
            else:
                node.children.append(self.PP(e + 1))
                node.mod = 1
        return node

    def PP(self, e):
        p = Leaf(self.ch(PREPS))
        return Node("PP", [p, self.NP(e, "obj")], head=p)

    def RC(self, e, head_num, head_leaf):
        if self.flip(self.p_objrc):
            rel = Leaf("whom")
            subj = self.NP(e, "rcsubj", allow_name=True)
            v = self.ch(V_TR)
            vleaf = Leaf(_verb_sg(v) if subj.num == "sg" else v)
            vleaf.agree, vleaf.kind, vleaf._subj = 1, 4, subj.head_leaf
            return Node("RC", [rel, subj, Node("VP", [vleaf], head=vleaf)], head=rel)
        rel = Leaf(self.ch(("who", "which")))
        return Node("RC", [rel, self.VP(e, head_num, head_leaf, kind=3)], head=rel)

    def VP(self, e, num, subj_head, kind):
        p_cc = self.p_cc * self.sc(e) if e < self.max_emb else 0.0
        choice = self.cat(("cc", "intr", "tr"), (p_cc, (1 - p_cc) * 0.45, (1 - p_cc) * 0.55))
        if choice == "cc":
            v = self.ch(V_CC)
            vleaf = Leaf(_verb_sg(v) if num == "sg" else v)
            o, c = self.ch(BRACKETS)
            op, cl = Leaf(o), Leaf(c)
            cl.close = 1
            comp = Leaf("that")
            kids = [vleaf, Node("CP", [op, comp, self.CLAUSE(e + 1), cl], head=comp)]
        elif choice == "intr":
            v = self.ch(V_INTR)
            vleaf = Leaf(_verb_sg(v) if num == "sg" else v)
            kids = [vleaf]
            if e < self.max_emb and self.flip(0.35 * self.sc(e)):
                kids.append(self.PP(e + 1))
        else:
            v = self.ch(V_TR)
            vleaf = Leaf(_verb_sg(v) if num == "sg" else v)
            kids = [vleaf, self.NP(e, "obj")]
        vleaf.agree, vleaf.kind, vleaf._subj = 1, kind, subj_head
        return Node("VP", kids, head=vleaf)

    def CLAUSE(self, e):
        subj = self.NP(e, "subj")
        kids = [subj]
        if self.flip(0.2):
            kids.append(Leaf(self.ch(ADVS)))
        vp = self.VP(e, subj.num, subj.head_leaf, kind=subj.mod)
        kids.append(vp)
        return Node("CL", kids, head=vp.head)

    def sentence(self):
        self.lp = 0.0
        k = self.cat((1, 2, 3), (0.3, 0.35, 0.35))
        pool = list(NAMES)
        names = []
        for _ in range(k):
            nm = self.ch(pool)
            pool.remove(nm)
            names.append(nm)
        intro_kids = []
        for i, nm in enumerate(names):
            if i:
                intro_kids.append(Leaf(","))
            lf = Leaf(nm)
            intro_kids.append(Node("NP", [lf], head=lf))
        intro_kids += [Leaf("said"), Leaf(":")]
        kids = [Leaf("<bos>"), Node("INTRO", intro_kids), self.CLAUSE(0)]
        if self.flip(self.p_coda):
            j = self.ch(list(range(k)))
            c = Leaf(names[j])
            c.copy = k          # >0 marks the copy check; value = number of names to choose from
            v = Leaf(_verb_sg(self.ch(V_INTR)))
            kids.append(Node("CODA", [Leaf(";"), Leaf(self.ch(CODA_MARKS)), Leaf(ORDINALS[j]),
                                      Node("NP", [c], head=c), Node("VP", [v], head=v)]))
        kids.append(Leaf("."))
        return Node("S", kids)


# ----------------------------------------------------------------------------- v4: state tracking
def _leaves(node):
    if isinstance(node, Leaf):
        yield node
    else:
        for c in node.children:
            yield from _leaves(c)


class Generator4(Generator):
    """v3 syntax + a swap story:
        <bos> N_1 , ... , N_k said : CLAUSE ; [ORD_a and ORD_b swap ;]*m MARK ORD_q NAME V .
    NAME = the name standing at position q after the m swaps (deterministic -> 0 entropy).
    k ~ U(k_choices), m ~ U(0..max_swaps), pair ~ U(unordered pairs), q ~ U(k)."""

    def __init__(self, seed, k_choices=(3, 4), max_swaps=6, **kw):
        super().__init__(seed, **kw)
        self.k_choices, self.max_swaps = tuple(k_choices), max_swaps

    def sentence(self):
        self.lp = 0.0
        k = self.ch(self.k_choices)
        pool = list(NAMES)
        names = []
        for _ in range(k):
            nm = self.ch(pool)
            pool.remove(nm)
            names.append(nm)
        intro_kids = []
        for i, nm in enumerate(names):
            if i:
                intro_kids.append(Leaf(","))
            lf = Leaf(nm)
            intro_kids.append(Node("NP", [lf], head=lf))
        intro_kids += [Leaf("said"), Leaf(":")]
        clause = self.CLAUSE(0)
        for lf in _leaves(clause):
            lf.part = 1
        m = self.ch(list(range(self.max_swaps + 1)))
        state = list(names)
        pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
        story = [Leaf(";")]
        for e in range(1, m + 1):
            a, b = self.ch(pairs)
            state[a], state[b] = state[b], state[a]
            evl = [Leaf(ORDINALS[a]), Leaf("and"), Leaf(ORDINALS[b]), Leaf("swap"), Leaf(";")]
            for lf in evl:
                lf.ev = e
            story.append(Node("EV", evl))
        q = self.ch(list(range(k)))
        mark = Leaf(self.ch(CODA_MARKS))
        ordq = Leaf(ORDINALS[q])
        ordq.ansq = 1
        ans = Leaf(state[q])
        ans.copy = k
        v = Leaf(_verb_sg(self.ch(V_INTR)))
        story += [mark, ordq, Node("NP", [ans], head=ans), Node("VP", [v], head=v)]
        story_node = Node("CODA", story)
        for lf in _leaves(story_node):
            lf.part = 2
        root = Node("S", [Leaf("<bos>"), Node("INTRO", intro_kids), clause, story_node, Leaf(".")])
        root.nsw, root.nk = m, k
        return root


def flatten4(root, vocab):
    f, spans = _flatten(root, vocab, sg_nouns=SG_NOUNS)
    n = len(f["tok"])
    f["nsw"] = [root.nsw] * n
    f["nk"] = [root.nk] * n
    return f, spans


# ----------------------------------------------------------------------------- v5: several questions
def question_points(m):
    """After which swap counts a question is asked: 1st, 3rd and last swap (m = 0 -> one question)."""
    return [0] if m == 0 else sorted({x for x in (1, 3, m) if 1 <= x <= m})


class Generator5(Generator4):
    """v4 + denser hard tokens: questions after the 1st, 3rd and last swap.
        ... ; [ORD_a and ORD_b swap ;]  MARK ORD_q NAME ;  [swap ;] [swap ;]  MARK ORD_q NAME ; ... MARK ORD_q NAME V .
    Every answer NAME is deterministic (0 entropy).  qm = swaps before the question."""

    def sentence(self):
        self.lp = 0.0
        k = self.ch(self.k_choices)
        pool = list(NAMES)
        names = []
        for _ in range(k):
            nm = self.ch(pool)
            pool.remove(nm)
            names.append(nm)
        intro_kids = []
        for i, nm in enumerate(names):
            if i:
                intro_kids.append(Leaf(","))
            lf = Leaf(nm)
            intro_kids.append(Node("NP", [lf], head=lf))
        intro_kids += [Leaf("said"), Leaf(":")]
        clause = self.CLAUSE(0)
        for lf in _leaves(clause):
            lf.part = 1
        m = self.ch(list(range(self.max_swaps + 1)))
        qpts = question_points(m)
        state = list(names)
        pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
        story = [Leaf(";")]

        def question(qi, done, last):
            q = self.ch(list(range(k)))
            mark = Leaf(self.ch(CODA_MARKS))
            ordq = Leaf(ORDINALS[q])
            ordq.ansq, ordq.qm, ordq.qi = 1, done, qi
            ans = Leaf(state[q])
            ans.copy, ans.qm, ans.qi = k, done, qi
            kids = [mark, ordq, Node("NP", [ans], head=ans)]
            if last:
                v = Leaf(_verb_sg(self.ch(V_INTR)))
                kids.append(Node("VP", [v], head=v))
            else:
                kids.append(Leaf(";"))
            return Node("Q", kids)

        qi = 0
        if 0 in qpts:
            story.append(question(qi, 0, last=True))
        for e in range(1, m + 1):
            a, b = self.ch(pairs)
            state[a], state[b] = state[b], state[a]
            evl = [Leaf(ORDINALS[a]), Leaf("and"), Leaf(ORDINALS[b]), Leaf("swap"), Leaf(";")]
            for lf in evl:
                lf.ev = e
            story.append(Node("EV", evl))
            if e in qpts:
                story.append(question(qi, e, last=(e == m)))
                qi += 1
        story_node = Node("CODA", story)
        for lf in _leaves(story_node):
            lf.part = 2
        root = Node("S", [Leaf("<bos>"), Node("INTRO", intro_kids), clause, story_node, Leaf(".")])
        root.nsw, root.nk = m, k
        return root
