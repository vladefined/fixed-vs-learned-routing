"""Round 6, part 2 driver (TinyStories, d=512, 8 blocks, 8 heads, ctx 512, 20k steps x 32 x 512 = 328M tokens).
  1. dd.ts_data (BPE 8192 + token streams), once
  2. A_s0 -> dd.ts_eval calib -> grid from A's state scale.
        --grid profile (default): one step per state, delta_l = RMS of A's state l / 3.2 (state 0 = embedding,
            l + 1 = output of block l), levels -K..K with K = --levels (default 8, i.e. 17 levels as in round 4).
            With --emb_scale s the embedding is multiplied by s before quantisation and delta_0 by s.
        --grid shared: one step for all states (round-4 rule: delta = mean inner RMS / 3.2,
            K = max(8, ceil(max inner q99 / delta)); --delta / --levels override).  Not used: see REPORT 10.4.
  3. Q0_s0, QG25_s0, <best>25_s0, QG25_s1 with that grid (one at a time: d=512 saturates the GPU)
  4. dd.ts_eval for A (per-token reference) and every run; dd.ts_speed for A, QG25_s0, <best>25_s0
python run_r6ts.py --best QGsplit [--grid profile --levels 8 --emb_scale 1]"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

PY = sys.executable
OUT = "runs_ts"


def sh(args, log):
    t = time.time()
    with open(os.path.join("logs", log), "w") as fh:
        subprocess.run([PY, "-m"] + args, stdout=fh, stderr=subprocess.STDOUT, check=True)
    print(f"{log}: done in {(time.time() - t) / 60:.1f} min", flush=True)


def train(name, variant, seed, extra):
    if not os.path.exists(os.path.join(OUT, name, "model.pt")):
        sh(["dd.ts_train", "--variant", variant, "--seed", str(seed), "--name", name, "--out", OUT] + extra,
           f"r6ts_{name}.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", required=True, help="variant name of the best split from part 1 (QGsplit|QGsplitFFN|QGsplitHead)")
    ap.add_argument("--grid", default="profile", choices=["profile", "shared"])
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--levels", type=int, default=None)
    ap.add_argument("--emb_scale", type=float, default=1.0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(os.path.join("data_ts", "meta.json")):
        sh(["dd.ts_data", "data_ts", "8192"], "r6ts_data.txt")
    train("A_s0", "A", 0, [])
    if not os.path.exists(os.path.join(OUT, "A_s0", "calib.json")):
        sh(["dd.ts_eval", "calib", os.path.join(OUT, "A_s0")], "r6ts_calib.txt")
    c = json.load(open(os.path.join(OUT, "A_s0", "calib.json")))
    if a.grid == "profile":
        K = a.levels or 8
        prof = [float(f"{st['rms'] / 3.2:.3g}") for st in c["per_state"]]
        prof[0] = float(f"{prof[0] * a.emb_scale:.3g}")
        grid = ["--q_delta_layers", ",".join(map(str, prof)), "--q_levels", str(K)]
        if a.emb_scale != 1.0:
            grid += ["--emb_scale", str(a.emb_scale)]
        info = dict(scheme="profile", delta_layers=prof, K=K, emb_scale=a.emb_scale)
    else:
        delta = a.delta or float(f"{c['inner_rms_mean'] / 3.2:.3g}")
        K = a.levels or max(8, math.ceil(c["inner_q99_max"] / delta))
        grid = ["--q_delta", str(delta), "--q_levels", str(K)]
        info = dict(scheme="shared", delta=delta, K=K)
    print(f"grid: {info}", flush=True)
    json.dump(dict(**info, calib=c), open(os.path.join(OUT, "grid.json"), "w"), indent=1)
    qg = ["--target", "0.25"]
    jobs = [("Q0_s0", "Q0", 0, grid), ("QG25_s0", "QG", 0, grid + qg), (f"{a.best}25_s0", a.best, 0, grid + qg),
            ("QG25_s1", "QG", 1, grid + qg)]                     # second QG25 seed last ("if time allows")
    for name, var, seed, extra in jobs:
        try:
            train(name, var, seed, extra)
        except subprocess.CalledProcessError as e:
            print(f"{name}: FAILED {e}", flush=True)
    for name in ["A_s0"] + [j[0] for j in jobs]:
        run = os.path.join(OUT, name)
        if os.path.exists(os.path.join(run, "model.pt")) and not os.path.exists(os.path.join(run, "ts_eval.json")):
            try:
                sh(["dd.ts_eval", "eval", run] + ([] if name == "A_s0" else [os.path.join(OUT, "A_s0")]),
                   f"r6ts_{name}_eval.txt")
            except subprocess.CalledProcessError as e:
                print(f"{name} eval: FAILED {e}", flush=True)
    try:
        sh(["dd.ts_speed", os.path.join(OUT, "A_s0"), os.path.join(OUT, "QG25_s0"), os.path.join(OUT, f"{a.best}25_s0")],
           "r6ts_speed.txt")
    except subprocess.CalledProcessError as e:
        print(f"speed: FAILED {e}", flush=True)
    print("R6TS DONE", flush=True)


if __name__ == "__main__":
    main()
