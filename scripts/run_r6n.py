"""Round 6, night block (TinyStories, same grid as part 2: runs_ts/grid.json; one GPU job at a time).
  1. quality vs savings frontier, 20k steps, 1 seed: QG50, QG-split50 (with generations), QG35, QG-split35
  2. longer training, 40k steps from scratch, cosine over 40k, val CE every 2k: A-40k, QG25-40k
     (+ QG-split25-40k only if it still ends before the deadline)
Each run: dd.ts_train -> dd.ts_eval (A_s0 = per-token difficulty reference).  The cost-target ramp stays 1250 steps.
python run_r6n.py [--deadline 10:00]"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time

PY = sys.executable
OUT = "runs_ts"
A_REF = os.path.join(OUT, "A_s0")


def sh(args, log):
    t = time.time()
    with open(os.path.join("logs", log), "w") as fh:
        subprocess.run([PY, "-m"] + args, stdout=fh, stderr=subprocess.STDOUT, check=True)
    print(f"{log}: done in {(time.time() - t) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline", default="10:00")
    ap.add_argument("--opt_minutes", type=float, default=95.0, help="expected train + eval time of the optional run")
    a = ap.parse_args()
    g = json.load(open(os.path.join(OUT, "grid.json")))
    assert g["scheme"] == "profile"
    grid = ["--q_delta_layers", ",".join(map(str, g["delta_layers"])), "--q_levels", str(g["K"])]
    if g.get("emb_scale", 1.0) != 1.0:
        grid += ["--emb_scale", str(g["emb_scale"])]
    long = ["--steps", "40000", "--eval_every", "2000", "--target_frac_steps", "0.03125"]
    # name, variant, extra args, generations, optional
    jobs = [("QG50_s0", "QG", grid + ["--target", "0.5"], True, False),
            ("QGsplit50_s0", "QGsplit", grid + ["--target", "0.5"], True, False),
            ("QG35_s0", "QG", grid + ["--target", "0.35"], False, False),
            ("QGsplit35_s0", "QGsplit", grid + ["--target", "0.35"], False, False),
            ("A40k_s0", "A", long, False, False),
            ("QG25_40k_s0", "QG", grid + long + ["--target", "0.25"], False, False),
            ("QGsplit25_40k_s0", "QGsplit", grid + long + ["--target", "0.25"], False, True)]
    hh, mm = map(int, a.deadline.split(":"))
    now = dt.datetime.now()
    deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if deadline < now:
        deadline += dt.timedelta(days=1)
    for name, var, extra, gen, optional in jobs:
        run = os.path.join(OUT, name)
        if optional and dt.datetime.now() + dt.timedelta(minutes=a.opt_minutes) > deadline:
            print(f"{name}: skipped (would end after {a.deadline})", flush=True)
            continue
        try:
            if not os.path.exists(os.path.join(run, "model.pt")):
                sh(["dd.ts_train", "--variant", var, "--seed", "0", "--name", name, "--out", OUT] + extra,
                   f"r6n_{name}.txt")
            if not os.path.exists(os.path.join(run, "ts_eval.json")):
                sh(["dd.ts_eval", "eval", run, A_REF] + ([] if gen else ["--nogen"]), f"r6n_{name}_eval.txt")
        except subprocess.CalledProcessError as e:
            print(f"{name}: FAILED {e}", flush=True)
    print("R6N DONE", flush=True)


if __name__ == "__main__":
    main()
