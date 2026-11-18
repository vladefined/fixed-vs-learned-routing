"""Round 7 (TinyStories, 20k steps, grid per layer as in round 6, 1 seed, one GPU job at a time).
The job list is r7_plan.json (written on the first start, re-read before every job, so it can be edited while
the block runs).  Each job: dd.ts_train -> dd.ts_eval (A_s0 = per-token difficulty reference, no generations).
  1. gate controls: QG25-random, QG25-static, QG25-position (mask not learned)
  2. grid: Q0 with a 2x and 4x finer step on the same range (161 and 321 levels)
  3. cost of the cut: A-split (QG-split cut, no gate, no grid), Q0-split (the cut + Q0 grid, no gate),
     QG-split25-wide (FFN widened to A's parameter count)
Random-mask controls (random, position) are evaluated with mask seeds 0, 1, 2 (dd.ts_eval)
python run_r7.py [--plan r7_plan.json]"""
import argparse
import json
import os
import subprocess
import sys
import time

PY = sys.executable
OUT = "runs_ts"
A_REF = os.path.join(OUT, "A_s0")


def default_plan():
    g = json.load(open(os.path.join(OUT, "grid.json")))
    assert g["scheme"] == "profile"
    dl, K, es = g["delta_layers"], g["K"], g["emb_scale"]

    def grid(f=1):
        return ["--q_delta_layers", ",".join(str(x / f) for x in dl), "--q_levels", str(K * f),
                "--emb_scale", str(es)] + (["--q_emb_init", str(float(f))] if f != 1 else [])

    return [dict(name="QG25random_s0", args=["--variant", "QGrandom", "--mask_p", "0.25"] + grid()),
            dict(name="QG25static_s0", args=["--variant", "QGstatic", "--mask_p", "0.25"] + grid()),
            dict(name="QG25position_s0", args=["--variant", "QGposition", "--pos_profile",
                                               os.path.join(OUT, "pos_profile_qg25.json")] + grid()),
            dict(name="Q0_161_s0", args=["--variant", "Q0"] + grid(2)),
            dict(name="Q0_321_s0", args=["--variant", "Q0"] + grid(4)),
            dict(name="Asplit_s0", args=["--variant", "Asplit"]),
            dict(name="Q0split_s0", args=["--variant", "Q0split"] + grid()),
            dict(name="QGsplit25wide_s0", args=["--variant", "QGsplit", "--target", "0.25", "--ffn_hidden", "2128"]
                 + grid())]


def sh(args, log):
    t = time.time()
    with open(os.path.join("logs", log), "w") as fh:
        subprocess.run([PY, "-m"] + args, stdout=fh, stderr=subprocess.STDOUT, check=True)
    print(f"{log}: done in {(time.time() - t) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="r7_plan.json")
    a = ap.parse_args()
    if not os.path.exists(a.plan):
        json.dump(default_plan(), open(a.plan, "w"), indent=1)
    failed = set()
    while True:
        plan = json.load(open(a.plan))                     # re-read: the plan may be edited while the block runs
        todo = [j for j in plan if j["name"] not in failed                 # done here or on vast.ai: skip
                and not any(os.path.exists(os.path.join(b, j["name"], "ts_eval.json")) for b in (OUT, "runs_vast"))]
        if not todo:
            break
        j = todo[0]
        run = os.path.join(OUT, j["name"])
        try:
            if not os.path.exists(os.path.join(run, "model.pt")):
                sh(["dd.ts_train", "--seed", "0", "--name", j["name"], "--out", OUT] + j["args"], f"r7_{j['name']}.txt")
            sh(["dd.ts_eval", "eval", run, A_REF, "--nogen"], f"r7_{j['name']}_eval.txt")
        except subprocess.CalledProcessError as e:
            print(f"{j['name']}: FAILED {e}", flush=True)
            failed.add(j["name"])
    print("R7 DONE", flush=True)


if __name__ == "__main__":
    main()
