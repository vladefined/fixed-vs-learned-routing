"""Round 6, part 1: what slows QG-split down.  All runs from scratch, 120k steps, LR cosine over 120k, cost from the
first step (target 1.0 -> 0.25 over the first 5k steps, as QG25 in round 5), 2 seeds, checkpoints every 8k.
  QG25           shared FFN, full W_O                      (reference)
  QGsplit25      FFN cut into 8 blocks + head h -> group h (round 5)
  QGsplitFFN25   FFN cut into 8 blocks, full W_O
  QGsplitHead25  shared FFN, head h -> group h
Each job: train -> dd.evaluate5 -> dd.evaluate_r4.  Output runs8/, logs/r6_<run>*.txt.
python run_r6.py [--workers 4] [--only QG25_s0,...]"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

PY = sys.executable
STEPS = 120000
START = ["--sched", "ramp", "--target_frac_steps", str(5000 / STEPS), "--target", "0.25"]
SPECS = [("QG25", "QG"), ("QGsplit25", "QGsplit"), ("QGsplitFFN25", "QGsplitFFN"), ("QGsplitHead25", "QGsplitHead")]


def run_job(a, rn, var, seed):
    run = os.path.join(a.out, rn)
    t = time.time()
    if not os.path.exists(os.path.join(run, "model.pt")):
        cmd = [PY, "-m", "dd.train", "--variant", var, "--seed", str(seed), "--steps", str(STEPS),
               "--eval_every", "2000", "--ckpt_every", "8000", "--name", rn, "--out", a.out, "--data", a.data] + START
        with open(os.path.join("logs", f"r6_{rn}.txt"), "w") as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True)
        print(f"{rn}: trained in {(time.time() - t) / 60:.1f} min", flush=True)
    t = time.time()
    for fn, mod in (("eval5.json", "dd.evaluate5"), ("eval_r4.json", "dd.evaluate_r4")):
        if not os.path.exists(os.path.join(run, fn)):
            with open(os.path.join("logs", f"r6_{rn}_{fn.split('.')[0]}.txt"), "w") as fh:
                subprocess.run([PY, "-m", mod, run, a.data], stdout=fh, stderr=subprocess.STDOUT, check=True)
    print(f"{rn}: evaluated in {(time.time() - t) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default="runs8")
    ap.add_argument("--data", default="data5")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    only = set(a.only.split(",")) if a.only else None
    # seed-major order: both seeds of every variant are in flight before any variant gets a second wave
    js = [(f"{n}_s{s}", v, s) for s in (0, 1) for n, v in SPECS if not only or f"{n}_s{s}" in only]
    print("jobs:", [j[0] for j in js], flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_job, a, *j) for j in js]
        for f, j in zip(futs, js):
            try:
                f.result()
            except Exception as e:                       # keep the rest of the grid going
                print(f"{j[0]}: FAILED {e}", flush=True)
    print("R6 DONE", flush=True)


if __name__ == "__main__":
    main()
