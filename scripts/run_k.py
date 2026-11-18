"""Variant-K grid (after the round-3 grid): K25, K50, K25L, K25-free, K25-norm x 2 seeds, 80k steps.
python run_k.py [--only K25_s0,...]"""
import argparse
import os
import subprocess
import sys
import time

PY = sys.executable
SPECS = [("K25", "K", 0.25), ("K50", "K", 0.50), ("K25L", "KL", 0.25), ("K25free", "Kfree", 0.25),
         ("K25norm", "Knorm", 0.25)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default="runs5")
    ap.add_argument("--data", default="data5")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    for seed in (0, 1):
        for name, var, tgt in SPECS:
            run_name = f"{name}_s{seed}"
            if only and run_name not in only:
                continue
            run = os.path.join(a.out, run_name)
            if not os.path.exists(os.path.join(run, "model.pt")):
                t = time.time()
                cmd = [PY, "-m", "dd.train", "--variant", var, "--target", str(tgt), "--seed", str(seed),
                       "--steps", str(a.steps), "--eval_every", "2000", "--name", run_name, "--out", a.out,
                       "--data", a.data]
                with open(os.path.join("logs", f"rk_{run_name}.txt"), "w") as fh:
                    subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True)
                print(f"{run_name}: trained in {(time.time() - t) / 60:.1f} min", flush=True)
            if not os.path.exists(os.path.join(run, "evalk.json")):
                t = time.time()
                with open(os.path.join("logs", f"rk_{run_name}_eval.txt"), "w") as fh:
                    subprocess.run([PY, "-m", "dd.evaluate_k", run, a.data], stdout=fh,
                                   stderr=subprocess.STDOUT, check=True)
                print(f"{run_name}: evaluated in {(time.time() - t) / 60:.1f} min", flush=True)
    print("K DONE", flush=True)


if __name__ == "__main__":
    main()
