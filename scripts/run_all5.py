"""Round-3 grid (language v5, group variants), sequential train + evaluate5, skipping finished runs.
python run_all5.py --lam 0.01 --lam_I <lam> [--only A_s0,A_s1] [--steps 80000]"""
import argparse
import os
import subprocess
import sys
import time

PY = sys.executable


def grid(a):
    g = []
    for s in (0, 1, 2):
        g.append((f"A_s{s}", "A", 0.0, s))
    order3 = [("E", a.lam), ("G", a.lam), ("A_groupnorm", 0.0), ("A_norm", 0.0)]
    order2 = [("H", a.lam), ("I", a.lam_I), ("J", a.lam_I)]
    for s in (0, 1, 2):
        for v, lam in order3:
            g.append((f"{v}_s{s}", v, lam, s))
        if s < 2:
            for v, lam in order2:
                g.append((f"{v}_s{s}", v, lam, s))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--lam_I", type=float, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default="runs5")
    ap.add_argument("--data", default="data5")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    for name, var, lam, seed in grid(a):
        if only and name not in only:
            continue
        if lam is None:
            raise SystemExit(f"{name}: --lam_I not set")
        run = os.path.join(a.out, name)
        if not os.path.exists(os.path.join(run, "model.pt")):
            t = time.time()
            cmd = [PY, "-m", "dd.train", "--variant", var, "--lam", str(lam), "--seed", str(seed),
                   "--steps", str(a.steps), "--eval_every", "2000", "--name", name, "--out", a.out,
                   "--data", a.data]
            with open(os.path.join("logs", f"r3_{name}.txt"), "w") as fh:
                subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True)
            print(f"{name}: trained in {(time.time() - t) / 60:.1f} min", flush=True)
        if not os.path.exists(os.path.join(run, "eval5.json")):
            t = time.time()
            with open(os.path.join("logs", f"r3_{name}_eval.txt"), "w") as fh:
                subprocess.run([PY, "-m", "dd.evaluate5", run, a.data], stdout=fh, stderr=subprocess.STDOUT,
                               check=True)
            print(f"{name}: evaluated in {(time.time() - t) / 60:.1f} min", flush=True)
    print("R3 DONE", flush=True)


if __name__ == "__main__":
    main()
