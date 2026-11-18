"""Round-5 grid.
P1: Q-L0-start (cost from step 0: target 1.0 -> 0.25 over the first 5k steps), extra seeds Q0 s2-s4, A s3-s4.
P2: QG25, QG50, QGsplit25, QGnogate25 with the cost schedule chosen from P1 (--p2_sched start|late).
2 seeds for new variants, 80k steps, checkpoints every 8k.  Each job: train -> dd.evaluate5 -> dd.evaluate_r4.
python run_r5.py --prio 1 [--workers 3] [--only ...]
python run_r5.py --prio 2 --p2_sched start|late"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

PY = sys.executable
LATE = ["--sched", "late", "--hold_frac", "0.3", "--end_frac", "0.6"]
START = ["--sched", "ramp", "--target_frac_steps", str(5000 / 80000)]


def specs(a):
    """(priority, name, variant, seeds, extra train args)"""
    p2 = START if a.p2_sched == "start" else LATE
    return [
        (1, "QL0start", "QL0", (0, 1), ["--target", "0.25"] + START),
        (1, "Q0", "Q0", (2, 3, 4), []),
        (1, "A", "A", (3, 4), []),
        (2, "QG25", "QG", (0, 1), ["--target", "0.25"] + p2),
        (2, "QG50", "QG", (0, 1), ["--target", "0.5"] + p2),
        (2, "QGsplit25", "QGsplit", (0, 1), ["--target", "0.25"] + p2),
        (2, "QGnogate25", "QGnogate", (0, 1), ["--target", "0.25"] + p2),
    ]


def jobs(a):
    out = []
    prios = {int(p) for p in a.prio.split(",")}
    only = set(a.only.split(",")) if a.only else None
    for p in sorted(prios):
        for pr, name, var, seeds, extra in specs(a):
            if pr != p:
                continue
            for seed in seeds:
                rn = f"{name}_s{seed}"
                if only and rn not in only:
                    continue
                out.append((rn, var, seed, extra))
    return out


def run_job(a, rn, var, seed, extra):
    run = os.path.join(a.out, rn)
    t = time.time()
    if not os.path.exists(os.path.join(run, "model.pt")):
        cmd = [PY, "-m", "dd.train", "--variant", var, "--seed", str(seed), "--steps", str(a.steps),
               "--eval_every", "2000", "--ckpt_every", "8000", "--name", rn, "--out", a.out, "--data", a.data] + extra
        with open(os.path.join("logs", f"r5_{rn}.txt"), "w") as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True)
        print(f"{rn}: trained in {(time.time() - t) / 60:.1f} min", flush=True)
    t = time.time()
    steps = [("eval5.json", ["dd.evaluate5", run, a.data]), ("eval_r4.json", ["dd.evaluate_r4", run, a.data])]
    for fn, mod in steps:
        if not os.path.exists(os.path.join(run, fn)):
            with open(os.path.join("logs", f"r5_{rn}_{fn.split('.')[0]}.txt"), "w") as fh:
                subprocess.run([PY, "-m"] + mod, stdout=fh, stderr=subprocess.STDOUT, check=True)
    print(f"{rn}: evaluated in {(time.time() - t) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--only", default=None)
    ap.add_argument("--prio", default="1")
    ap.add_argument("--p2_sched", default=None, choices=[None, "start", "late"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="runs7")
    ap.add_argument("--data", default="data5")
    ap.add_argument("--skip_eval", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    js = jobs(a)
    if any(j[1].startswith("QG") for j in js) and a.p2_sched is None:
        raise SystemExit("--p2_sched not set")
    print("jobs:", [j[0] for j in js], flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_job, a, *j) for j in js]
        for f, j in zip(futs, js):
            try:
                f.result()
            except Exception as e:                       # keep the rest of the grid going
                print(f"{j[0]}: FAILED {e}", flush=True)
    print("R5 DONE", flush=True)


if __name__ == "__main__":
    main()
