"""Round-4 grid, in priority order: Q (Q0, Q-L1 weak/strong, Q-L0) -> I' -> K-STE -> +free.
2 seeds, 80k steps, checkpoints every 8k.  Each job: train -> dd.evaluate5 -> dd.evaluate_r4 (+ dd.tails for I').
python run_r4.py --lam_Ip <lam> [--workers 2] [--only Q0_s0,...] [--prio 1,2]"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

PY = sys.executable
LATE = ["--sched", "late", "--hold_frac", "0.3", "--end_frac", "0.6"]


def specs(a):
    """(priority, name, variant, extra train args)"""
    return [
        (1, "Q0", "Q0", []),
        (1, "QL1w", "QL1", ["--target", "0.25"] + LATE),
        (1, "QL1s", "QL1", ["--target", "0.1"] + LATE),
        (1, "QL0", "QL0", ["--target", "0.25"] + LATE),
        (2, "Ip", "Ip", ["--lam", str(a.lam_Ip)]),
        (3, "K25STE", "KSTE", ["--target", "0.25"] + LATE),
        (3, "K50STE", "KSTE", ["--target", "0.5"] + LATE),
        (4, "A_groupnorm_free", "A_groupnorm_free", []),
        (4, "E_free", "E_free", ["--lam", "0.01"]),
        (4, "G_free", "G_free", ["--lam", "0.01"]),
    ]


def jobs(a):
    out = []
    prios = {int(p) for p in a.prio.split(",")}
    only = set(a.only.split(",")) if a.only else None
    for p in sorted(prios):
        for seed in (0, 1):
            for pr, name, var, extra in specs(a):
                if pr != p:
                    continue
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
        with open(os.path.join("logs", f"r4_{rn}.txt"), "w") as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True)
        print(f"{rn}: trained in {(time.time() - t) / 60:.1f} min", flush=True)
    t = time.time()
    steps = [("eval5.json", ["dd.evaluate5", run, a.data]), ("eval_r4.json", ["dd.evaluate_r4", run, a.data])]
    if var == "Ip":
        steps.append(("tails.json", ["dd.tails", run]))
    for fn, mod in steps:
        if not os.path.exists(os.path.join(run, fn)):
            with open(os.path.join("logs", f"r4_{rn}_{fn.split('.')[0]}.txt"), "w") as fh:
                subprocess.run([PY, "-m"] + mod, stdout=fh, stderr=subprocess.STDOUT, check=True)
    print(f"{rn}: evaluated in {(time.time() - t) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--lam_Ip", type=float, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--prio", default="1,2,3,4")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="runs6")
    ap.add_argument("--data", default="data5")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    js = jobs(a)
    if any(v == "Ip" for _, v, _, _ in js) and a.lam_Ip is None:
        raise SystemExit("--lam_Ip not set")
    print("jobs:", [j[0] for j in js], flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_job, a, *j) for j in js]
        for f, j in zip(futs, js):
            try:
                f.result()
            except Exception as e:                       # keep the rest of the grid going
                print(f"{j[0]}: FAILED {e}", flush=True)
    print("R4 DONE", flush=True)


if __name__ == "__main__":
    main()
