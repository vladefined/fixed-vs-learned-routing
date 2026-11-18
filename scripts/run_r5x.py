"""Round 5, extension: QG50_s1 and QGsplit25_s0 continued from their 80k checkpoints to 120k (user decision).
Weights and mu from runs7/<run>/ckpt_080000.pt, data stream fast-forwarded to step 80k, LR held at the floor of
the original 80k cosine (0.1 lr) after a 500-step re-warmup (AdamW state is not in the checkpoints), cost target
unchanged.  Output: runs7x/<run> (train -> dd.evaluate5 -> dd.evaluate_r4), logs/r5x_<run>*.txt.
python run_r5x.py"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

PY = sys.executable
JOBS = [("QG50_s1", "QG", 1, ["--target", "0.5"]),
        ("QGsplit25_s0", "QGsplit", 0, ["--target", "0.25"])]
COMMON = ["--sched", "ramp", "--target_frac_steps", str(5000 / 80000), "--steps", "120000", "--sched_steps", "80000",
          "--eval_every", "2000", "--ckpt_every", "8000", "--out", "runs7x", "--data", "data5"]


def job(rn, var, seed, extra):
    run = os.path.join("runs7x", rn)
    t = time.time()
    if not os.path.exists(os.path.join(run, "model.pt")):
        cmd = [PY, "-m", "dd.train", "--variant", var, "--seed", str(seed), "--name", rn,
               "--resume", os.path.join("runs7", rn, "ckpt_080000.pt")] + COMMON + extra
        with open(os.path.join("logs", f"r5x_{rn}.txt"), "w") as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True)
        print(f"{rn}: trained 80k->120k in {(time.time() - t) / 60:.1f} min", flush=True)
    t = time.time()
    for fn, mod in (("eval5.json", "dd.evaluate5"), ("eval_r4.json", "dd.evaluate_r4")):
        if not os.path.exists(os.path.join(run, fn)):
            with open(os.path.join("logs", f"r5x_{rn}_{fn.split('.')[0]}.txt"), "w") as fh:
                subprocess.run([PY, "-m", mod, run, "data5"], stdout=fh, stderr=subprocess.STDOUT, check=True)
    print(f"{rn}: evaluated in {(time.time() - t) / 60:.1f} min", flush=True)


def safe(j):
    try:
        job(*j)
    except Exception as e:                      # noqa: BLE001
        print(f"{j[0]}: FAILED {e!r}", flush=True)


if __name__ == "__main__":
    os.makedirs("runs7x", exist_ok=True)
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(safe, JOBS))
    print("R5X DONE", flush=True)
