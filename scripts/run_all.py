"""Run the experiment grid sequentially (train + evaluate), skipping finished runs.
python run_all.py --steps 40000 --lam_weak W --lam_mid M --lam_strong S [--only A_s0,B_m_s0]"""
import argparse
import os
import subprocess
import sys
import time

PY = sys.executable


def grid(a):
    w, m, s = a.lam_weak, a.lam_mid, a.lam_strong
    return [
        ("A_s0", "A", 0.0, 0), ("A_norm_s0", "A_norm", 0.0, 0),
        ("B_l0_s0", "B", 0.0, 0), ("B_m_s0", "B", m, 0), ("B_w_s0", "B", w, 0), ("B_s_s0", "B", s, 0),
        ("Bdd_m_s0", "B_dd", m, 0), ("Bpull_m_s0", "B_pull", m, 0),
        ("C_m_s0", "C", m, 0), ("E_m_s0", "E", m, 0), ("F_m_s0", "F", m, 0), ("G_m_s0", "G", m, 0),
        ("A_s1", "A", 0.0, 1), ("B_m_s1", "B", m, 1),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--lam_weak", type=float, required=True)
    ap.add_argument("--lam_mid", type=float, required=True)
    ap.add_argument("--lam_strong", type=float, required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default="runs4")
    ap.add_argument("--data", default="data4")
    a = ap.parse_args()
    os.makedirs("logs", exist_ok=True)
    only = set(a.only.split(",")) if a.only else None
    for name, var, lam, seed in grid(a):
        if only and name not in only:
            continue
        run = os.path.join(a.out, name)
        if not os.path.exists(os.path.join(run, "model.pt")):
            t = time.time()
            cmd = [PY, "-m", "dd.train", "--variant", var, "--lam", str(lam), "--seed", str(seed),
                   "--steps", str(a.steps), "--eval_every", "4000", "--name", name, "--out", a.out,
                   "--data", a.data]
            with open(os.path.join("logs", f"v4_{name}.txt"), "w") as fh:
                subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True)
            print(f"{name}: trained in {(time.time() - t) / 60:.1f} min", flush=True)
        if not os.path.exists(os.path.join(run, "eval.json")):
            t = time.time()
            with open(os.path.join("logs", f"v4_{name}_eval.txt"), "w") as fh:
                subprocess.run([PY, "-m", "dd.evaluate", run, a.data], stdout=fh,
                               stderr=subprocess.STDOUT, check=True)
            print(f"{name}: evaluated in {(time.time() - t) / 60:.1f} min", flush=True)
    print("GRID DONE", flush=True)


if __name__ == "__main__":
    main()
