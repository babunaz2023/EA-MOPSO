"""
Run Hybrid EA-MOPSO on standard benchmark problems from pymoo.

Examples
--------
    python run_example.py                          # ZDT1, 1 run
    python run_example.py --problem dtlz2 --runs 10
    python run_example.py --problem all --runs 5   # ZDT1, ZDT2, ZDT3, DTLZ2

Outputs (in --out, default "results/"):
    <problem>_pareto_front.csv   final non-dominated solutions (last run)
    <problem>_pareto_front.png   obtained front vs. true Pareto front
    summary.txt                  HV / IGD mean and std for every problem
"""

import argparse
import os
import time

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.problems import get_problem

from ea_mopso import EAMOPSO

# name -> keyword arguments for pymoo.problems.get_problem
PROBLEMS = {
    "zdt1": {},                          # 30 variables, 2 objectives
    "zdt2": {},
    "zdt3": {},
    "dtlz2": {"n_var": 7, "n_obj": 2},   # 7 variables, 2 objectives
}
REF_POINT = np.array([1.1, 1.1])


def run_problem(name, args):
    problem = get_problem(name, **PROBLEMS.get(name, {}))
    true_pf = problem.pareto_front()

    def func(X):
        return problem.evaluate(X)

    hv = HV(ref_point=REF_POINT)
    igd = IGD(true_pf)
    hv_true = hv(true_pf)

    hvs, igds, times = [], [], []
    for r in range(args.runs):
        seed = None if args.seed is None else args.seed + r
        t0 = time.time()
        opt = EAMOPSO(func, problem.xl, problem.xu,
                      n_particles=args.pop, max_iter=args.iters,
                      archive_size=args.archive, seed=seed)
        pareto_x, pareto_f = opt.run()
        times.append(time.time() - t0)
        hvs.append(hv(pareto_f) / hv_true)       # normalised hypervolume
        igds.append(igd(pareto_f))
        print(f"  run {r + 1:2d}: solutions={len(pareto_f):3d}  "
              f"HV={hvs[-1]:.4f}  IGD={igds[-1]:.4f}  time={times[-1]:.1f}s")

    # save last run
    os.makedirs(args.out, exist_ok=True)
    order = np.argsort(pareto_f[:, 0])
    header = ",".join([f"x{i + 1}" for i in range(pareto_x.shape[1])] +
                      [f"f{j + 1}" for j in range(pareto_f.shape[1])])
    np.savetxt(os.path.join(args.out, f"{name}_pareto_front.csv"),
               np.c_[pareto_x[order], pareto_f[order]], delimiter=",",
               header=header, comments="", fmt="%.6f")

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 4))
        plt.plot(true_pf[:, 0], true_pf[:, 1], ".", ms=3, color="0.65", label="True Pareto front")
        plt.scatter(pareto_f[:, 0], pareto_f[:, 1], s=18, color="tab:red", label="Hybrid EA-MOPSO")
        plt.xlabel("$f_1$")
        plt.ylabel("$f_2$")
        plt.title(f"{name.upper()}  (N={args.pop}, {args.iters} iterations)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"{name}_pareto_front.png"), dpi=150)
        plt.close()

    return {"problem": name, "n_var": problem.n_var, "hv": hvs, "igd": igds, "time": times}


def main():
    p = argparse.ArgumentParser(description="Hybrid EA-MOPSO on pymoo benchmark problems")
    p.add_argument("--problem", default="zdt1",
                   help="zdt1, zdt2, zdt3, dtlz2, all, or any 2-objective pymoo problem name")
    p.add_argument("--pop", type=int, default=10, help="number of particles (default 10)")
    p.add_argument("--iters", type=int, default=100, help="iterations (default 100)")
    p.add_argument("--archive", type=int, default=100, help="external archive size")
    p.add_argument("--runs", type=int, default=1, help="independent runs")
    p.add_argument("--seed", type=int, default=1, help="base random seed")
    p.add_argument("--out", default="results", help="output folder")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    names = list(PROBLEMS) if args.problem == "all" else [args.problem.lower()]
    rows = []
    for name in names:
        print(f"\n{name.upper()}")
        rows.append(run_problem(name, args))

    lines = [
        "Hybrid EA-MOPSO",
        f"particles={args.pop}  iterations={args.iters}  runs={args.runs}  "
        f"seed={args.seed}  HV reference point={REF_POINT.tolist()}",
        "HV is normalised by the hypervolume of the true Pareto front (1.0 = perfect).",
        "",
        f"{'Problem':<8}{'n_var':>6}{'HV mean':>11}{'HV std':>10}{'IGD mean':>11}{'IGD std':>10}{'time/run':>10}",
    ]
    for r in rows:
        lines.append(f"{r['problem'].upper():<8}{r['n_var']:>6}"
                     f"{np.mean(r['hv']):>11.4f}{np.std(r['hv']):>10.4f}"
                     f"{np.mean(r['igd']):>11.4f}{np.std(r['igd']):>10.4f}"
                     f"{np.mean(r['time']):>9.1f}s")
    text = "\n".join(lines)
    print("\n" + text)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
