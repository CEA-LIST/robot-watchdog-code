"""Run every (method, task, seed) of the paper benchmark sequentially.

    uv run python scripts/run_benchmark.py                                  # everything
    uv run python scripts/run_benchmark.py --methods TPC --seeds 1 2 3       # a subset

Runs whose metrics already exist in <output_dir>/seed_<s>/metrics/ are skipped (use --overwrite to
rerun them). Logs are written to <output_dir>/logs/.
"""

import argparse
import os
import subprocess
import sys

TASKS = [
    "domotic_setTheTable",
    "domotic_dishTidyUp",
    "domotic_vegetablesAndFruitsSorting",
    "domotic_groceriesSorting",
    "aloha_setTheTable_2",
    "sort_trash_conveyor",
]
METHODS = ["TPC", "gnn_transformer_3", "RMS", "Frechet", "DTW"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("overrides", nargs="*", help="extra Hydra overrides, e.g. data_dir=/path/to/data")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    failed = []
    for seed in args.seeds:
        for method in args.methods:
            for task in args.tasks:
                metrics = os.path.join(args.output_dir, f"seed_{seed}", "metrics", f"{method}_{task}.json")
                if os.path.exists(metrics) and not args.overwrite:
                    print(f"skip  {method:28s} {task:36s} seed {seed}")
                    continue
                log = os.path.join(args.output_dir, "logs", f"{method}_{task}_seed{seed}.log")
                cmd = [
                    sys.executable, "-m", "robot_watchdog.run",
                    f"method={method}", f"task={task}", f"seed={seed}", f"output_dir={args.output_dir}",
                    *args.overrides,
                ]  # fmt: skip
                print(f"run   {method:28s} {task:36s} seed {seed}", flush=True)
                with open(log, "w") as f:
                    ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
                if ret != 0:
                    print(f"  !! failed (exit code {ret}), see {log}", flush=True)
                    failed.append((method, task, seed))

    if failed:
        print(f"\n{len(failed)} failed run(s):")
        for method, task, seed in failed:
            print(f"  {method} {task} seed {seed}")


if __name__ == "__main__":
    main()
