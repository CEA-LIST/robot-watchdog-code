"""Aggregate the per-seed metrics into the tables of the paper.

    uv run python scripts/collect_results.py [--output_dir outputs]

Reads <output_dir>/seed_*/metrics/<method>_<task>.json and writes to <output_dir>/results/:
    threshold_independent.csv : AUPR, max MCC and AUROC per task (Fig. 3, Table 12)
    end_to_end.csv            : calibrated MCC, TPR and TNR per task (Fig. 5, Table 13), with the
                                modulation s1 ("variant1") averaged over alpha in [0.80, 0.95]
Values are mean and standard deviation over seeds. Dataset averages are computed per seed over the
tasks of the dataset, then averaged over seeds.
"""

import argparse
import glob
import json
import os

import pandas as pd

DATASETS = {
    "BotFails": [
        "domotic_setTheTable",
        "domotic_dishTidyUp",
        "domotic_vegetablesAndFruitsSorting",
        "domotic_groceriesSorting",
    ],
    "Watchdog": ["aloha_setTheTable_2", "sort_trash_conveyor"],
}
TASK_NAMES = {
    "domotic_setTheTable": "Table-setting (BotFails)",
    "domotic_dishTidyUp": "Dish storing",
    "domotic_vegetablesAndFruitsSorting": "Veg. sorting",
    "domotic_groceriesSorting": "Groc. sorting",
    "aloha_setTheTable_2": "Table-setting (Watchdog)",
    "sort_trash_conveyor": "Waste sorting",
}
METHOD_NAMES = {
    "RMS": "AP-RMS",
    "Frechet": "AP-Frechet",
    "DTW": "Soft-DTW",
    "TPC": "TPC (ours)",
    "gnn_transformer_3": "GnnT (ours)",
    "gnn_transformer": "gnn_transformer",
    "gnn_transformer_PE_global": "gnn_transformer_PE_global",
}


def load_rows(output_dir, variant, alpha_range):
    lo, hi = alpha_range
    cont_rows, e2e_rows = [], []
    for path in sorted(glob.glob(os.path.join(output_dir, "seed_*", "metrics", "*.json"))):
        with open(path) as f:
            m = json.load(f)
        key = dict(method=m["method"], task=m["task"], seed=m["seed"])
        cont_rows.append({**key, **{k: m["continuous"][k] for k in ("aupr", "max_mcc", "auroc")}})
        sel = [r for r in m["conformal"] if r["scal_variant"] == variant and lo - 1e-9 <= r["alpha"] <= hi + 1e-9]
        e2e_rows.append({**key, **{k: sum(r[k] for r in sel) / len(sel) for k in ("MCC", "TPR", "TNR")}})
    return pd.DataFrame(cont_rows), pd.DataFrame(e2e_rows)


def aggregate(df, metrics):
    """mean / std over seeds per (method, task), plus the dataset averages."""
    per_task = df.groupby(["method", "task"])[metrics].agg(["mean", "std", "count"])
    rows = [per_task]
    for dataset, tasks in DATASETS.items():
        sub = df[df["task"].isin(tasks)]
        # a seed contributes to the average only if all the tasks of the dataset are available
        complete = sub.groupby(["method", "seed"])["task"].nunique() == len(tasks)
        sub = sub.set_index(["method", "seed"]).loc[complete[complete].index].reset_index()
        if sub.empty:
            continue
        per_seed = sub.groupby(["method", "seed"])[metrics].mean().reset_index()
        avg = per_seed.groupby("method")[metrics].agg(["mean", "std", "count"])
        avg["task"] = f"{dataset} average"
        rows.append(avg.reset_index().set_index(["method", "task"]))
    return pd.concat(rows).sort_index()


def print_table(agg, metric, title):
    table = agg[metric].apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}" if r["count"] > 1 else f"{r['mean']:.3f}", axis=1)
    table = table.unstack("task")
    columns = [t for t in list(TASK_NAMES) + [f"{d} average" for d in DATASETS] if t in table.columns]
    table = table[columns].rename(columns=TASK_NAMES, index=METHOD_NAMES)
    print(f"\n### {title}\n")
    print(table.to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--variant", default="variant1", help="conformal modulation: variant1 (s1) or variant2 (s2)")
    parser.add_argument("--alpha_range", nargs=2, type=float, default=[0.80, 0.95])
    args = parser.parse_args()

    cont, e2e = load_rows(args.output_dir, args.variant, args.alpha_range)
    if cont.empty:
        print(f"no metrics found in {args.output_dir}/seed_*/metrics/")
        return

    out = os.path.join(args.output_dir, "results")
    os.makedirs(out, exist_ok=True)
    cont_agg = aggregate(cont, ["aupr", "max_mcc", "auroc"])
    e2e_agg = aggregate(e2e, ["MCC", "TPR", "TNR"])
    cont_agg.to_csv(os.path.join(out, "threshold_independent.csv"))
    e2e_agg.to_csv(os.path.join(out, "end_to_end.csv"))

    print_table(cont_agg, "aupr", "AUPR")
    print_table(cont_agg, "max_mcc", "Maximum MCC")
    print_table(cont_agg, "auroc", "AUROC")
    lo, hi = args.alpha_range
    print_table(e2e_agg, "MCC", f"Calibrated MCC ({args.variant}, alpha in [{lo:.2f}, {hi:.2f}])")
    print_table(e2e_agg, "TPR", "TPR")
    print_table(e2e_agg, "TNR", "TNR")
    print(f"\ntables saved in {out}/")


if __name__ == "__main__":
    main()
