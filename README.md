# Robot-WATCHDOG: failure detection through object-centric graph representation

Official code of **Robot-WATCHDOG** (CoRL 2026).

[Quentin Rolland](https://quentinrolld.github.io/)<sup>1,2</sup>,
[Fabrice Mayran de Chamisso](https://fabricemdc.ocfmcreation.com/)<sup>1</sup>,
[Jean-Baptiste Mouret](https://members.loria.fr/JBMouret/)<sup>2,3</sup>

<sup>1</sup>Université Paris-Saclay, CEA, List, F-91120, Palaiseau, France,
<sup>2</sup>Inria, CNRS, Université de Lorraine, LORIA, F-54000 Nancy, France,
<sup>3</sup>Bleu Robotics, Paris, France

*Conference on Robot Learning (CoRL), 2026*

<a href='https://cea-list.github.io/Robot-WATCHDOG/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://arxiv.org/abs/2610.13403'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='LICENSE'><img src='https://img.shields.io/badge/Code-MIT-yellow'></a> <a href='https://creativecommons.org/licenses/by/4.0/'><img src='https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey'></a>

[Project page](https://cea-list.github.io/Robot-WATCHDOG/)

Robot-WATCHDOG detects failures of learned robotic policies from nominal demonstrations only. Each
manipulation episode is abstracted into a sequence of object-centric graphs (tracked objects and their
pairwise relations), on which we study two failure scores:

- **TPC** (Trajectory Projection and Correlation), non-parametric: distance of each object trajectory to
  the closest expert trajectory of the same label, with exponentially increasing temporal weights;
- **GnnT**, a GATv2 + Transformer model trained to predict the future object positions; the failure
  score is its prediction error.

Both scores are turned into online binary decisions by a time-varying conformal band calibrated on
held-out expert demonstrations.

This repository contains the code to reproduce the main results of the paper (Fig. 3 and 5, Tables 12
and 13) from the object trajectories extracted from the videos (VLM initialisation + SAM2 tracking).

## Installation

The environment is managed with [uv](https://docs.astral.sh/uv/) (Python 3.10, PyTorch 2.6):

```bash
git clone <this repository> robot-watchdog
cd robot-watchdog
uv sync
```

A CUDA GPU is recommended; set `device=cpu` otherwise.

## Data

The object trajectories and the per-frame failure labels are distributed as an asset of the
[latest release](../../releases/latest) (`robot_watchdog_data.zip`, 1.3 MB). Download it and unzip it
at the root of the repository:

```bash
gh release download --pattern robot_watchdog_data.zip   # or download it from the release page
unzip robot_watchdog_data.zip                            # creates data/
```

The archive contains the 6 tasks of the paper:

| Task (config name)                   | Dataset        | Paper name    | Expert demos | Test episodes | Objects |
| ------------------------------------ | -------------- | ------------- | -----------: | ------------: | ------: |
| `domotic_setTheTable`                | BotFails       | Table-setting |           17 |            14 |       4 |
| `domotic_dishTidyUp`                 | BotFails       | Dish storing  |           20 |            15 |       8 |
| `domotic_vegetablesAndFruitsSorting` | BotFails       | Veg. sorting  |           19 |            20 |       6 |
| `domotic_groceriesSorting`           | BotFails       | Groc. sorting |           20 |            10 |       8 |
| `aloha_setTheTable_2`                | Robot-WATCHDOG | Table-setting |           49 |            20 |       4 |
| `sort_trash_conveyor`                | Robot-WATCHDOG | Waste sorting |           79 |            42 |       2 |

with, for every task:

```
data/<task>/
├── expert.json     # nominal demonstrations
├── anomaly.json    # test episodes (nominal executions, benign variations and failures)
└── labels/         # one CSV per test episode (same order as anomaly.json), one value per frame (1 = failure)
```

An episode is a dict `{"1": {"label": str, "points": [[x, y], ...]}, "2": {...}, ..., "fine_labels": [...],
"coarse_labels": [...], "episode_idx": int}`, with one entry per tracked object (2D centroid of its SAM2
mask, in pixels). The episodes where the tracker lost an object were removed.

## Usage

Evaluate one method on one task for one seed:

```bash
uv run python -m robot_watchdog.run method=TPC task=sort_trash_conveyor seed=1
uv run python -m robot_watchdog.run method=gnn_transformer_3 task=aloha_setTheTable_2 seed=1
```

The configuration is handled by [Hydra](https://hydra.cc/) (`configs/`): any value can be overridden from
the command line (e.g. `method.decay=0.1`, `data_dir=/path/to/data`). A seed draws the split of the expert
demonstrations, the conformal split and the network initialisation. Each run writes to
`outputs/seed_<seed>/`:

- `scores/<method>_<task>.pkl`: raw failure score and binary decisions of every conformal configuration,
  at every step of every test episode;
- `metrics/<method>_<task>.json`: step-level metrics on all the test episodes of the task.

| Method (config name)        | Paper name | Type                                          |
| --------------------------- | ---------- | --------------------------------------------- |
| `TPC`                       | TPC        | non-parametric                                |
| `gnn_transformer_3`         | GnnT       | predictive (trained on the expert demos)      |
| `gnn_transformer`           | -          | predictive: GCN encoder + Transformer         |
| `gnn_transformer_PE_global` | -          | predictive: GnnT + global positional encoding |
| `RMS`                       | AP-RMS     | trajectory matching baseline                  |
| `Frechet`                   | AP-Fréchet | trajectory matching baseline                  |
| `DTW`                       | Soft-DTW   | trajectory matching baseline                  |

The image-based baselines of the paper (FIDeL, logpZO, lopO, AE-Recon) work on the raw videos and are not
part of this repository.

## Reproducing the paper results

Run the benchmark (5 methods x 6 tasks x 10 seeds) and aggregate the results:

```bash
uv run python scripts/run_benchmark.py                 # e.g. --methods TPC --seeds 1 2 3 for a subset
uv run python scripts/collect_results.py
```

`collect_results.py` prints the tables and saves them to `outputs/results/`:

- `threshold_independent.csv`: AUPR, maximum MCC and AUROC of the raw score (Fig. 3, Table 12);
- `end_to_end.csv`: MCC, TPR and TNR after conformal calibration, with the modulation s1 averaged over
  alpha in [0.80, 0.95] (Fig. 5, Table 13).

Values are the mean and standard deviation over seeds; dataset averages are computed per seed over the
tasks of the dataset, then averaged over seeds.

Runs are independent: `run_benchmark.py` skips the runs that are already done, so several instances can be
launched in parallel on disjoint subsets (`--seeds`, `--methods`, `--tasks`). The non-parametric methods are
deterministic given the seed. The training of GnnT is not bit-exact on GPU, so its results vary slightly
from one machine to another, within the seed-to-seed standard deviations reported in the paper.

## Repository structure

```
configs/
├── config.yaml                        # task, seed, paths
├── method/                            # one file per failure detector
└── threshold/                         # conformal calibration (alpha grid, D_A / D_B split)
robot_watchdog/
├── run.py                             # entry point: one (method, task, seed)
├── train.py                           # training of the predictive models
├── evaluate.py                        # online evaluation, conformal calibration, outputs
├── conformal_prediction.py            # time-varying conformal band
├── metrics.py                         # step-level metrics
├── prediction_error.py                # position + edge prediction error (loss and score of GnnT)
├── anomaly_detection/                 # TPC, GnnT and variants, RMS, Frechet, DTW
└── dataset/                           # trajectory loading, prefix streaming, graph construction
scripts/
├── run_benchmark.py                   # all (method, task, seed) runs
└── collect_results.py                 # tables of the paper
```

## Citation

```bibtex
@inproceedings{rolland2026robotwatchdog,
  title     = {Robot-{WATCHDOG}: failure detection through object-centric graph representation},
  author    = {Rolland, Quentin and Mayran de Chamisso, Fabrice and Mouret, Jean-Baptiste},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```

## License

The code is released under the [MIT license](LICENSE). The data of the release (object trajectories and
failure labels) are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Acknowledgments

The soft-DTW implementation (`robot_watchdog/anomaly_detection/soft_dtw_cuda.py`) comes from
[pytorch-softdtw-cuda](https://github.com/Maghoumi/pytorch-softdtw-cuda) (MIT license). The BotFails tasks
come from the BotFails benchmark (Rolland et al., ICRA 2026 - https://ieeexplore.ieee.org/abstract/document/11696900).
