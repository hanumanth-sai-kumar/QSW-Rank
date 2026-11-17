# QSW-Rank

Code and run logs for the paper **"QSW-Rank: Quantum Stochastic Walks for Ranking Host Risk Under Budget-Limited Attackers"** ([venue, year]).

QSW-Rank ranks the hosts of a MulVAL logical attack graph by how likely a cautious attacker is to compromise them. A quantum stochastic walk starts at the attacker. It jumps forward along attack steps at rates set by each step's success probability (rule probability × CVSS v3.1 exploitability), and it hops both ways along the same links. A host's score is the time the walker spends on it, averaged over the 11 mixing rates ω = 0, 0.1, …, 1. The method needs no labels and no per-network tuning. Because every jump operator joins only two nodes, the Lindblad equation is computed exactly with an N × N density matrix, which needs O(N²) memory instead of O(N⁴).

## Repository structure

```
QSW-Rank/
├── qswrank/                      # library
│   ├── config.py                 #   settings (CFG), output folders, report helpers
│   ├── engine.py                 #   quantum stochastic walk: operators, O(N^2) Lindblad evolution, self-tests
│   ├── mulval.py                 #   MulVAL build (Kaggle), network generator, MulVAL run, graph parser
│   ├── ground_truth.py           #   Monte-Carlo budget-limited attackers (AND/OR semantics)
│   ├── methods.py                #   walk graph, QSW-Rank scores, baselines, NDCG / Spearman
│   └── flows.py                  #   flow-dataset helpers (exploratory, not in the paper)
├── experiments/
│   ├── 01_mulval_benchmark.py    #   40 MulVAL networks, ground truth, all methods
│   ├── 02_main_results.py        #   Tables 2 and 3, unlimited attacker of Table 5, cost
│   ├── 03_classical_hopping.py   #   Table 4 and Fig. 3 (is the gain quantum?)
│   ├── 04_variants_heldout.py    #   held-out networks of Table 5, variants
│   └── exploratory/
│       └── flow_benchmark.py     #   CIC-IDS2017 / CSE-CIC-IDS2018 flows (not in the paper)
├── figures/                      # scripts for Figs. 1-4 of the paper (matplotlib)
├── notebooks/
│   └── qsw-rank_kaggle_run.ipynb # the Kaggle notebook of the main run, with its outputs
├── results/
│   ├── README.md                 # paper tables and where each number comes from
│   └── logs/                     # full output of every run behind the paper
├── tests/
│   ├── test_engine.py            # engine self-tests and the two-host example of Fig. 1
│   └── smoke_pipeline.py         # runs experiments 1-4 end to end on small synthetic graphs
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone [repository URL] && cd QSW-Rank
pip install -r requirements.txt          # numpy >= 2.0, scipy, pandas, matplotlib; torch optional
python tests/test_engine.py              # about 2 s on a CPU
```

The engine runs on a CPU with NumPy (complex128). If PyTorch finds a GPU, `qsw_batch` runs all mixing rates in one batch in complex64. The two give the same populations to about 1e-7 (checked by `engine_self_test`).

## Reproducing the paper

The runs behind the paper were made in Kaggle notebooks (Python 3, T4 GPU, internet on). `qswrank/mulval.py` builds XSB and MulVAL in `/kaggle/working`, so experiment 1 is meant for Kaggle. Experiments 2-4 only read its output and run anywhere.

| Step | Command | Needs | Paper | Log |
|---|---|---|---|---|
| 1 | `QSAW_RUN_FLOW=0 python experiments/01_mulval_benchmark.py` | Kaggle, XSB-5.1.0 source archive attached | Table 1, Fig. 3 (QSW curve) | `results/logs/01_mulval_benchmark.log` |
| 2 | `QSAW_PREV_ROOTS=<output of 1> QSAW_OUT=<new folder> python experiments/02_main_results.py` | output of 1 | Tables 2, 3; Table 5 (unlimited attacker); cost | `results/logs/02_main_results.log` |
| 3 | `QSAW_PREV_ROOTS=<output of 1> QSAW_OUT=<new folder> python experiments/03_classical_hopping.py` | output of 1 | Table 4, Fig. 3 (classical hopping) | `results/logs/03_classical_hopping.log` |
| 4a | `QSAW_SEED0=300 QSAW_N_GRAPHS=5 QSAW_RUN_FLOW=0 QSAW_OUT=<held-out folder> python experiments/01_mulval_benchmark.py` | Kaggle | 20 held-out networks | — |
| 4b | `QSAW_PREV_ROOTS=<held-out folder> QSAW_OUT=<new folder> python experiments/04_variants_heldout.py` | output of 4a | Table 5 (held-out networks) | `results/logs/04_heldout_networks.log` |
| Figs. | `python figures/fig_graph.py` (and `fig_arch.py`, `fig_omega.py`, `fig_pop.py`) | — | Figs. 1-4 | — |

On Kaggle, instead of `QSAW_PREV_ROOTS`, attach the output of the experiment-1 notebook as an input; the scripts search `/kaggle/input` for it. In the logged run of experiment 1, building the 40 attack graphs took about 40 minutes and the ground truth 15 minutes; experiments 2, 3 and 4b took 14, 25 and 10 minutes on the GPU.

To check that the whole pipeline runs without MulVAL, `python tests/smoke_pipeline.py` runs experiments 1-4 on small synthetic graphs in about 20 s. Its numbers mean nothing.

### Settings

All experiment settings are in `CFG` in `qswrank/config.py` and were fixed before the runs: 4 sizes (25, 50, 75, 100 hosts) × 10 networks from seed 200, 20,000 ground-truth runs per attacker (B = 10, 30, unlimited), ω ∈ {0, 0.1, …, 1}, RK4 step Δt = 0.05, horizon T = mean directed hop distance from the attacker. Environment variables only choose folders or shrink a run for testing:

| Variable | Meaning |
|---|---|
| `QSAW_OUT` | output folder (default `/kaggle/working/qsaw_final`) |
| `QSAW_PREV_ROOTS` | where experiments 2-4 look for the output of experiment 1 (default `/kaggle/input`) |
| `QSAW_RUN_MULVAL`, `QSAW_RUN_FLOW` | `0` skips the MulVAL part or the flow part of experiment 1 |
| `QSAW_SIZES`, `QSAW_N_GRAPHS`, `QSAW_SEED0`, `QSAW_GT_SAMPLES` | override the network sizes, networks per size, first seed, ground-truth runs |
| `QSAW_FULL_FAMILY` | experiment 2: `1` (default) keeps the extra comparators of the logged Holm families |
| `QSAW_FLOW_ROOT` | folder of the improved CIC-IDS2017 / CSE-CIC-IDS2018 data (exploratory script only) |

## Main result

Stealthy attacker (10 attempts), 40 MulVAL networks with 25-100 hosts, means in %:

| Method | NDCG@10 | Spearman r_s |
|---|---|---|
| **QSW-Rank** | **98.1** | **80.0** |
| Pure CTQW (ω = 0) | 97.0 | 70.9 |
| Markov walk (ω = 1) | 92.7 | 79.0 |
| AO* | 91.0 | 71.5 |
| A* | 90.8 | 68.2 |

All tables, with significance tests, are in [`results/README.md`](results/README.md). Against an attacker that tries every exploit, AO* ranks better (84.3 vs 76.1 NDCG@10).

## Notes

- **Provenance.** `qswrank/` and experiment 1 are the code of the Kaggle notebook in `notebooks/`, split into modules without changing the computations. Experiments 3 and 4 are the later notebook cells that produced logs 03 and 04. The cell behind log 02 was not kept; `02_main_results.py` was rebuilt from an earlier version of it and the log. The held-out log 04 came from a copy of `04_variants_heldout.py` whose messages begin with `[fresh]`.
- **QAGC port.** `methods.py` contains our own implementation of the walks described in the QAGC paper (Chittem et al., ICISSP 2026), written from the paper because no code was available to us. It was used in exploratory runs and appears in the logs; the QSW-Rank paper does not report it.
- **Exploratory code.** `qswrank/flows.py` and `experiments/exploratory/` are the flow-dataset experiments of the notebook. They are not part of the paper.
- **Known log label.** In `01_mulval_benchmark.log`, the headings "mean over all 12 graphs" are a typo in the notebook; the means are over all 40 graphs.

## Citation

```bibtex
@inproceedings{qswrank,
  title     = {QSW-Rank: Quantum Stochastic Walks for Ranking Host Risk Under Budget-Limited Attackers},
  author    = {[Author names]},
  booktitle = {[Venue]},
  year      = {[Year]}
}
```

## License

MIT, see [`LICENSE`](LICENSE).
