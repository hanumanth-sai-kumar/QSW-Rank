# Results

The tables of the paper, with the log and script behind each one. All values are means in %, except Table 1. The full output of every run is in [`logs/`](logs/).

Measures: NDCG@10 checks whether the riskiest hosts are at the top of the ranking; Spearman's r_s checks the order of the whole list of reachable hosts. The true risk of a host is its probability of compromise in 20,000 simulated attacks. Attackers: stealthy (B = 10 attempts), moderate (B = 30), unlimited (tries every enabled step).

Significance: paired Wilcoxon signed-rank test over the networks, Holm correction per attacker × metric, p < 0.05. † = QSW-Rank significantly better, ‡ = significantly worse. The Holm families also contain comparators the paper does not report (see `experiments/02_main_results.py`), so the tests are conservative.

## Table 1: test networks (means of ten networks per size)

Source: `logs/01_mulval_benchmark.log` ("Attack graphs generated", "Ground-truth compromise probabilities"); walk nodes: `logs/02_main_results.log` ("QSW cost").

| Hosts | Graph nodes | Walk nodes | Reachable hosts | Mean risk B=10 | B=30 | Unlimited |
|---|---|---|---|---|---|---|
| 25 | 399 | 276 | 9.5 | 0.108 | 0.210 | 0.389 |
| 50 | 853 | 587 | 22.8 | 0.019 | 0.102 | 0.393 |
| 75 | 1,463 | 1,003 | 32.7 | 0.011 | 0.066 | 0.434 |
| 100 | 2,251 | 1,533 | 48.8 | 0.003 | 0.029 | 0.512 |

## Table 2: limited attackers, 40 networks

Source: `logs/02_main_results.log` ("MEAN (%)" and "QSW omega-average minus each method"). Script: `experiments/02_main_results.py`.

| Method | B=10 NDCG@10 | B=10 r_s | B=30 NDCG@10 | B=30 r_s |
|---|---|---|---|---|
| **QSW-Rank** | **98.1** | **80.0** | **97.6** | **82.9** |
| Pure CTQW (ω = 0) | 97.0 | 70.9† | 95.5† | 70.1† |
| Pure DTQW (best steps) | 92.5† | 63.7† | 91.7† | 59.6† |
| Markov walk (ω = 1) | 92.7† | 79.0 | 92.6† | 81.7† |
| PageRank | 88.1† | 77.3† | 88.3† | 80.1† |
| A* | 90.8† | 68.2 | 91.8† | 72.4 |
| AO* | 91.0† | 71.5 | 92.1† | 76.5 |
| BFS | 87.5† | 74.8 | 86.9† | 73.1† |
| DFS | 50.1† | 2.1† | 55.1† | 5.7† |

## Table 3: NDCG@10 by network size, stealthy attacker

Source: `logs/02_main_results.log` ("B10 NDCG@10 (%) by hosts"). Ten networks per size; not tested for significance.

| Method | 25 | 50 | 75 | 100 |
|---|---|---|---|---|
| QSW-Rank | 99.0 | 99.1 | 98.1 | 96.4 |
| Pure CTQW | 99.0 | 98.3 | 98.8 | 91.6 |
| Pure DTQW | 99.6 | 94.7 | 91.2 | 84.6 |
| Markov walk | 98.2 | 93.3 | 91.3 | 87.9 |
| A* | 99.6 | 89.3 | 93.0 | 81.2 |
| AO* | 99.7 | 89.5 | 93.5 | 81.2 |

## Table 4: ablation, stealthy attacker, 40 networks

Sources: rows 1-3 and 5, `logs/02_main_results.log`; classical hopping, `logs/03_classical_hopping.log` ("Classical mixture (sq)", B10). Fig. 3 plots the full ω curves: the QSW curve is in `logs/01_mulval_benchmark.log` ("Decoherence curve"), the classical one in `logs/03_classical_hopping.log`.

| Variant | NDCG@10 | r_s |
|---|---|---|
| Pure CTQW (ω = 0) | 97.0 | 70.9 |
| Markov walk (ω = 1) | 92.7 | 79.0 |
| QSW at a fixed rate (ω = 0.5) | 95.8 | 80.4 |
| Classical hopping (ω = 0.5) | 95.4 | 80.0 |
| QSW-Rank (average over ω) | 98.1 | 80.0 |

## Table 5: held-out networks and the unlimited attacker

Sources: 20 held-out networks (seeds 300-304 per size), `logs/04_heldout_networks.log` ("MEAN OVER FRESH GRAPHS" and "E2-E5 [B10]"); unlimited attacker, `logs/02_main_results.log`. On the held-out networks, QSW-Rank was not tested against the Markov walk and PageRank.

| Method | Held-out B=10 NDCG@10 | r_s | Unlimited NDCG@10 | r_s |
|---|---|---|---|---|
| QSW-Rank | 95.6 | 81.4 | 76.1 | 31.6 |
| QSW (ω = 0.5) | 91.6† | 82.3 | 75.1 | 32.5 |
| Markov walk | 87.2 | 80.2 | 74.7 | 30.8 |
| PageRank | 82.0 | 78.6 | 74.4 | 30.7 |
| A* | 84.6 | 60.4 | 84.0‡ | 57.3‡ |
| AO* | 84.6 | 63.9 | 84.3‡ | 60.5‡ |

## Cost

Source: `logs/01_mulval_benchmark.log` ("Fair CPU timing, peak memory"), first network of each size, one QSW rate in complex128 on a 4-core CPU.

| Hosts | 25 | 50 | 75 | 100 |
|---|---|---|---|---|
| Walk nodes | 156 | 542 | 659 | 1,406 |
| QSW, one rate (s) | 0.61 | 8.08 | 13.17 | 92.5 |
| QSW peak memory (MB) | 4.8 | 58 | 86 | 392 |
| A* (s) | 0.007 | 0.030 | 0.036 | 0.090 |
| AO* (s) | 0.012 | 0.062 | 0.075 | 0.186 |

MulVAL took 7.2 s, 17.8 s, 188.8 s and 552.3 s per network on average to build the attack graphs.
