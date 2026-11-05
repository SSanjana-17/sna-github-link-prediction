# Network Dataset Package — GitHub Developer Link Prediction

Source: MUSAE GitHub Social Network (Rozemberczki, Allen & Sarkar, 2019),
originally collected from the public GitHub API, June 2019.

## Network model
| Element | Meaning |
|---|---|
| Node | A GitHub developer (with at least 10 starred repositories) |
| Node attribute `community` | `web` (web developer) or `ml` (machine-learning developer) |
| Edge | Mutual follower relationship between two developers |
| Graph type | Undirected, unweighted, simple |

## Files
| File | Contents |
|---|---|
| `nodes.csv` | id, name, community, degree in full graph, degree in training graph |
| `edges_clean.csv` | All 289,003 cleaned edges (source < target) |
| `train_edges.csv` | 231,202 edges (80%) used to compute all link-prediction scores |
| `test_positive_edges.csv` | 57,801 hidden edges (20%), label = 1 |
| `test_negative_edges.csv` | 57,801 random non-edges, label = 0 |
| `test_hard_negative_edges.csv` | 57,801 hard negatives: unconnected pairs 2 hops apart in the training graph, label = 0 |
| `train_graph.graphml` | Training graph with node attributes (opens in Cytoscape / Gephi) |
| `dataset_stats.json` | Cleaning log and graph statistics |

## Preparation steps (`../1_Source_Code_Notebook/src/stage2_prepare.py`, seed = 42)
1. Removed null / non-integer ids, self-loops, duplicate and reversed edges, edges to unknown nodes, isolated nodes.
2. Built an undirected NetworkX graph: 37,700 nodes, 289,003 edges, density 0.00041, 1 connected component, average clustering 0.168.
3. Split edges 80/20. A random spanning tree (37,699 edges) is always kept in the training set, so the training graph stays connected and no developer loses all their links.
4. Negative sampling: 57,801 node pairs drawn uniformly at random that are not edges in the full graph (1:1 with test positives).
5. Hard negative sampling (stage 4): 57,801 pairs reached by random two-step walks in the training graph that are not edges in the full graph.

`data_raw/` holds the original MUSAE files (`git_edges.csv`, `git_target.csv`) exactly as downloaded.

Checks passed: no train/test overlap, no negative is a real edge, train ∪ test = full edge set, every node has degree ≥ 1 in training.
