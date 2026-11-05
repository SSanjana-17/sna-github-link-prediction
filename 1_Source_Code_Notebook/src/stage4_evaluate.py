"""Stage 4 - Algorithm comparison & validation.

Two test settings (positives are always the same 57,801 hidden edges):
  A. Random negatives  - uniformly random non-edges (easy, standard benchmark)
  B. Hard negatives    - non-edges exactly 2 hops apart in the training graph
                         (friends-of-friends who are NOT connected: the pairs a
                          real recommender actually has to choose between)

Metrics: ROC-AUC, Average Precision, and Precision / Recall / F1 @ K.
Ranking ties are broken randomly (seeded) so a method never gets credit
for the order in which pairs happen to be stored.
"""
import json

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve

from link_prediction import METHODS, adjacency, score_pairs

PKG = "dataset_package"
SEED = 42
rng = np.random.default_rng(SEED)

nodes = pd.read_csv(f"{PKG}/nodes.csv")
train = pd.read_csv(f"{PKG}/train_edges.csv")
full = pd.read_csv(f"{PKG}/edges_clean.csv")
G = nx.Graph()
G.add_nodes_from(nodes.id)
G.add_edges_from(train.itertuples(index=False, name=None))
adj = adjacency(G)
full_set = set(map(tuple, full[["source", "target"]].values))

pos = pd.read_csv(f"{PKG}/test_positive_edges.csv")
neg_random = pd.read_csv(f"{PKG}/test_negative_edges.csv")
n_pos = len(pos)

# --- Hard negative sampling: random 2-hop walks ---------------------------
node_arr = np.array(list(G.nodes()))
nbr = {n: np.fromiter(adj[n], dtype=np.int64) for n in adj}
hard = set()
while len(hard) < n_pos:
    u = int(rng.choice(node_arr))
    w = int(rng.choice(nbr[u]))
    x = int(rng.choice(nbr[w]))
    if x == u or x in adj[u]:
        continue
    p = (min(u, x), max(u, x))
    if p in full_set:            # a hidden test edge is not a negative
        continue
    hard.add(p)
neg_hard = pd.DataFrame(sorted(hard), columns=["source", "target"]).assign(label=0)
neg_hard.to_csv(f"{PKG}/test_hard_negative_edges.csv", index=False)

# --- Score -----------------------------------------------------------------
def scored(neg):
    t = pd.concat([pos, neg], ignore_index=True)
    s = score_pairs(adj, t[["source", "target"]].itertuples(index=False, name=None))
    s["label"] = t["label"].values
    s["tiebreak"] = rng.random(len(s))
    return s

KS = [100, 1000, 10000, n_pos]


def evaluate(s, setting):
    rows, curves = [], {}
    y = s["label"].values
    for m in METHODS:
        order = np.lexsort((s["tiebreak"].values, -s[m].values))  # score desc, random ties
        hits = np.cumsum(y[order])
        n_pos_here = int(y.sum())
        pr_, rc_, _ = precision_recall_curve(y, s[m])
        f1s = 2 * pr_ * rc_ / np.clip(pr_ + rc_, 1e-12, None)
        row = {"setting": setting, "method": m,
               "ROC_AUC": roc_auc_score(y, s[m]),
               "AvgPrecision": average_precision_score(y, s[m]),
               "BestF1": f1s.max()}
        for k in KS[:-1] + [n_pos_here]:
            p, r = hits[k - 1] / k, hits[k - 1] / y.sum()
            tag = "N" if k == n_pos_here else str(k)
            row[f"P@{tag}"] = p
            row[f"R@{tag}"] = r
            row[f"F1@{tag}"] = 2 * p * r / (p + r) if p + r else 0.0
        rows.append(row)
        fpr, tpr, _ = roc_curve(y, s[m])
        pr, rc, _ = precision_recall_curve(y, s[m])
        curves[m] = {"fpr": fpr, "tpr": tpr, "precision": pr, "recall": rc}
    return pd.DataFrame(rows), curves


s_rand, s_hard = scored(neg_random), scored(neg_hard)
s_hard.drop(columns="tiebreak").to_csv("results/test_scores_hard.csv", index=False)
res_rand, curves_rand = evaluate(s_rand, "Random negatives")
res_hard, curves_hard = evaluate(s_hard, "Hard (2-hop) negatives")
# Setting C: only friend-of-friend pairs on BOTH sides (positives with >=1
# common neighbour vs hard negatives) - the exact choice a recommender faces
s_c = s_hard[(s_hard.label == 0) | (s_hard.CN > 0)].copy()
res_c, curves_c = evaluate(s_c, "2-hop positives vs 2-hop negatives")
results = pd.concat([res_rand, res_hard, res_c], ignore_index=True)
results.to_csv("results/evaluation_metrics.csv", index=False)

np.save("results/curves.npy", {"random": curves_rand, "hard": curves_hard, "twohop": curves_c}, allow_pickle=True)
pd.set_option("display.width", 200)
print(results.round(4).to_string(index=False))
print("positives with CN=0:", round((s_rand[s_rand.label == 1].CN == 0).mean(), 4))
