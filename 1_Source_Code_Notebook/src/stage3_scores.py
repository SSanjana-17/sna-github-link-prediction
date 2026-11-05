"""Stage 3 - compute CN, Jaccard, Adamic-Adar, PA for every test pair
(positives + random negatives) on the TRAINING graph, and check the
from-scratch implementation against NetworkX's built-in functions."""
import json
import random
import time

import networkx as nx
import numpy as np
import pandas as pd

from link_prediction import METHODS, adjacency, score_pairs

PKG = "dataset_package"
random.seed(42)

train = pd.read_csv(f"{PKG}/train_edges.csv")
nodes = pd.read_csv(f"{PKG}/nodes.csv")
G = nx.Graph()
G.add_nodes_from(nodes.id)
G.add_edges_from(train.itertuples(index=False, name=None))
adj = adjacency(G)

pos = pd.read_csv(f"{PKG}/test_positive_edges.csv")
neg = pd.read_csv(f"{PKG}/test_negative_edges.csv")
test = pd.concat([pos, neg], ignore_index=True)

t = time.time()
scores = score_pairs(adj, test[["source", "target"]].itertuples(index=False, name=None))
scores["label"] = test["label"].values
elapsed = time.time() - t
scores.to_csv("results/test_scores.csv", index=False)

# --- Validate against NetworkX on a random sample ---------------------
sample = random.sample(list(zip(scores.source, scores.target)), 3000)
ref = {
    "CN": [len(list(nx.common_neighbors(G, u, v))) for u, v in sample],
    "Jaccard": [p for _, _, p in nx.jaccard_coefficient(G, sample)],
    "AdamicAdar": [p for _, _, p in nx.adamic_adar_index(G, sample)],
    "PA": [p for _, _, p in nx.preferential_attachment(G, sample)],
}
mine = score_pairs(adj, sample)
check = {m: bool(np.allclose(mine[m].values, ref[m])) for m in METHODS}

# --- Score summary by class -------------------------------------------
summary = scores.groupby("label")[METHODS].agg(["mean", "median"]).round(4)
zero_cn = scores.groupby("label")["CN"].apply(lambda s: round((s == 0).mean(), 4))

out = {
    "pairs_scored": len(scores),
    "seconds": round(elapsed, 1),
    "matches_networkx": check,
    "share_with_zero_common_neighbors": {"positive": float(zero_cn[1]), "negative": float(zero_cn[0])},
}
print(json.dumps(out, indent=2))
print(summary.T)
summary.T.to_csv("results/score_summary_by_class.csv")
json.dump(out, open("results/stage3_log.json", "w"), indent=2)
