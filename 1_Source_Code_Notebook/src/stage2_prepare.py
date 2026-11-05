"""Stage 2 - Data Preparation & Network Construction
GitHub developer follower network (MUSAE GitHub dataset).

Steps
1. Load raw edge list + node labels
2. Clean: drop nulls/non-integer ids, self-loops, duplicate & reversed edges,
   edges pointing to unknown nodes
3. Build an undirected NetworkX graph
4. 80/20 train-test edge split that keeps the training graph connected
   (a random spanning tree is always kept in train)
5. Negative sampling: as many random non-edges as test edges
6. Save the Network Dataset Package
"""
import json
import os
import random

import networkx as nx
import numpy as np
import pandas as pd

SEED = 42
TEST_FRAC = 0.20
RAW = "data_raw"
OUT = "dataset_package"

random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUT, exist_ok=True)
log = {}

# 1. Load ---------------------------------------------------------------
edges_raw = pd.read_csv(f"{RAW}/git_edges.csv")
nodes_raw = pd.read_csv(f"{RAW}/git_target.csv")
log["raw_edges"] = len(edges_raw)
log["raw_nodes"] = len(nodes_raw)

# 2. Clean --------------------------------------------------------------
nodes = nodes_raw.dropna(subset=["id"]).copy()
nodes["id"] = nodes["id"].astype(int)
nodes = nodes.drop_duplicates("id")
nodes["name"] = nodes["name"].fillna("unknown").astype(str).str.strip()
nodes["community"] = nodes["ml_target"].map({0: "web", 1: "ml"})

e = edges_raw.dropna().copy()
e = e[pd.to_numeric(e["id_1"], errors="coerce").notna() &
      pd.to_numeric(e["id_2"], errors="coerce").notna()]
e = e.astype(int)
n_before = len(e)
self_loops = int((e.id_1 == e.id_2).sum())
e = e[e.id_1 != e.id_2]
# undirected: store each pair as (smaller, larger) and drop duplicates
e = pd.DataFrame({"u": e[["id_1", "id_2"]].min(axis=1),
                  "v": e[["id_1", "id_2"]].max(axis=1)})
dups = len(e) - len(e.drop_duplicates())
e = e.drop_duplicates()
known = set(nodes["id"])
unknown = int((~e.u.isin(known) | ~e.v.isin(known)).sum())
e = e[e.u.isin(known) & e.v.isin(known)].reset_index(drop=True)
log.update(self_loops_removed=self_loops, duplicate_edges_removed=dups,
           unknown_node_edges_removed=unknown, rows_dropped_invalid=log["raw_edges"] - n_before,
           clean_edges=len(e))

# 3. Build graph --------------------------------------------------------
G = nx.Graph()
for r in nodes.itertuples():
    G.add_node(r.id, name=r.name, community=r.community)
G.add_edges_from(e.itertuples(index=False, name=None))
isolates = list(nx.isolates(G))
G.remove_nodes_from(isolates)
log.update(isolated_nodes_removed=len(isolates), nodes=G.number_of_nodes(),
           edges=G.number_of_edges(), density=nx.density(G),
           connected_components=nx.number_connected_components(G),
           avg_degree=2 * G.number_of_edges() / G.number_of_nodes(),
           avg_clustering=nx.average_clustering(G))

# 4. Train/test split ---------------------------------------------------
# Random spanning tree (random weights + MST) is locked into train so no
# node becomes isolated / unreachable in the training graph.
rng = np.random.default_rng(SEED)
for u, v in G.edges():
    G[u][v]["w"] = rng.random()
tree = {tuple(sorted(x)) for x in nx.minimum_spanning_edges(G, weight="w", data=False)}
for u, v in G.edges():
    del G[u][v]["w"]

all_edges = [tuple(sorted(x)) for x in G.edges()]
n_test = int(round(TEST_FRAC * len(all_edges)))
removable = [x for x in all_edges if x not in tree]
rng.shuffle(removable)
test_pos = removable[:n_test]
test_set = set(test_pos)
train_edges = [x for x in all_edges if x not in test_set]

G_train = nx.Graph()
G_train.add_nodes_from(G.nodes(data=True))
G_train.add_edges_from(train_edges)
assert nx.is_connected(G_train)
log.update(train_edges=len(train_edges), test_pos_edges=len(test_pos),
           spanning_tree_edges=len(tree), train_connected=True)

# 5. Negative sampling --------------------------------------------------
# Uniformly random node pairs that are NOT edges in the full graph.
node_list = np.array(list(G.nodes()))
edge_set = set(all_edges)
neg = set()
while len(neg) < n_test:
    a, b = rng.choice(node_list, 2, replace=False)
    p = (int(min(a, b)), int(max(a, b)))
    if p not in edge_set:
        neg.add(p)
test_neg = sorted(neg)
log["test_neg_edges"] = len(test_neg)

# 6. Save package -------------------------------------------------------
nodes_out = pd.DataFrame(
    [(n, d["name"], d["community"], G.degree(n), G_train.degree(n))
     for n, d in G.nodes(data=True)],
    columns=["id", "name", "community", "degree_full", "degree_train"])
nodes_out.to_csv(f"{OUT}/nodes.csv", index=False)
pd.DataFrame(all_edges, columns=["source", "target"]).to_csv(f"{OUT}/edges_clean.csv", index=False)
pd.DataFrame(train_edges, columns=["source", "target"]).to_csv(f"{OUT}/train_edges.csv", index=False)
pd.DataFrame(test_pos, columns=["source", "target"]).assign(label=1).to_csv(f"{OUT}/test_positive_edges.csv", index=False)
pd.DataFrame(test_neg, columns=["source", "target"]).assign(label=0).to_csv(f"{OUT}/test_negative_edges.csv", index=False)
nx.write_graphml(G_train, f"{OUT}/train_graph.graphml")
with open(f"{OUT}/dataset_stats.json", "w") as f:
    json.dump(log, f, indent=2, default=float)

print(json.dumps(log, indent=2, default=float))
