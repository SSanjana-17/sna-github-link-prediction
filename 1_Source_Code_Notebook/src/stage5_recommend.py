"""Stage 5 - Recommendation & interpretation.

1. Candidate generation (sparse matrix algebra, no 700M-pair brute force):
     CN  = A·A          AA = A·W·A  with W = diag(1/log deg)
   For every developer keep the 20 best unlinked candidates by CN and by AA,
   plus all unlinked pairs among the 300 highest-degree developers (PA's
   natural candidates). All four scores are then computed on that pool.
2. Raw Top-K per method.
3. Filtering:
     F1 not already connected (built in)
     F2 at least 2 common neighbours (an explainable reason: "you both follow X and Y")
     F3 drop "celebrity" accounts above the 99th degree percentile
        (everyone already sees them, recommending them adds nothing)
     F4 diversity - a developer may appear at most twice in a list
4. Back-test: run the same pipeline on the TRAINING graph and count how many
   of each Top-K list are hidden test edges (Hits@K).
5. Final Top-10 / Top-20 recommendations on the FULL graph + per-developer
   recommendations for two example developers.
"""
import json

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

from link_prediction import METHODS, adjacency, score_pairs

PKG = "dataset_package"
T_PER_NODE = 20
N_HUBS = 300
MAX_PER_DEV = 2
MIN_CN = 2

nodes = pd.read_csv(f"{PKG}/nodes.csv").set_index("id")


def build(edge_file):
    e = pd.read_csv(f"{PKG}/{edge_file}")
    G = nx.Graph()
    G.add_nodes_from(nodes.index)
    G.add_edges_from(e[["source", "target"]].itertuples(index=False, name=None))
    return G


def candidates(G):
    n = G.number_of_nodes()
    order = np.array(sorted(G.nodes()))
    assert (order == np.arange(n)).all()          # ids are 0..n-1
    A = nx.to_scipy_sparse_array(G, nodelist=order, format="csr", dtype=np.float32)
    deg = np.asarray(A.sum(axis=1)).ravel()
    w = np.where(deg >= 2, 1.0 / np.log(np.maximum(deg, 2)), 0.0).astype(np.float32)
    WA = (sp.diags(w) @ A).tocsr()
    pairs = set()
    for start in range(0, n, 2000):
        rows = np.arange(start, min(start + 2000, n))
        Ar = A[rows]
        for M in (Ar @ A, Ar @ WA):              # CN and AA for this block
            M = M - M.multiply(Ar)                 # remove existing links
            M = M.tocsr()
            for i, u in enumerate(rows):
                lo, hi = M.indptr[i], M.indptr[i + 1]
                cols, vals = M.indices[lo:hi], M.data[lo:hi]
                keep = (cols != u) & (vals > 0)
                cols, vals = cols[keep], vals[keep]
                if len(cols) == 0:
                    continue
                top = cols[np.argsort(-vals)[:T_PER_NODE]]
                pairs.update((min(u, v), max(u, v)) for v in top.tolist())
    hubs = order[np.argsort(-deg)[:N_HUBS]]
    for i, a in enumerate(hubs):
        for b in hubs[i + 1:]:
            if not G.has_edge(a, b):
                pairs.add((min(a, b), max(a, b)))
    adj = adjacency(G)
    pool = score_pairs(adj, sorted(pairs))
    return pool, deg


def annotate(df, deg):
    df = df.copy()
    df["source_name"] = nodes.loc[df.source, "name"].values
    df["target_name"] = nodes.loc[df.target, "name"].values
    cs, ct = nodes.loc[df.source, "community"].values, nodes.loc[df.target, "community"].values
    df["pair_type"] = np.where(cs == ct, np.char.add(cs.astype(str), "-" + ct.astype(str)), "cross (web-ml)")
    df["deg_source"], df["deg_target"] = deg[df.source], deg[df.target]
    return df


def top_k(pool, method, k, deg, filtered):
    ranked = pool.sort_values([method, "CN"], ascending=False)
    if not filtered:
        return ranked.head(k)
    cap = np.percentile(deg, 99)
    ranked = ranked[(ranked.CN >= MIN_CN) & (deg[ranked.source] <= cap) & (deg[ranked.target] <= cap)]
    out, count = [], {}
    for r in ranked.itertuples(index=False):
        if count.get(r.source, 0) >= MAX_PER_DEV or count.get(r.target, 0) >= MAX_PER_DEV:
            continue
        out.append(r)
        count[r.source] = count.get(r.source, 0) + 1
        count[r.target] = count.get(r.target, 0) + 1
        if len(out) == k:
            break
    return pd.DataFrame(out)


# ---------- 4. Back-test on the training graph ----------------------------
G_train = build("train_edges.csv")
pool_tr, deg_tr = candidates(G_train)
test_pos = set(map(tuple, pd.read_csv(f"{PKG}/test_positive_edges.csv")[["source", "target"]].values))
backtest = []
for m in METHODS:
    for filt in (False, True):
        for k in (10, 20, 100):
            t = top_k(pool_tr, m, k, deg_tr, filt)
            hits = sum((a, b) in test_pos for a, b in zip(t.source, t.target))
            backtest.append({"method": m, "list": "filtered" if filt else "raw", "K": k,
                             "hits": hits, "precision": hits / k})
backtest = pd.DataFrame(backtest)
backtest.to_csv("results/recommendation_backtest.csv", index=False)
print(backtest.pivot_table(index=["method", "list"], columns="K", values="hits"))
print("candidate pool on train graph:", len(pool_tr))

# ---------- 5. Final recommendations on the full graph --------------------
G_full = build("edges_clean.csv")
pool, deg = candidates(G_full)
print("candidate pool on full graph:", len(pool))
cols = ["rank", "source", "source_name", "target", "target_name", "pair_type",
        "deg_source", "deg_target"] + METHODS
final = {}
for m in METHODS:
    for filt in (False, True):
        t = annotate(top_k(pool, m, 20, deg, filt), deg)
        t.insert(0, "rank", range(1, len(t) + 1))
        tag = f"{m}_{'filtered' if filt else 'raw'}"
        t[cols].to_csv(f"results/top20_{tag}.csv", index=False)
        final[tag] = t[cols]

# Per-developer recommendations for two example developers (one ML, one web)
adj_full = adjacency(G_full)
examples = {}
for community in ("ml", "web"):
    cand = nodes[(nodes.community == community)]
    dev = int(cand[(cand.degree_full >= 25) & (cand.degree_full <= 40)].index[0])
    two_hop = {x for w in adj_full[dev] for x in adj_full[w]} - adj_full[dev] - {dev}
    s = score_pairs(adj_full, [(dev, x) for x in two_hop])
    s["candidate_name"] = nodes.loc[s.target, "name"].values
    s["candidate_community"] = nodes.loc[s.target, "community"].values
    s = s.sort_values(["AdamicAdar", "CN"], ascending=False).head(10)
    s.insert(0, "rank", range(1, 11))
    s.to_csv(f"results/top10_for_{nodes.loc[dev, 'name']}.csv", index=False)
    examples[community] = {"id": dev, "name": nodes.loc[dev, "name"], "degree": int(deg[dev]),
                           "two_hop_candidates": len(two_hop)}
    print(f"\nTop-10 for {nodes.loc[dev, 'name']} ({community}, degree {int(deg[dev])}):")
    print(s[["rank", "target", "candidate_name", "candidate_community", "CN", "AdamicAdar"]].to_string(index=False))

summary = {
    "pool_train": len(pool_tr), "pool_full": len(pool),
    "celebrity_degree_cap": float(np.percentile(deg, 99)),
    "examples": examples,
    "pair_type_mix": {k: v["pair_type"].value_counts().to_dict() for k, v in final.items()},
}
json.dump(summary, open("results/stage5_summary.json", "w"), indent=2, default=str)
for k in ("AdamicAdar_raw", "AdamicAdar_filtered", "PA_raw"):
    print(f"\n{k}\n", final[k].head(10).to_string(index=False))
