"""Stage 3 - SNA Link Prediction scores, implemented from scratch.

For a candidate pair (u, v) in the TRAINING graph, with N(x) = neighbours of x:

  Common Neighbors (CN)        = |N(u) ∩ N(v)|
  Jaccard Coefficient (JC)     = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
  Adamic-Adar (AA)             = Σ_{w ∈ N(u) ∩ N(v)} 1 / log(|N(w)|)
  Preferential Attachment (PA) = |N(u)| · |N(v)|

Intuition
  CN : friends of friends become friends (triadic closure)
  JC : CN normalised by how many neighbours the pair has in total
  AA : a shared neighbour counts more when it is rare (a small hub)
       than when it is a celebrity followed by everyone
  PA : popular developers keep attracting new links (rich get richer)
"""
import math

import pandas as pd

METHODS = ["CN", "Jaccard", "AdamicAdar", "PA"]


def adjacency(G):
    """Neighbour sets for O(1) membership and fast set intersection."""
    return {n: set(G.neighbors(n)) for n in G.nodes()}


def common_neighbors(adj, u, v):
    return len(adj[u] & adj[v])


def jaccard(adj, u, v):
    union = len(adj[u] | adj[v])
    return len(adj[u] & adj[v]) / union if union else 0.0


def adamic_adar(adj, u, v):
    # degree of a common neighbour is always >= 2, so log(deg) > 0
    return sum(1.0 / math.log(len(adj[w])) for w in adj[u] & adj[v])


def preferential_attachment(adj, u, v):
    return len(adj[u]) * len(adj[v])


def score_pairs(adj, pairs):
    """Return a DataFrame with all four scores for each (u, v) pair."""
    rows = []
    for u, v in pairs:
        common = adj[u] & adj[v]
        cn = len(common)
        union = len(adj[u] | adj[v])
        rows.append((
            u, v, cn,
            cn / union if union else 0.0,
            sum(1.0 / math.log(len(adj[w])) for w in common),
            len(adj[u]) * len(adj[v]),
        ))
    return pd.DataFrame(rows, columns=["source", "target"] + METHODS)
