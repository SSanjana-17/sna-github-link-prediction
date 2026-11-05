"""Deliverable 3 - static figures (PNG + SVG) and Cytoscape exports."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

PKG, RES, FIG, CY = "dataset_package", "results", "figures", "cytoscape"
os.makedirs(FIG, exist_ok=True)
os.makedirs(CY, exist_ok=True)

SURF, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#ecebe7", "#c9c8c2"
METHOD_COLOR = {"CN": "#2a78d6", "Jaccard": "#eb6834", "AdamicAdar": "#1baf7a", "PA": "#6250d6"}
METHOD_NAME = {"CN": "Common Neighbors", "Jaccard": "Jaccard", "AdamicAdar": "Adamic-Adar",
               "PA": "Preferential Attachment"}
COMM_COLOR = {"web": "#2a78d6", "ml": "#eb6834"}
PRED = "#e34948"
METHODS = list(METHOD_COLOR)

plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=.7)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(f"{FIG}/{name}.svg", bbox_inches="tight")
    plt.close(fig)


nodes = pd.read_csv(f"{PKG}/nodes.csv").set_index("id")
full = pd.read_csv(f"{PKG}/edges_clean.csv")
G = nx.from_pandas_edgelist(full, "source", "target")
nx.set_node_attributes(G, nodes["name"].to_dict(), "name")
nx.set_node_attributes(G, nodes["community"].to_dict(), "community")
deg = dict(G.degree())

# 1. Degree distribution ------------------------------------------------
d = np.array(list(deg.values()))
vals, counts = np.unique(d, return_counts=True)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(vals, counts / counts.sum(), s=10, color=METHOD_COLOR["CN"], alpha=.75, edgecolor="none")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Degree (number of mutual-follow links)"); ax.set_ylabel("Share of developers")
ax.set_title("Degree distribution is heavy-tailed (log-log)", loc="left")
ax.annotate(f"median = {int(np.median(d))}\nmean = {d.mean():.1f}\nmax = {d.max():,}",
            xy=(.97, .95), xycoords="axes fraction", ha="right", va="top", color=INK2)
style(ax, "both")
save(fig, "01_degree_distribution")

# 2. Network overview: 400 highest-degree developers -------------------
top = sorted(deg, key=deg.get, reverse=True)[:400]
H = G.subgraph(top).copy()
pos = nx.spring_layout(H, seed=7, k=.35, iterations=80)
fig, ax = plt.subplots(figsize=(10, 9))
nx.draw_networkx_edges(H, pos, ax=ax, edge_color="#b8b7b0", width=.15, alpha=.5)
sizes = np.array([deg[n] for n in H])
nx.draw_networkx_nodes(H, pos, ax=ax, node_size=8 + 260 * (sizes / sizes.max()) ** .6,
                       node_color=[COMM_COLOR[H.nodes[n]["community"]] for n in H],
                       edgecolors=SURF, linewidths=.6)
for i, n in enumerate(sorted(H, key=deg.get, reverse=True)[:5]):
    off = [(-90, 40), (40, 45), (60, -35), (-100, -40), (30, 70)][i]
    ax.annotate(f"{H.nodes[n]['name']} ({deg[n]:,})", pos[n], xytext=off, textcoords="offset points",
                fontsize=8.5, color=INK, bbox=dict(boxstyle="round,pad=.25", fc=SURF, ec=AXIS, lw=.6),
                arrowprops=dict(arrowstyle="-", color=INK2, lw=.7))
for c, lab in (("web", "Web developer"), ("ml", "ML developer")):
    ax.scatter([], [], color=COMM_COLOR[c], label=lab, s=40)
ax.legend(frameon=False, loc="lower left")
ax.set_title(f"Core of the GitHub network: 400 most-connected developers ({H.number_of_edges():,} links)", loc="left")
ax.axis("off")
save(fig, "02_network_core_overview")

# 3. ROC curves, 3 settings ---------------------------------------------
curves = np.load(f"{RES}/curves.npy", allow_pickle=True).item()
metrics = pd.read_csv(f"{RES}/evaluation_metrics.csv")
settings = [("random", "Random negatives"), ("hard", "Hard (2-hop) negatives"),
            ("twohop", "2-hop positives vs 2-hop negatives")]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
for ax, (key, title) in zip(axs, settings):
    for m in METHODS:
        c = curves[key][m]
        auc = metrics[(metrics.setting == title) & (metrics.method == m)].ROC_AUC.iloc[0]
        ax.plot(c["fpr"], c["tpr"], color=METHOD_COLOR[m], lw=2, label=f"{METHOD_NAME[m]}  (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=AXIS, lw=1)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel("False positive rate")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax, "both")
axs[0].set_ylabel("True positive rate")
fig.suptitle("ROC curves: PA leads in every setting; neighbour-based methods drop towards chance on hard negatives",
             x=.01, ha="left", fontsize=12, color=INK)
fig.tight_layout()
save(fig, "03_roc_curves")

# 4. Metric comparison (grouped bars) ----------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
for ax, metric, title in zip(axs, ["ROC_AUC", "AvgPrecision", "P@N"],
                             ["ROC-AUC", "Average precision", "Precision@N (= Recall@N = F1@N)"]):
    x = np.arange(len(settings))
    for i, m in enumerate(METHODS):
        v = [metrics[(metrics.setting == t) & (metrics.method == m)][metric].iloc[0] for _, t in settings]
        b = ax.bar(x + (i - 1.5) * .2, v, .18, color=METHOD_COLOR[m], label=METHOD_NAME[m])
        for xi, vi in zip(x + (i - 1.5) * .2, v):
            ax.text(xi, vi + .01, f"{vi:.2f}", ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(x, ["Random\nnegatives", "Hard\nnegatives", "2-hop vs\n2-hop"])
    ax.set_title(title, loc="left", fontsize=11)
    ax.axhline(.5, color=AXIS, ls="--", lw=1)
    ax.set_ylim(0, 1.05)
    style(ax)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=9, loc="upper center", ncol=4, bbox_to_anchor=(.5, 1.06))
fig.tight_layout()
save(fig, "04_metric_comparison")

# 5. Precision@K at small K --------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, (_, title) in zip(axs, settings):
    x = np.arange(3)
    for i, m in enumerate(METHODS):
        r = metrics[(metrics.setting == title) & (metrics.method == m)].iloc[0]
        ax.bar(x + (i - 1.5) * .2, [r["P@100"], r["P@1000"], r["P@10000"]], .18,
               color=METHOD_COLOR[m], label=METHOD_NAME[m])
    ax.set_xticks(x, ["K = 100", "K = 1,000", "K = 10,000"])
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_ylim(0, 1.05)
    style(ax)
axs[0].set_ylabel("Precision@K")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=9, loc="upper center", ncol=4, bbox_to_anchor=(.5, 1.07))
fig.tight_layout()
save(fig, "05_precision_at_k")

# 6. Back-test of Top-K lists ------------------------------------------
bt = pd.read_csv(f"{RES}/recommendation_backtest.csv")
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, lst in zip(axs, ["raw", "filtered"]):
    x = np.arange(3)
    for i, m in enumerate(METHODS):
        v = [bt[(bt.method == m) & (bt.list == lst) & (bt.K == k)].precision.iloc[0] for k in (10, 20, 100)]
        ax.bar(x + (i - 1.5) * .2, v, .18, color=METHOD_COLOR[m], label=METHOD_NAME[m])
        for xi, vi in zip(x + (i - 1.5) * .2, v):
            ax.text(xi, vi + .01, f"{vi:.0%}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, ["Top-10", "Top-20", "Top-100"])
    ax.set_title(f"{lst.capitalize()} lists: share that were real hidden links", loc="left", fontsize=11)
    ax.set_ylim(0, 1)
    style(ax)
axs[0].set_ylabel("Hit rate")
axs[0].legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
save(fig, "06_recommendation_backtest")


# 7. Recommendation network: filtered Adamic-Adar Top-20 ----------------
def rec_network(top_csv, max_cn_per_pair=6):
    t = pd.read_csv(top_csv)
    R = nx.Graph()
    for r in t.itertuples():
        for n in (r.source, r.target):
            R.add_node(n, name=nodes.loc[n, "name"], community=nodes.loc[n, "community"],
                       degree=deg[n], role="candidate")
    for r in t.itertuples():
        common = sorted(set(G[r.source]) & set(G[r.target]), key=deg.get)[:max_cn_per_pair]
        for w in common:
            if w not in R:
                R.add_node(w, name=nodes.loc[w, "name"], community=nodes.loc[w, "community"],
                           degree=deg[w], role="shared neighbour")
            R.add_edge(r.source, w, kind="existing")
            R.add_edge(r.target, w, kind="existing")
    for r in t.itertuples():
        R.add_edge(r.source, r.target, kind="predicted", rank=int(r.rank),
                   score=round(float(r.AdamicAdar), 3), cn=int(r.CN))
    return R, t


def bridge_layout(sub):
    """Candidates on a circle (recommended pairs side by side), shared neighbours
    stacked between the candidates they connect."""
    cand = [n for n in sub if sub.nodes[n]["role"] == "candidate"]
    P = nx.Graph([(u, v) for u, v, d in sub.edges(data=True) if d.get("kind") == "predicted"])
    order = list(nx.dfs_preorder_nodes(P, cand[0])) if cand[0] in P else cand
    order += [c for c in cand if c not in order]
    pos = {}
    if len(order) == 2:
        pos[order[0]], pos[order[1]] = np.array([-1.0, 0]), np.array([1.0, 0])
    else:
        for i, n in enumerate(order):
            a = 2 * np.pi * i / len(order) + np.pi / 2
            pos[n] = np.array([np.cos(a), np.sin(a)]) * 1.2
    groups = {}
    for w in sub:
        if w in pos:
            continue
        groups.setdefault(frozenset(x for x in sub[w] if x in cand), []).append(w)
    for key, ws in groups.items():
        pts = [pos[k] for k in key]
        centre = np.mean(pts, axis=0)
        if len(key) == 1:                           # private neighbour: push outward
            centre = centre * 1.45
            direction = np.array([-centre[1], centre[0]])
        elif len(key) == 2:                         # between the two candidates, off the dashed line
            seg = pts[1] - pts[0]
            direction = np.array([-seg[1], seg[0]])
            centre = centre * .45
        else:
            centre = centre * .3
            direction = np.array([1.0, 0.0])
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        k = len(ws)
        span = np.linspace(-1, 1, k) * min(.9, .17 * (k - 1)) if k > 1 else np.array([0.0])
        if len(key) == 2 and len(order) > 2:      # keep clear of the dashed line itself
            span = np.where(np.abs(span) < .12, np.sign(span + 1e-9) * .12, span)
        for w, o in zip(ws, span):
            pos[w] = centre + direction * o
    return pos


def tiled_layout(R, cols=4):
    comps = sorted(nx.connected_components(R), key=len, reverse=True)
    pos, x, y, rowh = {}, 0.0, 0.0, 0.0
    for i, c in enumerate(comps):
        p = bridge_layout(R.subgraph(c))
        arr = np.array(list(p.values()))
        arr = (arr - arr.min(0)) / max(np.ptp(arr, 0).max(), 1e-9)
        size = 1.0 + .35 * np.sqrt(len(c) / 10)
        for n, (px, py) in zip(p, arr):
            pos[n] = (x + px * size, y + py * size)
        x += size + .55
        rowh = max(rowh, size)
        if (i + 1) % cols == 0:
            x, y, rowh = 0.0, y - rowh - .55, 0.0
    return pos


R, t = rec_network(f"{RES}/top20_AdamicAdar_filtered.csv")
pos = tiled_layout(R)
fig, ax = plt.subplots(figsize=(14, 10))
ex = [(u, v) for u, v, d in R.edges(data=True) if d["kind"] == "existing"]
pr = [(u, v) for u, v, d in R.edges(data=True) if d["kind"] == "predicted"]
nx.draw_networkx_edges(R, pos, edgelist=ex, edge_color="#b8b7b0", width=.6, alpha=.7, ax=ax)
cand = [n for n in R if R.nodes[n]["role"] == "candidate"]
oth = [n for n in R if R.nodes[n]["role"] != "candidate"]
nx.draw_networkx_nodes(R, pos, nodelist=oth, node_size=28, node_color=[COMM_COLOR[R.nodes[n]["community"]] for n in oth],
                       alpha=.45, edgecolors=SURF, linewidths=.5, ax=ax)
nx.draw_networkx_nodes(R, pos, nodelist=cand, node_size=170, node_color=[COMM_COLOR[R.nodes[n]["community"]] for n in cand],
                       edgecolors=INK, linewidths=1, ax=ax)
pe = nx.draw_networkx_edges(R, pos, edgelist=pr, edge_color=PRED, width=2.4, style="--", ax=ax)
for txt in nx.draw_networkx_labels(R, pos, labels={n: R.nodes[n]["name"] for n in cand}, font_size=7.5,
                                   font_color=INK, verticalalignment="bottom", ax=ax).values():
    txt.set_bbox(dict(boxstyle="round,pad=.15", fc=SURF, ec="none", alpha=.8))
    txt.set_y(txt.get_position()[1] + .04)
for c, lab in (("web", "Web developer"), ("ml", "ML developer")):
    ax.scatter([], [], color=COMM_COLOR[c], s=60, label=lab)
ax.plot([], [], color=PRED, ls="--", lw=2, label="Recommended link (Top-20)")
ax.plot([], [], color="#b8b7b0", lw=1, label="Existing link to a shared neighbour")
ax.legend(frameon=False, loc="lower left", fontsize=9)
ax.set_title("Top-20 filtered Adamic-Adar recommendations with the shared neighbours that justify them", loc="left")
ax.axis("off")
save(fig, "07_top20_recommendation_network")


# 8. Ego view for each example developer ---------------------------------
summary = json.load(open(f"{RES}/stage5_summary.json"))
for comm, info in summary["examples"].items():
    dev, name = int(info["id"]), info["name"]
    rec = pd.read_csv(f"{RES}/top10_for_{name}.csv")
    E = nx.Graph()
    E.add_node(dev)
    for n in G[dev]:
        E.add_edge(dev, n, kind="existing")
    for r in rec.itertuples():
        E.add_edge(dev, r.target, kind="predicted")
        for w in set(G[dev]) & set(G[r.target]):
            E.add_edge(r.target, w, kind="support")
    recs_order = list(rec.target)
    nbr_order = sorted(G[dev], key=lambda n: -len(set(G[n]) & set(recs_order)))
    p = {dev: np.array([0.0, 0.0])}
    for i, n in enumerate(recs_order):
        a = 2 * np.pi * i / len(recs_order) + np.pi / 2
        p[n] = np.array([np.cos(a), np.sin(a)]) * 1.0
    for i, n in enumerate(nbr_order):
        a = 2 * np.pi * (i + .5) / len(nbr_order) + np.pi / 2
        p[n] = np.array([np.cos(a), np.sin(a)]) * 2.1
    fig, ax = plt.subplots(figsize=(10, 10))
    for kind, col, wd, st, al in (("support", "#d8d7d0", .5, "-", .8), ("existing", "#a9a8a1", .8, "-", .8),
                                  ("predicted", PRED, 2.2, "--", 1)):
        el = [(u, v) for u, v, d in E.edges(data=True) if d["kind"] == kind]
        nx.draw_networkx_edges(E, p, edgelist=el, edge_color=col, width=wd, style=st, alpha=al, ax=ax)
    recs = set(rec.target)
    nbrs = [n for n in E if n != dev and n not in recs]
    nx.draw_networkx_nodes(E, p, nodelist=nbrs, node_size=40, node_color=[COMM_COLOR[nodes.loc[n, "community"]] for n in nbrs],
                           alpha=.6, edgecolors=SURF, ax=ax)
    nx.draw_networkx_nodes(E, p, nodelist=list(recs), node_size=150, node_color=[COMM_COLOR[nodes.loc[n, "community"]] for n in recs],
                           edgecolors=PRED, linewidths=1.5, ax=ax)
    nx.draw_networkx_nodes(E, p, nodelist=[dev], node_size=420, node_color=COMM_COLOR[comm], edgecolors=INK, linewidths=2, ax=ax)
    for n in recs:
        v = p[n] / np.linalg.norm(p[n])
        ax.annotate(nodes.loc[n, "name"], p[n], xytext=v * 22, textcoords="offset points", ha="center", va="center",
                    fontsize=8.5, color=INK, bbox=dict(boxstyle="round,pad=.2", fc=SURF, ec="none", alpha=.9))
    ax.annotate(name, p[dev], xytext=(0, -24), textcoords="offset points", ha="center", fontsize=10,
                fontweight="bold", color=INK, bbox=dict(boxstyle="round,pad=.2", fc=SURF, ec="none", alpha=.9))
    ax.text(0, -2.55, f"Inner ring: 10 recommended developers  ·  Outer ring: the {len(nbr_order)} developers {name} already follows",
            ha="center", fontsize=9, color=INK2)
    ax.set_aspect("equal")
    ax.plot([], [], color=PRED, ls="--", lw=2, label="Top-10 recommendation")
    ax.plot([], [], color="#8e8d86", lw=1, label=f"Existing links of {name}")
    ax.plot([], [], color="#d8d7d0", lw=1, label="Candidate's link to a shared neighbour")
    for c, lab in (("web", "Web developer"), ("ml", "ML developer")):
        ax.scatter([], [], color=COMM_COLOR[c], s=50, label=lab)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5, bbox_to_anchor=(0, .98))
    ax.set_title(f"Who should {name} follow next? Top-10 by Adamic-Adar", loc="left")
    ax.axis("off")
    save(fig, f"08_ego_recommendations_{name}")

# 9. Cytoscape desktop exports -------------------------------------------
for n in R:
    R.nodes[n]["degree"] = int(R.nodes[n]["degree"])
nx.write_graphml(R, f"{CY}/top20_recommendation_network.graphml")
cyjs = nx.cytoscape_data(R)
for nd in cyjs["elements"]["nodes"]:
    nd["data"]["id"] = str(nd["data"]["id"])
for e in cyjs["elements"]["edges"]:
    e["data"]["source"], e["data"]["target"] = str(e["data"]["source"]), str(e["data"]["target"])
json.dump(cyjs, open(f"{CY}/top20_recommendation_network.cyjs", "w"), indent=1)
pd.DataFrame([{"source": u, "target": v, **d} for u, v, d in R.edges(data=True)]).to_csv(
    f"{CY}/top20_recommendation_edges.csv", index=False)
pd.DataFrame([{"id": n, **d} for n, d in R.nodes(data=True)]).to_csv(f"{CY}/top20_recommendation_nodes.csv", index=False)
print("figures:", sorted(os.listdir(FIG)))
print("recommendation network:", R.number_of_nodes(), "nodes", R.number_of_edges(), "edges")


# 0. Score distributions (stage 3) --------------------------------------
s = pd.read_csv(f"{RES}/test_scores.csv")
fig, axs = plt.subplots(2, 2, figsize=(11, 7))
for a, m in zip(axs.flat, METHODS):
    off = 1 if m in ("CN", "PA") else 1e-3
    x = np.log10(s[m] + off)
    bins = np.linspace(x.min(), x.max(), 40)
    for lab, col, txt in ((0, "#eb6834", "Random non-links (test −)"), (1, "#2a78d6", "Real hidden links (test +)")):
        a.hist(x[s.label == lab], bins=bins, density=True, alpha=.55, color=col, label=txt, edgecolor=SURF, linewidth=.6)
    a.set_title(METHOD_NAME[m], loc="left", fontsize=11)
    a.set_xlabel(f"log10({m} + {off:g})")
    style(a)
axs[0, 0].set_ylabel("Density"); axs[1, 0].set_ylabel("Density")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, 1.03))
fig.tight_layout()
save(fig, "00_score_distributions")


# 10. Interactive Cytoscape.js page ----------------------------------------
def cy_elements(Rg, pred_key):
    els = []
    for n, d in Rg.nodes(data=True):
        els.append({"data": {"id": str(n), "label": d["name"], "community": d["community"],
                             "degree": int(d["degree"]), "role": d["role"]}})
    for u, v, d in Rg.edges(data=True):
        e = {"id": f"{u}-{v}", "source": str(u), "target": str(v), "kind": d["kind"]}
        if d["kind"] == "predicted":
            e.update(rank=d["rank"], cn=d["cn"], score=d["score"])
        els.append({"data": e})
    return els


views = {}
for m in METHODS:
    for lst in ("filtered", "raw"):
        Rm, tm = rec_network(f"{RES}/top20_{m}_{lst}.csv", max_cn_per_pair=4)
        for u, v, d in Rm.edges(data=True):
            if d["kind"] == "predicted":
                row = tm[((tm.source == u) & (tm.target == v)) | ((tm.source == v) & (tm.target == u))].iloc[0]
                d["score"] = round(float(row[m]), 4)
        views[f"{m}|{lst}"] = cy_elements(Rm, m)
json.dump(views, open(f"{CY}/views.json", "w"))
