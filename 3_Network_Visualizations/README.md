# Network Visualisation Artifacts

| Folder | Contents |
|---|---|
| `static_png/`, `static_svg/` | All figures used in the report and deck, in PNG (170 dpi) and vector SVG |
| `interactive_html/` | `09_interactive_recommendation_explorer.html` — Cytoscape.js explorer of the Top-20 recommendations. Switch method (CN, Jaccard, Adamic-Adar, PA) and raw / filtered lists, click a dashed link to see its scores, drag nodes. Works offline (library embedded). |
| `cytoscape_renders/` | PNG renders of the Cytoscape.js explorer for all 8 method × list combinations (`_crop` versions show only the graph) |
| `cytoscape_import_files/` | The filtered Adamic-Adar Top-20 network for Cytoscape Desktop: `.cyjs`, `.graphml`, plus node and edge tables |

## Figure index
| File | Shows |
|---|---|
| 00_score_distributions | Score distributions for real vs random pairs, all four methods |
| 01_degree_distribution | Heavy-tailed degree distribution (log-log) |
| 02_network_core_overview | 400 most-connected developers, coloured by community |
| 03_roc_curves | ROC curves in the three evaluation settings |
| 04_metric_comparison | ROC-AUC, average precision and P@N per method and setting |
| 05_precision_at_k | Precision@100 / 1,000 / 10,000 |
| 06_recommendation_backtest | Share of each Top-K list that were real hidden links |
| 07_top20_recommendation_network | Filtered Adamic-Adar Top-20 with supporting shared neighbours |
| 08_ego_recommendations_* | Personal Top-10 recommendations for two example developers |

## Opening the network in Cytoscape Desktop
1. File → Import → Network from File… → choose `cytoscape_import_files/top20_recommendation_network.cyjs` (or the `.graphml`).
2. Layout → Prefuse Force Directed Layout (or yFiles Organic).
3. Style panel:
   - Node Fill Color → Discrete Mapping on `community`: web = #2A78D6, ml = #EB6834.
   - Node Size → Discrete Mapping on `role`: candidate = 40, shared neighbour = 15.
   - Node Label → Passthrough Mapping on `name`.
   - Edge Line Type → Discrete Mapping on `kind`: predicted = Dash, existing = Solid; Edge Stroke Color: predicted = #E34948, existing = #B8B7B0.
4. File → Export → Network to Image… (PNG or SVG) to produce a desktop render.
