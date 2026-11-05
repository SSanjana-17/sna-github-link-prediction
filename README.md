# SNA AAT — Link Prediction in the GitHub Developer Network

Predicting which GitHub developers should follow each other, using Common Neighbors, Jaccard,
Adamic-Adar and Preferential Attachment on the MUSAE GitHub social network
(37,700 developers, 289,003 mutual-follow links).

## Folder map

| Folder | Deliverable | Open this first |
|---|---|---|
| `1_Source_Code_Notebook/` | 1. Source code & executable notebook | `GitHub_Link_Prediction.ipynb` |
| `2_Network_Dataset_Package/` | 2. Network dataset package | `README.md` |
| `3_Network_Visualizations/` | 3. Visualisation artifacts (.png / .svg / .html) | `interactive_html/09_interactive_recommendation_explorer.html` |
| `4_Report/` | 4. Recommendation & evaluation report (.pdf) | `SNA_AAT_Link_Prediction_Report.pdf` |
| `5_Presentation/` | 5. Presentation deck (.pptx / .pdf) | `SNA_AAT_Link_Prediction_Presentation.pptx` |

## Headline results

| Setting | Best method | ROC-AUC |
|---|---|---|
| Random negatives | Preferential Attachment | 0.941 |
| Hard (2-hop) negatives | Preferential Attachment | 0.809 |
| 2-hop positives vs 2-hop negatives | Preferential Attachment | 0.868 |

Recommendation back-test: filtered Adamic-Adar — 12 of its Top-20 and 54 of its Top-100 were real hidden links.

## Running the notebook

```bash
cd 1_Source_Code_Notebook
pip install -r requirements.txt
jupyter notebook GitHub_Link_Prediction.ipynb     # Run All, about 3 minutes
```

The notebook reads `../2_Network_Dataset_Package/data_raw/` and writes everything to `1_Source_Code_Notebook/outputs/`.
All random steps use seed 42, so every number in the report and deck is reproduced exactly.

Before submitting, fill in your name and USN on the report cover page and on the first slide of the deck.
