"""
Retrieval score fusion analysis — dense, CLIP, and BM25 contribution breakdown.
Shows how each retrieval signal contributes to final hybrid score across query types.
Run: python graphs/retrieval_score_fusion.py
"""
import matplotlib.pyplot as plt
import numpy as np

query_types = ["Text-only\nfactual", "Image +\ntext", "Code\ngeneration", "Multi-hop\nreasoning", "Domain-\nspecific"]
dense_contrib  = [0.55, 0.30, 0.50, 0.60, 0.58]
clip_contrib   = [0.10, 0.52, 0.08, 0.12, 0.07]
bm25_contrib   = [0.35, 0.18, 0.42, 0.28, 0.35]

x = np.arange(len(query_types))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

width = 0.55
p1 = ax1.bar(x, dense_contrib, width, label="Dense (Ada)", color="#3F51B5", zorder=3, alpha=0.88)
p2 = ax1.bar(x, clip_contrib,  width, bottom=dense_contrib, label="CLIP ViT-B/32", color="#E91E63", zorder=3, alpha=0.88)
p3 = ax1.bar(x, bm25_contrib,  width,
             bottom=[d + c for d, c in zip(dense_contrib, clip_contrib)],
             label="BM25Okapi", color="#FF9800", zorder=3, alpha=0.88)

ax1.set_xticks(x)
ax1.set_xticklabels(query_types, fontsize=9)
ax1.set_ylabel("Score Weight (normalised)", fontsize=11)
ax1.set_title("Retrieval Signal Contribution\nby Query Type", fontsize=12, fontweight="bold")
ax1.set_ylim(0, 1.15)
ax1.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax1.set_axisbelow(True)
ax1.legend(fontsize=9)

avg = [np.mean(dense_contrib), np.mean(clip_contrib), np.mean(bm25_contrib)]
labels = ["Dense\n(Ada)", "CLIP\nViT-B/32", "BM25\nOkapi"]
colors = ["#3F51B5", "#E91E63", "#FF9800"]
wedges, texts, autotexts = ax2.pie(avg, labels=labels, colors=colors, autopct="%1.0f%%",
                                    startangle=90, pctdistance=0.75,
                                    wedgeprops={"edgecolor": "white", "linewidth": 2})
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight("bold")
ax2.set_title("Average Score Weight\nacross All Query Types", fontsize=12, fontweight="bold")

plt.suptitle("Helios — Hybrid Retrieval Score Fusion", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("graphs/retrieval_score_fusion.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: graphs/retrieval_score_fusion.png")
