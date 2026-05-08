"""
Critic agent score distribution — groundedness, faithfulness, completeness.
Simulates score distributions across 500 sampled queries to show QA gate behaviour.
Run: python graphs/critic_score_distribution.py
"""
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)
n = 500

groundedness  = np.clip(rng.beta(8, 2, n), 0, 1)
faithfulness  = np.clip(rng.beta(9, 1.5, n), 0, 1)
completeness  = np.clip(rng.beta(6, 3, n), 0, 1)
overall       = (groundedness + faithfulness + completeness) / 3

CRITIC_MIN = 0.7
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
dims = [
    (axes[0, 0], groundedness, "Groundedness", "#3F51B5"),
    (axes[0, 1], faithfulness,  "Faithfulness",  "#4CAF50"),
    (axes[1, 0], completeness,  "Completeness",  "#FF9800"),
    (axes[1, 1], overall,       "Overall Score", "#9C27B0"),
]

for ax, scores, title, color in dims:
    ax.hist(scores, bins=30, color=color, alpha=0.80, edgecolor="white", zorder=3)
    ax.axvline(CRITIC_MIN, color="#F44336", linestyle="--", linewidth=1.8, label=f"min threshold ({CRITIC_MIN})")
    ax.axvline(scores.mean(), color="black", linestyle="-", linewidth=1.4, label=f"mean = {scores.mean():.3f}")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Score [0, 1]", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    blocked = (scores < CRITIC_MIN).sum()
    ax.text(0.04, 0.92, f"Blocked: {blocked}/{n} ({100*blocked/n:.1f}%)",
            transform=ax.transAxes, fontsize=9, color="#F44336",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8})
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8)

plt.suptitle("Helios — Critic Agent Score Distributions (n=500 queries)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("graphs/critic_score_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: graphs/critic_score_distribution.png")
