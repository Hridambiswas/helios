"""
Agent pipeline latency breakdown for the Helios 5-agent LangGraph system.
Shows simulated p50/p95/p99 latencies per agent stage.
Run: python graphs/agent_latency.py
"""
import matplotlib.pyplot as plt
import numpy as np

agents = ["Planner\n(GPT-4o)", "Retriever\n(CLIP+BM25)", "Executor\n(Sandbox)", "Synthesizer\n(GPT-4o)", "Critic\n(GPT-4o)"]
p50 = [310, 85, 45, 420, 290]
p95 = [580, 145, 210, 760, 510]
p99 = [890, 220, 850, 1100, 740]

x = np.arange(len(agents))
width = 0.26

fig, ax = plt.subplots(figsize=(11, 6))
b1 = ax.bar(x - width, p50, width, label="p50", color="#4CAF50", alpha=0.88, zorder=3)
b2 = ax.bar(x,         p95, width, label="p95", color="#2196F3", alpha=0.88, zorder=3)
b3 = ax.bar(x + width, p99, width, label="p99", color="#F44336", alpha=0.88, zorder=3)

ax.set_xlabel("Agent", fontsize=12)
ax.set_ylabel("Latency (ms)", fontsize=12)
ax.set_title("Helios — Per-Agent Latency (p50 / p95 / p99)", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(agents, fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=10)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 10, f"{h}", ha="center", va="bottom", fontsize=7.5)

plt.tight_layout()
plt.savefig("graphs/agent_latency.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: graphs/agent_latency.png")
