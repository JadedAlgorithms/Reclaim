"""
EDA — Exploratory Data Analysis
Generates charts from ml_dataset.csv and saves them to data/eda_*.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#111',
    'axes.facecolor':   '#1a1a1a',
    'axes.edgecolor':   '#333',
    'axes.labelcolor':  '#ccc',
    'xtick.color':      '#aaa',
    'ytick.color':      '#aaa',
    'text.color':       '#eee',
    'grid.color':       '#2a2a2a',
    'font.family':      'monospace',
})

ACCENT = ['#4ade80', '#4a9ade', '#f87171', '#facc15']

# ── load ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/ml_dataset.csv")

ACTION_MAP   = {0: "Retry now", 1: "Retry later", 2: "Switch method", 3: "Reminder"}
RISK_MAP     = {0: "Low", 1: "Medium", 2: "High"}
FAILURE_MAP  = {0: "Insufficient\nfunds", 1: "UPI\ntimeout"}

df["action_label"]  = df["action"].map(ACTION_MAP)
df["risk_label"]    = df["risk_tier"].map(RISK_MAP)
df["failure_label"] = df["failure_reason"].map(FAILURE_MAP)


def bar(ax, series, title, color="#4a9ade"):
    ax.bar(series.index, series.values * 100, color=color, width=0.5)
    for i, v in enumerate(series.values):
        ax.text(i, v * 100 + 0.5, f"{v*100:.1f}%", ha="center", fontsize=9, color="#eee")
    ax.set_title(title, pad=10, fontsize=11)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, min(series.max() * 130, 100))
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels(series.index, fontsize=9)


# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle("Reclaim — Exploratory Data Analysis", fontsize=15, y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

# 1. Success rate by action
ax1 = fig.add_subplot(gs[0, 0])
sr_action = df.groupby("action_label")["recovery_success"].mean().reindex(ACTION_MAP.values())
bar(ax1, sr_action, "Success rate by action", "#4ade80")

# 2. Success rate by risk tier
ax2 = fig.add_subplot(gs[0, 1])
sr_risk = df.groupby("risk_label")["recovery_success"].mean().reindex(RISK_MAP.values())
bar(ax2, sr_risk, "Success rate by risk tier", "#4a9ade")

# 3. Success rate by failure reason
ax3 = fig.add_subplot(gs[0, 2])
sr_fail = df.groupby("failure_label")["recovery_success"].mean().reindex(FAILURE_MAP.values())
bar(ax3, sr_fail, "Success rate by failure reason", "#facc15")

# 4. Success rate by attempt number
ax4 = fig.add_subplot(gs[1, 0])
sr_attempt = df.groupby("attempt_number")["recovery_success"].mean()
bar(ax4, sr_attempt, "Success rate by attempt #", "#f87171")

# 5. Amount distribution
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(df["amount"] / 1000, bins=20, color="#a78bfa", edgecolor="#111")
ax5.set_title("Transaction amount distribution", pad=10, fontsize=11)
ax5.set_xlabel("Amount (₹ thousands)")
ax5.set_ylabel("Count")
ax5.grid(axis="y", alpha=0.3)

# 6. Overall outcome split
ax6 = fig.add_subplot(gs[1, 2])
counts = df["recovery_success"].value_counts().reindex([0, 1])
ax6.pie(
    counts,
    labels=["Failure", "Success"],
    colors=["#f87171", "#4ade80"],
    autopct="%1.1f%%",
    startangle=90,
    textprops={"color": "#eee", "fontsize": 10},
    wedgeprops={"edgecolor": "#111", "linewidth": 2},
)
ax6.set_title("Overall outcome split", pad=10, fontsize=11)

plt.savefig("data/eda_overview.png", dpi=150, bbox_inches="tight")
print("Saved: data/eda_overview.png")
plt.show()

# ── key findings ──────────────────────────────────────────────────────────────
print("\n── Key Findings ──────────────────────────────────────────")
print(f"  Best action:    {sr_action.idxmax()}  ({sr_action.max()*100:.1f}% success)")
print(f"  Worst action:   {sr_action.idxmin()}  ({sr_action.min()*100:.1f}% success)")
print(f"  Low vs High risk: {sr_risk['Low']*100:.1f}% vs {sr_risk['High']*100:.1f}%")
print(f"  Attempt 1 vs 3: {df[df.attempt_number==1].recovery_success.mean()*100:.1f}% vs "
      f"{df[df.attempt_number==3].recovery_success.mean()*100:.1f}%")
print(f"  Dataset size:   {len(df)} rows")
print("─────────────────────────────────────────────────────────")
