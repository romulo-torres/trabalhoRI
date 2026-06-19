from glob import glob
from scipy import stats
import matplotlib.pyplot as plt
import json

metricas = ["P@10", "R@10", "P@100", "R@100", "MRR", "MAP", "nDCG@10"]

with open("data/evaluation/results.json") as f:
    results = json.load(f)

qtype = {}
for query in glob("data/pooling/q_*.json"):
    with open(query) as arq:
        meujson = json.load(arq)
    qtype[meujson["query_id"]] = meujson["query_type"]

data = {"factoid": {m: [] for m in metricas}, "keyword": {m: [] for m in metricas}}
for qid, qd in results["per_query"].items():
    t = qtype.get(qid)
    if (t != None):
        for m in metricas:
            data[t][m].append(qd[m])

print(f"{'Métrica':8s}  {'U1':<8s}  {'U2':<8s}  p")
for m in metricas:
    u1, p = stats.mannwhitneyu(data["factoid"][m], data["keyword"][m], alternative="two-sided")
    u2 = len(data["factoid"][m]) * len(data["keyword"][m]) - u1
    print(f"{m:<8s}  {u1:<8.1f}  {u2:<8.1f}  {p:.4f}")

all_data = {m: [qd[m] for qd in results["per_query"].values()] for m in metricas}
fig2, axs2 = plt.subplots(2, 4, figsize=(16, 8))
axs2 = axs2.flatten()
for i, m in enumerate(metricas):
    ax = axs2[i]
    ax.boxplot(all_data[m])
    ax.set_title(m, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.set_xticks([])
axs2[-1].axis("off")
plt.tight_layout()
plt.savefig("boxplot_geral.png", dpi=150)

fig, axs = plt.subplots(2, 4, figsize=(16, 8))
axs = axs.flatten()
for i, m in enumerate(metricas):
    ax = axs[i]
    ax.boxplot([data["factoid"][m], data["keyword"][m]])
    ax.set_xticklabels(["factoid", "keyword"])
    ax.set_title(m, fontsize=10)
    ax.tick_params(labelsize=8)
axs[-1].axis("off")
plt.tight_layout()
plt.savefig("boxplot_factoid_vs_keyword.png", dpi=150)

print("\nBoxplots salvos em boxplot_geral.png e boxplot_factoid_vs_keyword.png")
