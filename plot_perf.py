import matplotlib.pyplot as plt
import numpy as np

rouge_scores_fanpage = [
    15.59,
    15.88,
    15.76,
    17.52,
]

rouge_scores_ilpost = [
    15.53,
    19.44,
    17.41,
    21.41,
]

rouge_scores_wits = [
    24.32,
    28.83,
    26.9,
    26.18,
]

models = [
    "IT5-base",
    "BART-IT",
    "mT5-base",
    "mBART",
]

element_per_sec = [
    1/0.09,
    1/0.03,
    1/0.09,
    1/0.24,
]

# plot multi-bar chart - 

# x label for the datasets - Fanpage, ILPost, WITS, Element per second - colorblind friendly
# y label for the rouge scores and element per second - colorblind friendly
# different colors for the different models - colorblind friendly
# legend for the models - colorblind friendly

datasets = ["Fanpage", "IlPost", "WITS", "Summaries/sec"]
x = np.arange(len(datasets))

width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 5))
rects1 = ax.bar(x - width, [rouge_scores_fanpage[0], rouge_scores_ilpost[0], rouge_scores_wits[0], element_per_sec[0]], width, label=models[0], color="#0072B2", hatch="\\\\")
rects2 = ax.bar(x, [rouge_scores_fanpage[1], rouge_scores_ilpost[1], rouge_scores_wits[1], element_per_sec[1]], width, label=models[1], color="#009E73", hatch="--")
rects3 = ax.bar(x + width, [rouge_scores_fanpage[2], rouge_scores_ilpost[2], rouge_scores_wits[2], element_per_sec[2]], width, label=models[2], color="#D55E00", hatch="..")
rects4 = ax.bar(x + 2*width, [rouge_scores_fanpage[3], rouge_scores_ilpost[3], rouge_scores_wits[3], element_per_sec[3]], width, label=models[3], color="#CC79A7", hatch="xx")

ax.axvline(x=2.6, color="black", linestyle="--")

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend()

fig.tight_layout()

plt.savefig('plot.png')
plt.savefig('plot.svg')
