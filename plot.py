import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("nonsteady.csv", delimiter=",", skiprows=1)
data = data.transpose()
data = data[11:]

colors = ["g", "b", "r", "m", "c", "#ff7f00", "#a65628", "g", "b", "r", "m"]
markers = ["o", "s", "^", "v", "<", ">", "p", "D", "x", "1", "2"]
line_styles = ["dashed", "dashed", "dashed", "dashed", "dashed", "dashed", "dashed", "solid", "solid", "solid", "solid"]
# colors = ["g", "g", "g", "g", "b", "b", "b", "r", "r", "r", "r"]
# markers = ["o", "s", "^", "v", "s", "^", "v", "o", "s", "^", "v"]
# line_styles = ["solid", "solid", "solid", "solid", "dotted", "dotted", "dotted", "dashed", "dashed", "dashed", "dashed"]

plt.figure(figsize=(8, 4), dpi=100)
plt.ylim([0, 3000])
# plt.xscale("log")
plt.xlabel('step', fontsize=12)
plt.ylabel("regret", fontsize=12)
for l in range(len(data)):
    plt.plot(data[l], alpha=0.8, linewidth=1.2, label=" ", color=colors[l], linestyle=line_styles[l],
             marker=markers[l], markevery=4999)
plt.legend(loc="best", frameon=False)
plt.savefig("regret_nonsteady2")
plt.clf()