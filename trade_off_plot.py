import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 15), dpi=50)

#ファイルの読み込み。最後だけ
st = pd.read_csv('steady.csv')[99900:-1]
non_st = pd.read_csv('nonsteady.csv')[99900:-1]
non_st_const = pd.read_csv('nonsteady_constant.csv')[99900:-1]

#いらない列の削除
drop_column = ["UCB1T regret", "RS regret", "RS OPT regret", "RS gamma regret", "RS OPT gamma regret", "meta UCB1T regret", "meta RS regret", "meta RS OPT regret"]
st = st.drop(drop_column, axis=1)
non_st = non_st.drop(drop_column, axis=1)
non_st_const = non_st_const.drop(drop_column, axis=1)

#平均の計算
st_means = st.mean()
non_st_means = non_st.mean()
non_st_const_means = non_st_const.mean()

names = st.columns.values
print(names)

#グラフの描画

fig = plt.figure(figsize=(9,7))

ax = fig.add_subplot(1,1,1)

labels = ["UCB1T", "RS", "RS OPT", "RS gamma", "RS OPT gamma", "meta UCB1T", "meta RS", "meta RS OPT"]
markers = ["D", "o", "^", "v", "s", "p", "h", "*"]
colors = ["r", "g", "b", "c", "m", "y", "k", "#a65628"]
# 非同期の点をうつ
for i,(x,y) in enumerate(zip(st_means , non_st_means)):
    #ax.annotate(names[i],(x,y))
    ax.annotate("no sync",(x+0.001,y-0.04))
    plt.plot(x, y, marker=markers[i], ms=10, color=colors[i])


# 同期の点をうって線で繋ぐ
for i in range(len(names)):
    x = st_means[names[i]]
    y = non_st_const_means[names[i]]
    dx = 0
    dy = non_st_means[names[i]] - y
    ax.arrow(x, y, dx, dy, width=0.0001, color='black', head_length=0.0, head_width=0.0)
    # plt.xlim([0.4, 1.0])
    plt.plot(x, y, label=labels[i], marker=markers[i], ms=10, color=colors[i])
    plt.legend(loc="best")
    ax.annotate("sync", (x+0.001, y+0.015))
    # ax.annotate(labels[i], (x+0.001, y+0.015))

# for i,(x,y) in enumerate(zip(st_means , non_st_means)):
#     ax.annotate("no sync",(x,y-0.03))

plt.xlabel("steady")
plt.ylabel("non-steady")
plt.savefig("trade_off")
plt.show()