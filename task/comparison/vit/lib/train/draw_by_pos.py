import numpy as np
import os 
import csv
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib as mpl

## 说明 ##
# 1. 读取csv，并可视化不同label的score。

with open('ncc_results.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    labels = []
    scores_ok = []
    scores_ng = []
    positions = []
    for i in range(1, len(data)):
        line = data[i]
        scores_ok.append(float(line[2]))
        scores_ng.append(float(line[3]))
        labels.append(int(line[4]))
        positions.append(os.path.normpath(line[0]).split('\\')[2])

labels = np.array(labels) # 真实标签
scores_ok = np.array(scores_ok) # ncc得分
scores_ng = np.array(scores_ng) # ncc得分
unique_pos = set(positions) # 点位
positions = np.array(positions)

for p in unique_pos:
    print(p)
    mask = (positions == p) # 点位mask
    masked_scores_ok = scores_ok[mask]
    masked_scores_ng = scores_ng[mask]
    masked_labels = labels[mask]

    colors = ['green', 'red']

    plt.cla()
    for i in range(2):
        xs = masked_scores_ok[masked_labels == i]
        ys = masked_scores_ng[masked_labels == i]
        if len(xs) == 0: continue
        
        cs = [colors[i] for _ in range(len(ys))]
        plt.scatter(xs, ys, c=cs)

    plt.plot([0, 1], [0, 1], ls="--", c=".3")
    plt.grid()
    plt.title(p)
    plt.xlabel("OK scores")
    plt.ylabel("NG scores")

    x_major_locator = MultipleLocator(0.1)
    y_major_locator = MultipleLocator(0.1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)


    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)

    # plt.savefig(os.path.join('test_figures', p+'.png'))
    plt.show()


