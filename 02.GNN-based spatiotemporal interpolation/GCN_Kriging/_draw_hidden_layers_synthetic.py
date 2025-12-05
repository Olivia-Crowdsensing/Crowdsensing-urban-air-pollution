import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# step1: read excel the input (gt)\hidden layers\estimation layer

namelist = ["highAuto.highHete", "highAuto.lowHete", "highAuto.zeroHete",
            "lowAuto.highHete", "lowAuto.lowHete", "lowAuto.zeroHete",
            "zeroAuto.highHete", "zeroAuto.lowHete", "zeroAuto.zeroHete"]

for name in namelist:
    gt = pd.read_excel("./features/synthetic/" + name + ".gT.xlsx")
    hidden1, hidden2 = pd.read_excel("./features/synthetic/" + name + ".hidden1.xlsx"), pd.read_excel("./features/synthetic/" + name + ".hidden2.xlsx")
    est = pd.read_excel("./features/synthetic/"+ name +".est.xlsx")

    # step 2: prepare the dataframe
    x = y = np.arange(-12, 13, 1)
    xx, yy = np.meshgrid(x, y)

    # To plot the process
    (fig, subplots) = plt.subplots(4, 3, figsize=(28, 35))
    # 设置子图的间距
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    # 1. draw the input
    ax = subplots[0][0]
    try:
        ax.pcolor(xx, yy, np.asarray(gt["train1"]).reshape(25, 25), shading='auto')
    except:
        ax.pcolor(xx, yy, np.asarray(gt["gt1"]).reshape(25, 25), shading='auto')

    plt.setp(ax.get_xticklabels(), fontsize=24)
    plt.setp(ax.get_yticklabels(), fontsize=24)

    # 2. draw the hidden layers - dims
    for i in range(0, 3):
        ax = subplots[1][i]
        ax.pcolor(xx, yy, np.asarray(hidden1["d" + str(i+1)]).reshape(25, 25), shading='auto')
        plt.setp(ax.get_xticklabels(), fontsize=24)
        plt.setp(ax.get_yticklabels(), fontsize=24)


    for i in range(0, 3):
        ax = subplots[2][i]
        ax.pcolor(xx, yy, np.asarray(hidden2["d" + str(i+1)]).reshape(25, 25), shading='auto')
        plt.setp(ax.get_xticklabels(), fontsize=24)
        plt.setp(ax.get_yticklabels(), fontsize=24)

    # 3. draw the last output layers
    ax = subplots[3][0]
    ax.pcolor(xx, yy, np.asarray(est["est1"]).reshape(25, 25), shading='auto')
    plt.setp(ax.get_xticklabels(), fontsize=24)
    plt.setp(ax.get_yticklabels(), fontsize=24)

    # fig.tight_layout()
    # plt.savefig(name + ".hidlay.png", dpi=300)
    plt.show()