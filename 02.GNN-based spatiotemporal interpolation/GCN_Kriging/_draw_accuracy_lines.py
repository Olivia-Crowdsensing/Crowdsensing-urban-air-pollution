import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def general():
    national_acc = pd.read_excel("./accuracy/china/Accuracy_matrix_combination_china.xlsx", sheet_name="plot-general")
    sns.set_theme(style="ticks")
    # sns.lineplot(data=national_acc, x=national_acc["Month"], y=national_acc["RMSE"],
    #              hue=national_acc["Method"], style=national_acc["Method"],
    #              markers=True,
    #              ci=None)
    # plt.show()

    g = sns.relplot(x=national_acc["Month"], y=national_acc["MAE"], kind="line",
                hue=national_acc["Method"], style=national_acc["Method"],
                markers=True, ci=None)

    g.set(xticks=np.arange(1, 13, 1))
    g.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

    # g.set(yticks=np.arange(10, 90, 10))
    # plt.show()

    # plt.savefig("MAE-national.png", dpi=300)



# 1. RMSE(east-north)
# generate multiple plots
national_acc = pd.read_excel("./accuracy/china/Accuracy_matrix_combination_china.xlsx",
                             sheet_name="plot-four zones")
sns.set_theme(style="ticks")
g = sns.relplot(x=national_acc["Month"], y=national_acc["RMSE"], kind="line",
                hue=national_acc["Method"], style=national_acc["Method"],
                markers=True, ci=None)
g.set(xticks=np.arange(1, 13, 1))
g.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.show()
# plt.savefig("RMSE_east-north.png", dpi=300)
