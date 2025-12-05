import pandas as pd
import numpy as np


def reshape_excel(df):
    sid, x, y = df["sid"], df["x"], df["y"]
    for i in range(0, 24):
        # 1. create the current slice
        slice_tmp = pd.concat([sid, x, y, pd.DataFrame(np.asarray(df["t" + str(i)])),
                               pd.DataFrame(np.full(len(x), i))], 1)
        # 2. append
        if i == 0:
            slice = slice_tmp
        else:
            slice = slice.append([slice_tmp])
    return slice


if __name__ == '__main__':
    import os

    folder = "./"
    for filename in os.listdir(folder):
        print(filename)
        if filename[-4:] == "xlsx":
            df = pd.read_excel(filename)
            output = reshape_excel(df)
            output.to_excel("reshape/" + filename[:-5] + "_reshape.xlsx")

