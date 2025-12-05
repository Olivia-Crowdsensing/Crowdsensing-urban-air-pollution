import pandas as pd
import numpy as np

origin = pd.read_excel("./data/China/2017-01_geo.xlsx", header=0)
unknown_df = pd.read_excel("./accuracy/china/2017-01.gt.xlsx", header=0)

# to match `tid` <- with `unknown_df`
i = 0
for i in range(0, 300):
    tmp_index = unknown_df["index"][i]
    row = origin.loc[tmp_index]  # the row
    row_useful = pd.DataFrame(row[0:8]).T

    if i == 0:
        df_f = row_useful
    else:
        df_f = pd.concat([df_f, row_useful])

df_f.to_excel("df_f.xlsx")