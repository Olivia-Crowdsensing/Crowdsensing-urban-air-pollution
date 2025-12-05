import numpy as np
from pykrige.ok import OrdinaryKriging
import skgstat as skg
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel("classcial_STK/STKriging-baseline/"
                     "pm25/2021-11_geo_v1.xlsx")
data_s = data[data["t_tmp"] == 1]
data_s_na_o = data_s.dropna(axis=0)

xy, v = data_s_na_o[["lng", "lat"]].values, data_s_na_o["value"]
V = skg.Variogram(coordinates=xy, values=v,
                  estimator="matheron",
                  dist_func="euclidean",
                  use_nugget=True,
                  model="spherical",
                  n_lags=20,
                  maxlag=20,
                  verbose=True)


