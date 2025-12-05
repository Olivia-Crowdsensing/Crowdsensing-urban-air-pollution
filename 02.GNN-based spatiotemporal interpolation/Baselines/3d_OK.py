import numpy as np
from pykrige.ok3d import OrdinaryKriging3D
from tqdm import tqdm
import pandas as pd

name = "2021-11"
data = pd.read_excel("../Diffusion_Kriging/data/China/" + name + "_geo.xlsx")
gt_data = pd.read_excel("../Diffusion_Kriging/accuracy/" + name + ".gt.xlsx")

unkown_index = set(gt_data["index"])  # the unknown_index 300
known_index = set(data["tid"]) - unkown_index  # 1300

# second, using the set to divide the data file into known and unknown
unknown_df = data.T[unkown_index].T
known_df = data.T[known_index].T

# Below is to choose data and to fit the variogram
from time import time
starttime = time()
for t in tqdm(range(1, len(gt_data.columns) - 1)):
    x, y = known_df["lat"], known_df["lng"]
    row_name = gt_data.columns[t]  # the date to be predicted
    time = np.ones(x.shape[0])

    # coordinates
    x_f = x.append(x)
    y_f = y.append(y)
    time_f = np.append(time, time + t - 1)

    # about the values
    tmp_val = known_df[row_name]
    tmp_val_f = tmp_val.append(tmp_val)

    ok3d = OrdinaryKriging3D(x_f, y_f, time_f, tmp_val_f,
                             variogram_model="spherical",
                             verbose=True)
endtime = time()
print("####", endtime - starttime)

for i in range(1, len(gt_data.columns) - 1):
    ux, uy = unknown_df["lat"], unknown_df["lng"]  # The unknown dataframe ->
    ut = np.ones(ux.shape[0]) + i  # iteration for different t
    zz, sigmasq = ok3d.execute("points", ux, uy, ut)

    if i == 1:
        est_v = pd.DataFrame(zz)
    else:
        est_v = pd.concat([est_v, pd.DataFrame(zz)], axis=1)

est_v.to_excel(name + ".OK3d.est.xlsx")



def synthetic():
    name = "zeroAuto_zeroHete"
    data = pd.read_excel("../Diffusion_Kriging/data/Synthetic/" + name + ".xlsx")
    gt_data = pd.read_excel("../Diffusion_Kriging/accuracy/" + name + ".gt.xlsx")

    # First, convert the gt_data index (unknown) -> set
    unknown_index = set(gt_data["index"])
    known_index = set(data["sid"]) - unknown_index

    # Second, using the set to divide the data file (known and unkown)
    unknown_df = data.T[unknown_index].T
    known_df = data.T[known_index].T

    # Third, to fit the spatiotemporal variogram
    x, y = known_df["x"], known_df["y"]
    t = np.ones(x.shape[0])

    x4 = x.append(x.append(x.append(x)))
    y4 = y.append(y.append(y.append(y)))
    t2 = np.append(t, t + 1)
    t2_b = t2 + 2
    t4 = np.append(t2, t2_b)

    val2_b = np.append(np.asarray(known_df.loc[:, ["t10"]]),
                       np.asarray(known_df.loc[:, ["t11"]]))
    val2_a = np.append(np.asarray(known_df.loc[:, ["t12"]]),
                       np.asarray(known_df.loc[:, ["t13"]]))
    val4 = np.append(val2_b, val2_a)

    import time
    starttime = time.time()
    ok3d = OrdinaryKriging3D(x4, y4, t4, val4,
                             variogram_model="spherical",
                             verbose=True)
    endtime = time.time()
    print("# Processing time: ", endtime - starttime)

    for i in range(0, 4):
        ux, uy = unknown_df["x"], unknown_df["y"]  # The unknown dataframe ->
        ut = np.ones(ux.shape[0]) + i  # iteration for different t
        zz, sigmasq = ok3d.execute("points", ux, uy, ut)

        if i == 0:
            est_v = pd.DataFrame(zz)
        else:
            est_v = pd.concat([est_v, pd.DataFrame(zz)], axis=1)

    est_v.to_excel(name + ".OK3d.est.xlsx")