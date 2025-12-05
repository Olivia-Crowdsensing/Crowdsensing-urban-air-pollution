import numpy as np
from pykrige.ok import OrdinaryKriging

import pandas as pd

def china():
    name = "2021-12"
    data = pd.read_excel("../Diffusion_Kriging/data/China/" + name + "_geo.xlsx", header=0)
    gt_data = pd.read_excel("../Diffusion_Kriging/accuracy/" + name + ".gt.xlsx", header=0)

    unkown_index = set(gt_data["index"])  # the unknown_index 300
    known_index = set(data["tid"]) - unkown_index  # 1300

    # second, using the set to divide the data file into known and unknown
    unknown_df = data.T[unkown_index].T
    known_df = data.T[known_index].T

    for i in range(1, gt_data.columns.shape[0]):
        print(i)
        row_name = gt_data.columns[i]  # the date to be predicted
        x, y = known_df["lat"], known_df["lng"]
        v = known_df[row_name]  # to pickout data which need to do the fitting variogram

        # working with zero, replace it with nan
        x = x[v != 0]
        y = y[v != 0]
        v = v[v != 0]

        try:
            OK = OrdinaryKriging(x, y, v, variogram_model="spherical", verbose=False,
                                 enable_plotting=False, nlags=10, coordinates_type="geographic")
            ux, uy = unknown_df["lat"], unknown_df["lng"]
            z, sigmasq = OK.execute("points",
                                    xpoints=np.asarray(ux).astype(np.float64),
                                    ypoints=np.asarray(uy).astype(np.float64))
            if i == 1:
                est_v = pd.DataFrame(z)
            else:
                est_v = pd.concat([est_v, pd.DataFrame(z)], axis=1)
            print(est_v)
        except:
            z = pd.DataFrame(np.zeros(shape=len(ux)))
            est_v = pd.concat([est_v, z], axis=1)
            print(est_v)

        est_v.to_excel(name + ".OK.est.xlsx")


def synthetic():
    name = "highAuto_highHete"
    data = pd.read_excel("../Diffusion_Kriging/data/Synthetic/" + name + ".xlsx")
    gt_data = pd.read_excel("../Diffusion_Kriging/accuracy/synthetic/" + name + ".gt.xlsx")

    # First, convert the gt_data index (unknown) -> set
    unknown_index = set(gt_data["index"])
    known_index = set(data["sid"]) - unknown_index

    # Second, using the set to divide the data file (known and unkown)
    unknown_df = data.T[unknown_index].T
    known_df = data.T[known_index].T

    # Third, conduct the Kriging approach to do the interpolation
    # 3.1 To fit the variogram from known_df
    for i in range(0, 1):
        v_name = 10 + i
        x, y, v = known_df["x"], known_df["y"], known_df["t" + str(v_name)]
        OK = OrdinaryKriging(x, y, v, variogram_model="spherical", verbose=True,
                             enable_plotting=True, nlags=20, enable_statistics=True)

        ux, uy = unknown_df["x"], unknown_df["y"]
        z, sigmasq = OK.execute("points", xpoints=np.asarray(ux), ypoints=np.asarray(uy))
        if i == 0:
            est_v = pd.DataFrame(z)
        else:
            est_v = pd.concat([est_v, pd.DataFrame(z)], axis=1)

    # est_v.to_excel(name + ".OK.est.xlsx")


if __name__ == '__main__':
    import skgstat as skg
    import matplotlib.pyplot as plt

    plt.style.use('ggplot')

    name = "highAuto_highHete"
    data = pd.read_excel("../Diffusion_Kriging/data/Synthetic/" + name + ".xlsx")
    gt_data = pd.read_excel("../Diffusion_Kriging/accuracy/synthetic/" + name + ".gt.xlsx")

    '''
    # First, convert the gt_data index (unknown) -> set
    unknown_index = set(gt_data["index"])
    known_index = set(data["sid"]) - unknown_index

    # Second, using the set to divide the data file (known and unkown)
    unknown_df = data.T[unknown_index].T
    known_df = data.T[known_index].T
    '''

    # Third, conduct the Kriging approach to do the interpolation
    # 3.1 To fit the variogram from known_df

    i = 1
    v_name = 10 + i
    xy, v = data[["x", "y"]].values, data["t0"]
    V = skg.Variogram(coordinates=xy, values=v, use_nugget=True)

    # 3.2 change parameters
    V.n_lags = 15

    V.model = "spherical"
    V.plot()

    print("output: ", V.mse, V.rmse)
    print("para: ", V.parameters)


