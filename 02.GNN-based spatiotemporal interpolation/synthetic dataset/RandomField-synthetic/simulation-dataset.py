import numpy as np
import matplotlib.pyplot as plt
import gstools as gs

# The first step is to generate the lattic (define the location) `meshgrid`
x = y = np.arange(-12, 13, 1)
xx, yy = np.meshgrid(x, y)

# The mean process
yi = np.random.uniform(low=0, high=1, size=len(x) * len(y)).reshape(len(x), len(y))

# Case1: zero heterogeneity
zeroHete = 1 + 6.5 * yi

# Case2: low heterogeneity
lowHete_beta0 = 1 + (1 / 6) * (np.abs(xx) + np.abs(yy))
lowHete_beta1 = np.abs(xx) / 3
lowHete = lowHete_beta0 + lowHete_beta1 * yi

# Case3: high heterogeneity
highHete_beta0 = 1 + 4 * np.sin((1 / 12) * np.pi * xx)
highHete_beta1 = 1 + (1 / 5184) * (144 - (0 - xx) ** 2) * (144 - (0 - yy) ** 2)
highHete = highHete_beta0 + highHete_beta1 * yi


def zeroCorre(meanProcess, x, y):
    return np.random.normal(0, scale=np.sqrt(np.var(meanProcess) / 2),
                            size=len(x) * len(y)).reshape(len(x), len(y))


def lowCorre(meanProcess, x, y):
    # First, generate the semivariance
    model = gs.Exponential(dim=2, var=(1 / 2) * np.var(meanProcess) / 2,
                           len_scale=1.5, nugget=0.5)  # cov
    srf = gs.SRF(model, seed=0)  # the spatial random field
    srf_surf = srf((x, y), mesh_type="structured")  # type of `ndarray`
    return srf, srf_surf


def highCorre(meanProcess, x, y):
    # First, generate the semivariance surface
    model = gs.Exponential(dim=2, var=(3 / 4) * np.var(meanProcess) / 2,
                           len_scale=3, nugget=0.25)
    srf = gs.SRF(model, seed=1)
    srf_surf = srf((x, y), mesh_type="structured")
    return srf, srf_surf


# To generate order 1 autoregressive
from statsmodels.tsa.arima_model import AutoReg

t = [x for x in range(0, 24)]  # t = [x + random() for x in range(0, 10)]

# to generate model
order1_t = AutoReg(t, lags=11)  # order 1 autoregressive
order1_t_fit = order1_t.fit()
# print(order1_t_fit.params)


if __name__ == '__main__':
    zeroAuto_highHete = zeroCorre(highHete, x, y)[1] + highHete
    # To draw the plot
    plt.pcolor(xx, yy, zeroAuto_highHete)  # This is the generation over meshgrid.
    plt.colorbar()
    # plt.show()
    plt.savefig("zeroAuto_highHete.png", dpi=300)

    '''
    import pandas as pd
    zeroAuto_zeroHete_flat = pd.concat([pd.DataFrame(xx.reshape(25*25, -1)),
                                        pd.DataFrame(yy.reshape(25*25, -1)),
                                        pd.DataFrame(highAuto_lowHete.reshape(25 * 25, -1))], axis=1)

    zeroAuto_zeroHete_flat.to_excel("highAuto_lowHete.xlsx")
    # lowAuto_lowHete = lowCorre(lowHete, x, y)[1] + lowHete
    # highAuto_highHete = highCorre(highHete, x, y)[1] + highHete

    '''
