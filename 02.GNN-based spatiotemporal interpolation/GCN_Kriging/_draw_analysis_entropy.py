import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.convolution import convolve
from scipy.stats import normaltest, probplot, spearmanr
from scipy.stats.mstats import pearsonr


def hh():
    # draw the entropy
    # reshape the above df -> numpy array as the grid
    # Step 1: highAuto highHete
    hh = pd.read_excel("features/synthetic/entropy/highAuto.highHete.entropy.xlsx")
    kernel = np.ones((3, 3))
    for i in range(0, 4):
        str_ = "gt" + str(i+1)
        grid = np.asarray(hh[str_]).reshape(25, 25)
        grid = grid + abs(np.nanmin(grid))  # Because the filter will erase these negative
        grid2 = grid / 1  # np.nanmax(grid)  # to do the standardize to [0, 1]

        str_e = "error" + str(i+1)
        error = np.abs(np.asarray(hh[str_e])).reshape(25, 25)  # pay attention to the abs.
        error2 = error / 1  # np.nanmax(error)

        mw_grid = convolve(grid2, kernel).reshape(-1, 1)
        mw_error = convolve(error2, kernel).reshape(-1, 1)

        if i == 0:
            grid_f, error_f = mw_grid, mw_error
        else:
            grid_f = np.append(grid_f, mw_grid)
            error_f = np.append(error_f, mw_error)

    hh_entropy_df = pd.DataFrame(grid_f, columns=["entropy"])
    hh_error_df = pd.DataFrame(error_f, columns=["error"])
    hh_output = pd.concat([hh_entropy_df, hh_error_df], axis=1)

    # to plot
    sns.set_theme(style="white", color_codes=True)
    g = sns.JointGrid(data=hh_output, x=hh_output["entropy"], y=hh_output["error"])
    g = g.plot_joint(sns.histplot, bins=30, alpha=.7, color='m')  # the main plot

    g = g.plot_marginals(sns.histplot, stat="probability",
                     kde=True, fill=False, color='m')
    g = g.set_axis_labels("Local variation", "Error", fontsize=15)

    # Calculate the pearson correlation
    pearson_corr = pearsonr(x=np.asarray(hh_entropy_df), y=np.asarray(hh_error_df))
    sp_corr = spearmanr(hh_output)
    print(pearson_corr, sp_corr)
    g.fig.text(0.14, 0.78, r"$\rho$ = " + str(np.around(pearson_corr[0], 3)), fontsize=13,
               bbox=dict(boxstyle='round', fc="white", alpha=1))
    g.fig.text(0.14, 0.73, r"$r$ = " + str(np.around(sp_corr[0], 3)), fontsize=13,
               bbox=dict(boxstyle='round', fc="white", alpha=1))
    # plt.show()
    plt.savefig("features/synthetic/entropy/20210609_highAuto.highHete.Entropy.Error.png", dpi=300)

def hl():
    # draw the entropy
    # reshape the above df -> numpy array as the grid
    # Step 1: highAuto highHete
    hl = pd.read_excel("features/synthetic/entropy/highAuto.lowHete.entropy.xlsx")
    kernel = np.ones((3, 3))
    for i in range(0, 4):
        str_ = "gt" + str(i+1)
        grid = np.asarray(hl[str_]).reshape(25, 25)
        grid = grid + abs(np.nanmin(grid))  # Because the filter will erase these negative
        grid2 = grid / 1  # np.nanmax(grid)  # to do the standardize to [0, 1]

        str_e = "error" + str(i+1)
        error = np.abs(np.asarray(hl[str_e])).reshape(25, 25)  # pay attention to the abs.
        error2 = error / 1  # np.nanmax(error)

        mw_grid = convolve(grid2, kernel).reshape(-1, 1)
        mw_error = convolve(error2, kernel).reshape(-1, 1)

        if i == 0:
            grid_f, error_f = mw_grid, mw_error
        else:
            grid_f = np.append(grid_f, mw_grid)
            error_f = np.append(error_f, mw_error)

    hl_entropy_df = pd.DataFrame(grid_f, columns=["entropy"])
    hl_error_df = pd.DataFrame(error_f, columns=["error"])
    hl_output = pd.concat([hl_entropy_df, hl_error_df], axis=1)

    # to plot
    sns.set_theme(style="white", color_codes=True)
    g = sns.JointGrid(data=hl_output, x=hl_output["entropy"], y=hl_output["error"])
    g = g.plot_joint(sns.histplot, bins=30, alpha=.7, color='m')  # the main plot

    g = g.plot_marginals(sns.histplot, stat="probability",
                     kde=True, fill=False, color='m')
    g = g.set_axis_labels("Local variation", "Error", fontsize=15)

    # Calculate the pearson correlation
    pearson_corr = pearsonr(x=np.asarray(hl_entropy_df), y=np.asarray(hl_error_df))
    sp_corr = spearmanr(hl_output)
    print(pearson_corr, sp_corr)
    g.fig.text(0.14, 0.78, r"$\rho$ = " + str(np.around(pearson_corr[0], 3)), fontsize=13,
               bbox=dict(boxstyle='round', fc="white", alpha=1))
    g.fig.text(0.14, 0.73, r"$r$ = " + str(np.around(sp_corr[0], 3)), fontsize=13,
               bbox=dict(boxstyle='round', fc="white", alpha=1))
    # plt.show()
    plt.savefig("features/synthetic/entropy/20210609_highAuto.lowHete.Entropy.Error.png", dpi=300)


def hz():
    # draw the entropy
    # reshape the above df -> numpy array as the grid
    # Step 1: highAuto highHete
    hz = pd.read_excel("features/synthetic/entropy/highAuto.zeroHete.entropy.xlsx")
    kernel = np.ones((3, 3))
    for i in range(0, 4):
        str_ = "gt" + str(i+1)
        grid = np.asarray(hz[str_]).reshape(25, 25)
        grid = grid + abs(np.nanmin(grid))  # Because the filter will erase these negative
        grid2 = grid / 1  # np.nanmax(grid)  # to do the standardize to [0, 1]

        str_e = "error" + str(i+1)
        error = np.abs(np.asarray(hz[str_e])).reshape(25, 25)  # pay attention to the abs.
        error2 = error / 1  # np.nanmax(error)

        mw_grid = convolve(grid2, kernel).reshape(-1, 1)
        mw_error = convolve(error2, kernel).reshape(-1, 1)

        if i == 0:
            grid_f, error_f = mw_grid, mw_error
        else:
            grid_f = np.append(grid_f, mw_grid)
            error_f = np.append(error_f, mw_error)

    hz_entropy_df = pd.DataFrame(grid_f, columns=["entropy"])
    hz_error_df = pd.DataFrame(error_f, columns=["error"])
    hz_output = pd.concat([hz_entropy_df, hz_error_df], axis=1)

    # to plot
    sns.set_theme(style="white", color_codes=True)
    g = sns.JointGrid(data=hz_output, x=hz_output["entropy"], y=hz_output["error"])
    g = g.plot_joint(sns.histplot, bins=30, alpha=.7, color='m')  # the main plot

    g = g.plot_marginals(sns.histplot, stat="probability",
                     kde=True, fill=False, color='m')
    g = g.set_axis_labels("Local variation", "Error", fontsize=15)

    # Calculate the pearson correlation
    pearson_corr = pearsonr(x=np.asarray(hz_entropy_df), y=np.asarray(hz_error_df))
    sp_corr = spearmanr(hz_output)
    print(pearson_corr, sp_corr)
    g.fig.text(0.14, 0.78, r"$\rho$ = " + str(np.around(pearson_corr[0], 3)), fontsize=13,
               bbox=dict(boxstyle='round', fc="white", alpha=1))
    g.fig.text(0.14, 0.73, r"$r$ = " + str(np.around(sp_corr[0], 3)), fontsize=13,
               bbox=dict(boxstyle='round', fc="white", alpha=1))
    # plt.show()
    plt.savefig("features/synthetic/entropy/20210609_highAuto.zeroHete.Entropy.Error.png", dpi=300)


def lh():
    # draw the entropy
    # reshape the above df -> numpy array as the grid
    # Step 1: highAuto highHete
    df = pd.read_excel("features/synthetic/entropy/lowAuto.highHete.entropy.xlsx")
    kernel = np.ones((3, 3))
    for i in range(0, 4):
        str_ = "gt" + str(i+1)
        grid = np.asarray(df[str_]).reshape(25, 25)
        grid = grid + abs(np.nanmin(grid))  # Because the filter will erase these negative
        grid2 = grid / 1  # np.nanmax(grid)  # to do the standardize to [0, 1]

        str_e = "error" + str(i+1)
        error = np.abs(np.asarray(df[str_e])).reshape(25, 25)  # pay attention to the abs.
        error2 = error / 1  # np.nanmax(error)

        mw_grid = convolve(grid2, kernel).reshape(-1, 1)
        mw_error = convolve(error2, kernel).reshape(-1, 1)

        if i == 0:
            grid_f, error_f = mw_grid, mw_error
        else:
            grid_f = np.append(grid_f, mw_grid)
            error_f = np.append(error_f, mw_error)

    df_entropy_df = pd.DataFrame(grid_f, columns=["entropy"])
    df_error_df = pd.DataFrame(error_f, columns=["error"])
    df_output = pd.concat([df_entropy_df, df_error_df], axis=1)

    # to plot
    sns.set_theme(style="white", color_codes=True)
    g = sns.JointGrid(data=df_output, x=df_output["entropy"], y=df_output["error"])
    g = g.plot_joint(sns.histplot, bins=30, alpha=.7, color='m')  # the main plot

    g = g.plot_marginals(sns.histplot, stat="probability",
                     kde=True, fill=False, color='m')
    g = g.set_axis_labels("Local variation", "Error", fontsize=15)

    # Calculate the pearson correlation
    pearson_corr = pearsonr(x=np.asarray(df_entropy_df), y=np.asarray(df_error_df))
    sp_corr = spearmanr(df_output)
    print(pearson_corr, sp_corr)
    g.fig.text(0.14, 0.78, r"$\rho$ = " + str(np.around(pearson_corr[0], 3)), fontsize=13,
               bbox=dict(boxstyle='round', fc="white", alpha=1))
    g.fig.text(0.14, 0.73, r"$r$ = " + str(np.around(sp_corr[0], 3)), fontsize=13,
               bbox=dict(boxstyle='round', fc="white", alpha=1))
    # plt.show()
    plt.savefig("features/synthetic/entropy/20210609_lowAuto.highHete.Entropy.Error.png", dpi=300)


def ll():
    # draw the entropy
    # reshape the above df -> numpy array as the grid
    # Step 1: highAuto highHete
    df = pd.read_excel("features/synthetic/entropy/lowAuto.lowHete.entropy.xlsx")
    kernel = np.ones((3, 3))
    for i in range(0, 4):
        str_ = "gt" + str(i+1)
        grid = np.asarray(df[str_]).reshape(25, 25)
        grid = grid + abs(np.nanmin(grid))  # Because the filter will erase these negative
        grid2 = grid / 1  # np.nanmax(grid)  # to do the standardize to [0, 1]

        str_e = "error" + str(i+1)
        error = np.abs(np.asarray(df[str_e])).reshape(25, 25)  # pay attention to the abs.
        error2 = error / 1  # np.nanmax(error)

        mw_grid = convolve(grid2, kernel).reshape(-1, 1)
        mw_error = convolve(error2, kernel).reshape(-1, 1)

        if i == 0:
            grid_f, error_f = mw_grid, mw_error
        else:
            grid_f = np.append(grid_f, mw_grid)
            error_f = np.append(error_f, mw_error)

    df_entropy_df = pd.DataFrame(grid_f, columns=["entropy"])
    df_error_df = pd.DataFrame(error_f, columns=["error"])
    df_output = pd.concat([df_entropy_df, df_error_df], axis=1)

    # to plot
    sns.set_theme(style="white", color_codes=True)
    g = sns.JointGrid(data=df_output, x=df_output["entropy"], y=df_output["error"])
    g = g.plot_joint(sns.histplot, bins=30, alpha=.7, color='m')  # the main plot

    g = g.plot_marginals(sns.histplot, stat="probability",
                     kde=True, fill=False, color='m')
    g = g.set_axis_labels("Local variation", "Error", fontsize=15)

    # Calculate the pearson correlation
    pearson_corr = pearsonr(x=np.asarray(df_entropy_df), y=np.asarray(df_error_df))
    sp_corr = spearmanr(df_output)
    print(pearson_corr, sp_corr)
    g.fig.text(0.14, 0.78, r"$\rho$ = " + str(np.around(pearson_corr[0], 3)), fontsize=13,
               bbox=dict(boxstyle='round', fc="white", alpha=1))
    g.fig.text(0.14, 0.73, r"$r$ = " + str(np.around(sp_corr[0], 3)), fontsize=13,
               bbox=dict(boxstyle='round', fc="white", alpha=1))
    # plt.show()
    plt.savefig("features/synthetic/entropy/20210609_lowAuto.lowHete.Entropy.Error.png", dpi=300)


def lz():
    # draw the entropy
    # reshape the above df -> numpy array as the grid
    # Step 1: highAuto highHete
    df = pd.read_excel("features/synthetic/entropy/lowAuto.zeroHete.entropy.xlsx")
    kernel = np.ones((3, 3))
    for i in range(0, 4):
        str_ = "gt" + str(i+1)
        grid = np.asarray(df[str_]).reshape(25, 25)
        grid = grid + abs(np.nanmin(grid))  # Because the filter will erase these negative
        grid2 = grid / 1  # np.nanmax(grid)  # to do the standardize to [0, 1]

        str_e = "error" + str(i+1)
        error = np.abs(np.asarray(df[str_e])).reshape(25, 25)  # pay attention to the abs.
        error2 = error / 1  # np.nanmax(error)

        mw_grid = convolve(grid2, kernel).reshape(-1, 1)
        mw_error = convolve(error2, kernel).reshape(-1, 1)

        if i == 0:
            grid_f, error_f = mw_grid, mw_error
        else:
            grid_f = np.append(grid_f, mw_grid)
            error_f = np.append(error_f, mw_error)

    df_entropy_df = pd.DataFrame(grid_f, columns=["entropy"])
    df_error_df = pd.DataFrame(error_f, columns=["error"])
    df_output = pd.concat([df_entropy_df, df_error_df], axis=1)

    # to plot
    sns.set_theme(style="white", color_codes=True)
    g = sns.JointGrid(data=df_output, x=df_output["entropy"], y=df_output["error"])
    g = g.plot_joint(sns.histplot, bins=30, alpha=.7, color='m')  # the main plot

    g = g.plot_marginals(sns.histplot, stat="probability",
                     kde=True, fill=False, color='m')
    g = g.set_axis_labels("Local variation", "Error", fontsize=15)

    # Calculate the pearson correlation
    pearson_corr = pearsonr(x=np.asarray(df_entropy_df), y=np.asarray(df_error_df))
    sp_corr = spearmanr(df_output)
    print(pearson_corr, sp_corr)
    g.fig.text(0.14, 0.78, r"$\rho$ = " + str(np.around(pearson_corr[0], 3)), fontsize=13,
               bbox=dict(boxstyle='round', fc="white", alpha=1))
    g.fig.text(0.14, 0.73, r"$r$ = " + str(np.around(sp_corr[0], 3)), fontsize=13,
               bbox=dict(boxstyle='round', fc="white", alpha=1))
    # plt.show()
    plt.savefig("features/synthetic/entropy/20210609_lowAuto.zeroHete.Entropy.Error.png", dpi=300)


def zh():
    # draw the entropy
    # reshape the above df -> numpy array as the grid
    # Step 1: highAuto highHete
    df = pd.read_excel("features/synthetic/entropy/zeroAuto.highHete.entropy.xlsx")
    kernel = np.ones((3, 3))
    for i in range(0, 4):
        str_ = "gt" + str(i+1)
        grid = np.asarray(df[str_]).reshape(25, 25)
        grid = grid + abs(np.nanmin(grid))  # Because the filter will erase these negative
        grid2 = grid / 1  # np.nanmax(grid)  # to do the standardize to [0, 1]

        str_e = "error" + str(i+1)
        error = np.abs(np.asarray(df[str_e])).reshape(25, 25)  # pay attention to the abs.
        error2 = error / 1  # np.nanmax(error)

        mw_grid = convolve(grid2, kernel).reshape(-1, 1)
        mw_error = convolve(error2, kernel).reshape(-1, 1)

        if i == 0:
            grid_f, error_f = mw_grid, mw_error
        else:
            grid_f = np.append(grid_f, mw_grid)
            error_f = np.append(error_f, mw_error)

    df_entropy_df = pd.DataFrame(grid_f, columns=["entropy"])
    df_error_df = pd.DataFrame(error_f, columns=["error"])
    df_output = pd.concat([df_entropy_df, df_error_df], axis=1)

    # to plot
    sns.set_theme(style="white", color_codes=True)
    g = sns.JointGrid(data=df_output, x=df_output["entropy"], y=df_output["error"])
    g = g.plot_joint(sns.histplot, bins=30, alpha=.7, color='m')  # the main plot

    g = g.plot_marginals(sns.histplot, stat="probability",
                     kde=True, fill=False, color='m')
    g = g.set_axis_labels("Local variation", "Error", fontsize=15)

    # Calculate the pearson correlation
    pearson_corr = pearsonr(x=np.asarray(df_entropy_df), y=np.asarray(df_error_df))
    sp_corr = spearmanr(df_output)
    print(pearson_corr, sp_corr)
    g.fig.text(0.14, 0.78, r"$\rho$ = " + str(np.around(pearson_corr[0], 3)), fontsize=13,
               bbox=dict(boxstyle='round', fc="white", alpha=1))
    g.fig.text(0.14, 0.73, r"$r$ = " + str(np.around(sp_corr[0], 3)), fontsize=13,
               bbox=dict(boxstyle='round', fc="white", alpha=1))
    # plt.show()
    plt.savefig("features/synthetic/entropy/20210609_zeroAuto.highHete.Entropy.Error.png", dpi=300)


def zl():
    # draw the entropy
    # reshape the above df -> numpy array as the grid
    # Step 1: highAuto highHete
    df = pd.read_excel("features/synthetic/entropy/zeroAuto.lowHete.entropy.xlsx")
    kernel = np.ones((3, 3))
    for i in range(0, 4):
        str_ = "gt" + str(i+1)
        grid = np.asarray(df[str_]).reshape(25, 25)
        grid = grid + abs(np.nanmin(grid))  # Because the filter will erase these negative
        grid2 = grid / 1  # np.nanmax(grid)  # to do the standardize to [0, 1]

        str_e = "error" + str(i+1)
        error = np.abs(np.asarray(df[str_e])).reshape(25, 25)  # pay attention to the abs.
        error2 = error / 1  # np.nanmax(error)

        mw_grid = convolve(grid2, kernel).reshape(-1, 1)
        mw_error = convolve(error2, kernel).reshape(-1, 1)

        if i == 0:
            grid_f, error_f = mw_grid, mw_error
        else:
            grid_f = np.append(grid_f, mw_grid)
            error_f = np.append(error_f, mw_error)

    df_entropy_df = pd.DataFrame(grid_f, columns=["entropy"])
    df_error_df = pd.DataFrame(error_f, columns=["error"])
    df_output = pd.concat([df_entropy_df, df_error_df], axis=1)

    # to plot
    sns.set_theme(style="white", color_codes=True)
    g = sns.JointGrid(data=df_output, x=df_output["entropy"], y=df_output["error"])
    g = g.plot_joint(sns.histplot, bins=30, alpha=.7, color='m')  # the main plot

    g = g.plot_marginals(sns.histplot, stat="probability",
                     kde=True, fill=False, color='m')
    g = g.set_axis_labels("Local variation", "Error", fontsize=15)

    # Calculate the pearson correlation
    pearson_corr = pearsonr(x=np.asarray(df_entropy_df), y=np.asarray(df_error_df))
    sp_corr = spearmanr(df_output)
    print(pearson_corr, sp_corr)
    g.fig.text(0.14, 0.78, r"$\rho$ = " + str(np.around(pearson_corr[0], 3)), fontsize=13,
               bbox=dict(boxstyle='round', fc="white", alpha=1))
    g.fig.text(0.14, 0.73, r"$r$ = " + str(np.around(sp_corr[0], 3)), fontsize=13,
               bbox=dict(boxstyle='round', fc="white", alpha=1))
    # plt.show()
    plt.savefig("features/synthetic/entropy/20210609_zeroAuto.lowHete.Entropy.Error.png", dpi=300)


def zz():
    # draw the entropy
    # reshape the above df -> numpy array as the grid
    # Step 1: highAuto highHete
    df = pd.read_excel("features/synthetic/entropy/zeroAuto.zeroHete.entropy.xlsx")
    kernel = np.ones((3, 3))
    for i in range(0, 4):
        str_ = "gt" + str(i+1)
        grid = np.asarray(df[str_]).reshape(25, 25)
        grid = grid + abs(np.nanmin(grid))  # Because the filter will erase these negative
        grid2 = grid / 1  # np.nanmax(grid)  # to do the standardize to [0, 1]

        str_e = "error" + str(i+1)
        error = np.abs(np.asarray(df[str_e])).reshape(25, 25)  # pay attention to the abs.
        error2 = error / 1  # np.nanmax(error)

        mw_grid = convolve(grid2, kernel).reshape(-1, 1)
        mw_error = convolve(error2, kernel).reshape(-1, 1)

        if i == 0:
            grid_f, error_f = mw_grid, mw_error
        else:
            grid_f = np.append(grid_f, mw_grid)
            error_f = np.append(error_f, mw_error)

    df_entropy_df = pd.DataFrame(grid_f, columns=["entropy"])
    df_error_df = pd.DataFrame(error_f, columns=["error"])
    df_output = pd.concat([df_entropy_df, df_error_df], axis=1)

    # to plot
    sns.set_theme(style="white", color_codes=True)
    g = sns.JointGrid(data=df_output, x=df_output["entropy"], y=df_output["error"])
    g = g.plot_joint(sns.histplot, bins=30, alpha=.7, color='m')  # the main plot

    g = g.plot_marginals(sns.histplot, stat="probability",
                     kde=True, fill=False, color='m')
    g = g.set_axis_labels("Local variation", "Error", fontsize=15)

    # Calculate the pearson correlation
    pearson_corr = pearsonr(x=np.asarray(df_entropy_df), y=np.asarray(df_error_df))
    sp_corr = spearmanr(df_output)
    print(pearson_corr, sp_corr)
    g.fig.text(0.14, 0.78, r"$\rho$ = " + str(np.around(pearson_corr[0], 3)), fontsize=13,
               bbox=dict(boxstyle='round', fc="white", alpha=1))
    g.fig.text(0.14, 0.73, r"$r$ = " + str(np.around(sp_corr[0], 3)), fontsize=13,
               bbox=dict(boxstyle='round', fc="white", alpha=1))
    # plt.show()
    plt.savefig("features/synthetic/entropy/20210609_zeroAuto.zeroHete.Entropy.Error.png", dpi=300)
