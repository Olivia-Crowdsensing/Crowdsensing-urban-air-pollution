import pandas as pd
import numpy as np

baseline_folder = "../Baselines/china/"
gcn_folder = "./accuracy/china/"


# The root mean square error
def RMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def mae(predictions, targets):
    mae = np.sum(np.absolute(targets - predictions)) / len(targets)
    return mae


month = "12"
gt = pd.read_excel(gcn_folder + "2021-" + str(month) + ".gt.xlsx")  # ground-truth
pred_ok = pd.read_excel(baseline_folder + "2021-" + str(month) + ".OK.est.xlsx")  # the ok
pred_gcn = pd.read_excel(gcn_folder + "2021-" + str(month) + ".predict.xlsx")  # the gcn

predictions = pred_gcn.iloc[:, 1:].loc[:].T  # the first row
targets = gt.iloc[:, 1:].loc[:].T  # the row in ft dataframe
# convert 2 np asarray
predictions_np = np.asarray(predictions)
targets_np = np.asarray(targets)
# convert 2 datafra,e
predictions_df = pd.DataFrame(predictions_np)
targets_df = pd.DataFrame(targets_np)

# to calculate the rmse
out = RMSE(predictions_df, targets_df)
out_df = pd.DataFrame(out).to_excel("out.xlsx")


def china_general():
    month = "02"
    gt = pd.read_excel(gcn_folder + "2021-" + str(month) + ".gt.xlsx")  # ground-truth
    pred_ok = pd.read_excel(baseline_folder + "2021-" + str(month) + ".OK.est.xlsx")  # the ok
    output_ok = RMSE(predictions=np.asarray(pred_ok.iloc[:, 1:]),
                     targets=np.asarray(gt.iloc[:, 1:]))
    output_ok_mae = mae(predictions=np.asarray(pred_ok.iloc[:, 1:]),
                     targets=np.asarray(gt.iloc[:, 1:]))
    print("ok", output_ok_mae)

    pred_gcn = pd.read_excel(gcn_folder + "2021-" + str(month) + ".predict.xlsx")
    output_gcn = RMSE(predictions=np.asarray(pred_gcn.iloc[:, 1:]),
                      targets=np.asarray(gt.iloc[:, 1:]))
    output_gcn_mae = mae(predictions=np.asarray(pred_gcn.iloc[:, 1:]),
                      targets=np.asarray(gt.iloc[:, 1:]))
    print("gcn", output_gcn_mae)



def synthetic():
    namelist = ["highAuto_highHete", "highAuto_lowHete", "highAuto_zeroHete",
                "lowAuto_highHete", "lowAuto_lowHete", "lowAuto_zeroHete",
                "zeroAuto_highHete", "zeroAuto_lowHete", "zeroAuto_zeroHete"]

    for name in namelist:
        print("\n", name)
        gt = pd.read_excel(gcn_folder + name + ".gt.xlsx")  # ground-truth
        pred_ok = pd.read_excel(baseline_folder + name + ".OK.est.xlsx")  # the ok
        pred_3dok = pd.read_excel(baseline_folder + name + ".OK3d.est.xlsx")  # the 3d-ok

        print("ok")
        output = RMSE(predictions=pred_ok.iloc[:, 1:5], targets=gt.iloc[:, 1:5])
        print(pd.DataFrame(output).T)

        print("3d ok")
        output = RMSE(predictions=pred_3dok.iloc[:, 1:5], targets=gt.iloc[:, 1:5])
        print(pd.DataFrame(output).T)
