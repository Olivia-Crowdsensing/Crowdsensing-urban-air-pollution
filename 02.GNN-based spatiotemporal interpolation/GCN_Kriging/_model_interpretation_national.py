from __future__ import division
from __future__ import print_function

import random
import pandas as pd
import torch.optim as optim
import sys

import Diffusion_Kriging.prepare_data as pre_data
from Diffusion_Kriging.basic_structure import GSTKriging
from Diffusion_Kriging.utils import *

from sklearn import manifold


# t-SNE 降维
def t_sne(output, dimension):
    # output -> data need to be reduced dimensions.
    # dimension -> to dim that we need to.
    tsne = manifold.TSNE(n_components=dimension, init='pca', random_state=0)
    result = tsne.fit_transform(output)
    return result


def reduce_dimensions_train(df, dim, flag=True):
    if not flag:
        # This means => we are going to process the output of training.
        df = df.permute(0, 2, 1)
        dim_reduced = df[0].detach().numpy()
    if flag:
        # To do dimensionality reduction
        dim_reduced = t_sne(df[0].detach().numpy(), dim)

    # First, convert from the 1200 -> 1337
    f_train_set_s = set(range(0, len(training_set_s[0])))  # this set contains 1337, structured range 0 -> 1337
    exc_training_set_s = f_train_set_s - know_mask  # These are (137) index not in training
    df_result = pd.DataFrame([know_mask, dim_reduced]).T  # This contains 500 known points
    df_exc = pd.DataFrame([exc_training_set_s, np.zeros((len(exc_training_set_s), dim)) + 9999]).T  # df (137)

    df_trains = pd.concat([df_result, df_exc], axis=0, ignore_index=True)  # This is the 1337
    df_trains.sort_values(0, inplace=True)  # resort and form from 0 - 1337
    df_trains.columns = ["index", "value"]

    # Now, to expand from 1337 to 1637
    f_training = set(range(0, len(training_set[0])))
    exc_training_set = f_training - know_set
    # 1. convert df & know_mask -> dataframe
    df_0 = pd.DataFrame([know_set, df_trains['value']]).T
    df_0_exc = pd.DataFrame([exc_training_set, np.zeros((len(exc_training_set), dim)) + 9999]).T
    df_0_f = pd.concat([df_0, df_0_exc], axis=0, ignore_index=True)
    df_0_f.sort_values(0, inplace=True)  # This is the
    df_0_f.columns = ["index", "value"]
    return df_0_f



if __name__ == '__main__':
    dataset = "pm25_China"  # `pm25_BJ`; `pm25_China`
    name = "2021-10"
    v_df = pd.read_excel("data/China/"+name + "_geo.xlsx", header=0)

    n_o_n_m, n_u, n_m, t, hd, E_maxvalue, K, lr, batch_size = pre_data.load_param(dataset)
    A, X, training_set, test_set, unknow_set, full_set, know_set, training_set_s, A_s = pre_data.load_data(dataset,
                                                                                             True, v_df)
    # Define model & make the INSTANCE
    GSTKriging_Model = GSTKriging(t, hd, K)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(GSTKriging_Model.parameters(), lr=lr)

    RMSE_list = []
    MAE_list = []
    score = None

    for epoch in range(100):
        print("# epoch: ", epoch)
        # _notes: The Algorithm 1. Subgraph signal and random mask generation.
        for i in range(training_set.shape[0] // (t * batch_size)):  # <IGNNK  alg1. l2 (for sample = 1:S do)>
            # print("subgraph #", i)
            # <IGNNK alg1. l5 [for random choose j.]>
            t_random = np.random.randint(low=0, high=(training_set_s.shape[0]-t), size=batch_size, dtype='l')
            know_mask = set(random.sample(range(0, training_set_s.shape[1]), n_o_n_m))  # <IGNNK alg1.l3,4(n_o+n_m)>
            # The above is for random sample -> indices for stations., known mask for stations.

            feed_batch = []
            # From here, we are going to see the sub-set / sub-graph.
            for j in range(batch_size):
                # <IGNNK alg1. l5, l6>
                feed_batch.append(training_set_s[t_random[j]: t_random[j]+t, :][:, list(know_mask)])
            inputs = np.array(feed_batch)
            inputs_omask = np.ones(np.shape(inputs))  # <IGNNK alg1. l8>

            missing_index = np.ones(inputs.shape)
            for j in range(batch_size):
                missing_mask = random.sample(range(0, n_o_n_m), n_m)  # Masked missing locations (300 missing)
                # From 900 <- select 300 as missing.
                missing_index[j, :, missing_mask] = 0  # <IGNNK alg1. l8>

            # o-mask -> observed mask; missing_index (missing); inputs -> all.
            Mf_inputs = inputs * inputs_omask * missing_index / E_maxvalue  # norm value with experience [E-..]
            # print(Mf_inputs.shape)  # with a shape of (4, 6, 900); (1, 3, 500) (batch; t; space)

            # _notes: The Graph - part.
            mask = torch.from_numpy(inputs_omask.astype('float32'))  # to fix some problems in data type.
            A_dynamic = A_s[list(know_mask), :][:, list(know_mask)]  # 就是一个抽取的操作，从(918,918) -> pick (900,900)

            # Then, we create two-directional Adj. matrix from this A_dynamic
            # random_walk -> the inherent part of original d-gcn paper
            A_q = torch.from_numpy(calculate_random_walk_matrix(A_dynamic).T.astype('float32'))
            A_h = torch.from_numpy(calculate_random_walk_matrix(A_dynamic.T).T.astype('float32'))
            outputs = torch.from_numpy((inputs / E_maxvalue).astype('float32'))  # The label

            # Re-change the type from nd-array -> tensor
            Mf_inputs = torch.from_numpy(Mf_inputs.astype('float32'))

            optimizer.zero_grad()
            # _notes: The model - part
            X_h1, X_h2, X_est = GSTKriging_Model(Mf_inputs, A_q, A_h)  # Obtain the reconstruction

            loss = criterion(X_est * mask, outputs * mask)
            print("Train loss:  ", loss)
            loss.backward()
            optimizer.step()  # Error backward

            # writer.add_scalars("Train loss", {'loss': loss}, epoch)

        # print("\n test graph")
        MAE_t, RMSE_t, pred_pts, truth_pts = test_error(GSTKriging_Model, unknow_set,
                                                               test_set, A, E_maxvalue, True)
        print("\n Test Error: ")
        RMSE_list.append(RMSE_t)
        MAE_list.append(MAE_t)

        pred_df = pd.DataFrame(pred_pts)[unknow_set]
        truth_df = pd.DataFrame(truth_pts)[unknow_set]
        acc_df = pred_df - truth_df

        print(MAE_t, RMSE_t)

        # Early stop and save checkpoints
        if score is None:
            score = MAE_t
        elif score >= MAE_t:
            score = MAE_t
        elif score < MAE_t:
            # torch.save(GSTKriging_Model.state_dict(), "GSTKriging.earlystop.pth")
            break
            # pass

        # torch.save(GSTKriging_Model.state_dict(), "syn_model_parameter.pth")
        # torch.load("syn_model_parameter.pth")


# reduce_dimensions_train(outputs, dim=1, flag=False).to_excel("2021-01.gT.xlsx")
# reduce_dimensions_train(X_est, dim=1, flag=False).to_excel("2021-01.est.xlsx")
# reduce_dimensions_train(X_h1, dim=3).to_excel("2021-01.hidden1.xlsx")
# reduce_dimensions_train(X_h2, dim=3).to_excel("2021-01.hidden2.xlsx")