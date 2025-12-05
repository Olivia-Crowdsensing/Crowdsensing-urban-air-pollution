from __future__ import division
from __future__ import print_function

import random
import torch.optim as optim

import Diffusion_Kriging.prepare_data as pre_data
from Diffusion_Kriging.basic_structure import GSTKriging
from Diffusion_Kriging.utils import *


if __name__ == '__main__':
    dataset = "synthetic"  # `pm25_BJ`; `pm25_China`; synthetic
    # name = "2022-01"
    # v_df = pd.read_excel("data/China/"+name + "_geo.xlsx", header=0)
    name = "highAuto_highHete"
    v_df = pd.read_excel("data/Synthetic/" + name + ".xlsx", header=0)

    n_o_n_m, n_u, n_m, t, hd, E_maxvalue, K, lr, batch_size = pre_data.load_param(dataset)
    A, X, training_set, test_set, unknow_set, full_set, know_set, training_set_s, A_s = pre_data.load_data(dataset,
                                                                                             True, v_df)

    # Define model & make the INSTANCE
    GSTKriging_Model = GSTKriging(t, hd, K)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(GSTKriging_Model.parameters(), lr=lr)

    RMSE_list = []
    MAE_list = []
    score = None  # For early stop

    for epoch in range(1):
        print("# epoch: ", epoch)
        # _notes: The Algorithm 1. Subgraph signal and random mask generation.
        for i in range(training_set.shape[0] // (t * batch_size)):  # <IGNNK  alg1. l2 (for sample = 1:S do)>
            # <IGNNK alg1. l5 [for random choose j.]>
            t_random = np.random.randint(low=0, high=(training_set_s.shape[0]-t), size=batch_size, dtype='l')
            know_mask = set(random.sample(range(0, training_set_s.shape[1]), n_o_n_m))  # <IGNNK alg1.l3,4(n_o+n_m)>
            # The above is for random sample -> indices for stations, known mask for stations.

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
            A_dynamic = A_s[list(know_mask), :][:, list(know_mask)]  # pick-out operation. From (918,918)->(900,900)

            # Then, we create two-directional Adj. matrix from this A_dynamic
            # random_walk -> the inherent part of original d-gcn paper
            A_q = torch.from_numpy(calculate_random_walk_matrix(A_dynamic).T.astype('float32'))
            A_h = torch.from_numpy(calculate_random_walk_matrix(A_dynamic.T).T.astype('float32'))
            outputs = torch.from_numpy((inputs / E_maxvalue).astype('float32'))  # The label

            # Re-change the type from nd-array -> tensor
            Mf_inputs = torch.from_numpy(Mf_inputs.astype('float32'))

            optimizer.zero_grad()
            # _notes: The model - part
            _, _, X_est = GSTKriging_Model(Mf_inputs, A_q, A_h)  # Obtain the reconstruction

            loss = criterion(X_est * mask, outputs * mask)
            print("Train loss:  ", loss)
            loss.backward()
            optimizer.step()  # Error backward

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


# pred_df.T.to_excel(name + ".predict.xlsx")
# truth_df.T.to_excel(name + ".gt.xlsx")