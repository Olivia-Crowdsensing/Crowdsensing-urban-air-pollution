from __future__ import division
import os
import numpy as np
import scipy.sparse as sp
import pandas as pd
from math import radians, cos, sin, asin, sqrt

import joblib
import scipy.io
import torch
from torch import nn


def get_Laplace(A):
    """
    Returns the laplacian adjacency matrix. This is for C_GCN
    """
    if A[0, 0] == 1:
        A = A - np.diag(np.ones(A.shape[0], dtype=np.float32))  # if the diag has been added by 1s
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix. This is for K_GCN
    """
    if A[0, 0] == 0:
        A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))  # if the diag has been added by 1s
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5  # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def calculate_random_walk_matrix(adj_mx):
    # set the probability for random walk
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()


def test_error(STmodel, unknown_set, test_data, A_s, E_maxvalue, Missing0):
    """
    :param STmodel: The graph neural networks
    :unknow_set: The unknow locations for spatial prediction which totally deleted in training.
    :test_data: The true value test_data of shape (test_num_timesteps, num_nodes)
    :A_s: The full adjacent matrix
    :Missing0: True: 0 in original datasets means missing data
    :return: NAE, MAPE and RMSE
    """
    unknown_set = set(unknown_set)  # points that seed set as unknown.
    time_dim = STmodel.time_dimension

    test_omask = np.ones(test_data.shape)  # Put, (1637, 30% time) set as 1.
    if Missing0:
        # 如果有Missing，那么就把 test_data中为0 (因为我们 assign those pts not known as 0.)的在omask中=0
        test_omask[test_data == 0] = 0  # 把 testdata 中本来就是空缺值【0】的在mask中设为0
        # 到这里，只要不是原本数据（xlsx）就缺失，则全部为1.

    # 这样就做了一个mask，test_inputs中为0的都是del,其余都是observation
    test_inputs = (test_data * test_omask).astype('float32')
    test_inputs_s = test_inputs

    missing_index = np.ones(np.shape(test_data))
    missing_index[:, list(unknown_set)] = 0  # set unknown的全部时间刻度下的数值都为0
    missing_index_s = missing_index

    o = np.zeros([test_data.shape[0] // time_dim * time_dim,  # sth like take int to lower
                  test_inputs_s.shape[1]])  # Separate the test data into several h period

    for i in range(0, test_data.shape[0] // time_dim * time_dim, time_dim):
        inputs = test_inputs_s[i:i + time_dim, :]  # 按照time_dim / window来取数据
        missing_inputs = missing_index_s[i:i + time_dim, :]
        T_inputs = inputs * missing_inputs  # set unknown -> 0
        T_inputs = T_inputs / E_maxvalue
        T_inputs = np.expand_dims(T_inputs, axis=0)  # 输入进去的，unknown的区域是0
        T_inputs = torch.from_numpy(T_inputs.astype('float32'))
        A_q = torch.from_numpy(calculate_random_walk_matrix(A_s).T.astype('float32'))
        A_h = torch.from_numpy(calculate_random_walk_matrix(A_s.T).T.astype('float32'))

        _, _, imputation = STmodel(T_inputs, A_q, A_h)
        imputation = imputation.data.numpy()
        o[i:i + time_dim, :] = imputation[0, :, :]  # Thus, the output of the test is a global output.

    o = o * E_maxvalue  # 出来之后，原本unknown的地方已经有数值了。
    truth = test_inputs_s[0:test_data.shape[0] // time_dim * time_dim]  # 采取与上面同样的做法，因为我们希望取整的，好分块

    # step1. boolean, to pick the 经过model的数据中为1的，即不是unknown的 T/F
    # step2. 将那些不是unknown的数值，用对应的真实值去赋值过去。
    # step3. 这样，o 和 true中唯一不同的就是那些被标记成为unknown的点。
    o_p_pts = o
    truth_pts = truth
    o[missing_index_s[0:test_data.shape[0] // time_dim * time_dim] == 1] = truth[
        missing_index_s[0:test_data.shape[0] // time_dim * time_dim] == 1]

    test_mask = 1 - missing_index_s[0:test_data.shape[0] // time_dim * time_dim]  # 仅用于计数使用
    if Missing0:
        test_mask[truth == 0] = 0
        o[truth == 0] = 0

    # Thus, in reality, the difference between `o` and `truth` are those pts that set as unknown
    # Therefore, the RMSE..error measures those `unknown` pts.
    MAE = np.sum(np.abs(o - truth)) / np.sum(test_mask)
    RMSE = np.sqrt(np.sum((o - truth) * (o - truth)) / np.sum(test_mask))

    # return a pure predicton points results
    o_p_pts[missing_index_s[0:test_data.shape[0] // time_dim * time_dim] == 1] = 0
    truth_pts[missing_index_s[0:test_data.shape[0] // time_dim * time_dim] == 1] = 0
    return MAE, RMSE, o_p_pts, truth_pts
