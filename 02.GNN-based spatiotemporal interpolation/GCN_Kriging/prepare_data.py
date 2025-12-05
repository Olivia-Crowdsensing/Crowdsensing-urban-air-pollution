import pandas as pd
import numpy as np

import torch
from sklearn import metrics


def prepare_graph(e_df, o_df, dataset_flag):
    global src, dst
    if dataset_flag == 'BJ':
        nodes_idx = e_df['V_idx']  # nodes idx
        e_start = 7  # edge start [nb1] from here. -> in BJ dataset

        # To prepare edges iteratively through nodes
        for idx in nodes_idx:
            V_idx = idx
            dst_tmp = e_df.loc[V_idx][e_start:].dropna().values.astype(int)
            src_tmp = np.zeros((len(dst_tmp)), dtype=int)
            src_tmp[:] = V_idx

            if idx == 0:
                src = src_tmp
                dst = dst_tmp
            else:
                src = np.concatenate((src, src_tmp), axis=0)
                dst = np.concatenate((dst, dst_tmp), axis=0)

        A = np.zeros((e_df.shape[0], e_df.shape[0]))  # construct the Adj - matrix
        for i, _ in enumerate(src):
            A[src[i], dst[i]] = 1

        # Then, to prepare the feats for the graph.
        feat_df = o_df.iloc[:, 6:6+38]  # Currently, we use 1 month to predict 1 week  # 335:361;6:6 + 38
        X = feat_df.values
    if dataset_flag == "China":
        nodes_idx = e_df['V_idx']  # nodes idx
        e_start = 6  # edge start [nb1] from here -> China dataset

        for idx in nodes_idx:
            V_idx = idx
            dst_tmp = e_df.loc[V_idx][e_start:].dropna().values.astype(int)
            src_tmp = np.zeros((len(dst_tmp)), dtype=int)
            src_tmp[:] = V_idx

            if idx == 0:
                src = src_tmp
                dst = dst_tmp
            else:
                src = np.concatenate((src, src_tmp), axis=0)
                dst = np.concatenate((dst, dst_tmp), axis=0)

        A = np.zeros((e_df.shape[0], e_df.shape[0]))  # construct the Adj-matrix
        for i, _ in enumerate(src):
            A[src[i], dst[i]] = 1

        # Then, to prepare the feats for the graph.
        # todo: This is important to feed as it determines the length of the temporal.
        feat_df = o_df.iloc[:, 8: 8 + (24 * 30)]  # Currently, we use 24 hours * 3 days; the (24*3 hours - 6) as
        X = feat_df.values

    if dataset_flag == "synthetic":
        nodes_idx = e_df["V_idx"]  # nodes index
        e_start = 4  # edge start from [nb1] here -> Synthetic dataset

        for idx in nodes_idx:
            V_idx = idx
            dst_tmp = e_df.loc[V_idx][e_start:].dropna().values.astype(int)
            src_tmp = np.zeros((len(dst_tmp)), dtype=int)
            src_tmp[:] = V_idx

            if idx == 0:
                src = src_tmp
                dst = dst_tmp
            else:
                src = np.concatenate((src, src_tmp), axis=0)
                dst = np.concatenate((dst, dst_tmp), axis=0)

        A = np.zeros((e_df.shape[0], e_df.shape[0]))  # construct the Adj-matrix
        for i, _ in enumerate(src):
            A[src[i], dst[i]] = 1

        # Then, to prepare the feats for the graph.
        feat_df = o_df.iloc[:, 3: 3 + 15]  # Currently, we use 6 time intervals * 3 ranges.
        X = feat_df.values
    return A, X


def load_data(dataset, edge_flag, df):
    global split_line1, X
    if dataset == 'pm25_BJ':
        # df = pd.read_excel("data/BJ/2022_Daily_AverageT2.xlsx", header=0)
        n_e_df = pd.read_excel("data/BJ/2022_BJ_edge.xlsx", header=0)
        A, X = prepare_graph(n_e_df, df, dataset_flag="BJ")
        if edge_flag:
            # Don't forget the add edge attributes
            vector = np.vstack((df['lat'], df['lng'])).T
            distance_pairs = metrics.pairwise_distances(X=vector)
            A = A * np.exp(-distance_pairs/2)  # Here, the 2 should be set during the training
            # add a self-loop
            np.fill_diagonal(A, 1.0)
        else:
            pass
        X = X / 1  # a type of normalization? -> just abandon it.
        split_line1 = int(X.shape[1] * 0.8)  # split line to Training (80%) & Test set (20%).

    elif dataset == 'pm25_China':
        n_e_df = pd.read_excel("data/China/MS_Edge_byIDX.xlsx", header=0)
        A, X = prepare_graph(n_e_df, df, dataset_flag='China')

        if edge_flag:
            # need to set edge attributes
            vector = np.vstack((df['lat'], df['lng'])).T
            distance_pairs = metrics.pairwise_distances(X=vector)
            A = A * np.exp(-distance_pairs/2)  # Here, the 2 should be set during the training
            # add a self-loop
            np.fill_diagonal(A, 1.0)
        else:
            pass
        print(X.shape)
        X = X / 1  # a type of normalization?
        split_line1 = int(X.shape[1] * 0.7)  # split line to Training (70%) & Test set (30%).

    elif dataset == "synthetic":
        n_e_df = pd.read_excel("./data/Synthetic/neighbour_edge.xlsx", header=0)

        # to prepare graph for synthetic dataset
        A, X = prepare_graph(n_e_df, df, dataset_flag="synthetic")
        if edge_flag:
            # to set edge attributes
            vector = np.vstack((df['x'], df['y'])).T
            distance_pairs = metrics.pairwise_distances(X=vector)
            A = A * np.exp(-distance_pairs / 2)  # exponential distance
            # add a self-loop
            np.fill_diagonal(A, 1.0)
        else:
            pass
        X = X / 1  # a normalization (but, in the synthetic case, we don't need to)
        split_line1 = int(X.shape[1] * 0.7)  # training 0.7; test 0.3; separate for temporal

    training_set = X[:, :split_line1].transpose()  # Training dataset -> transpose
    test_set = X[:, split_line1:].transpose()  # Test dataset -> transpose

    rand = np.random.RandomState(0)  # Fixed random output [生成相同的伪随机数]-RandomState(MT19937)
    unknow_set = rand.choice(list(range(0, X.shape[0])), n_u, replace=False)  # n_u's unknow set.[从sensors里挑选n_u点]
    unknow_set = set(unknow_set)  # Those to be predicted (testing unobserved sample)
    full_set = set(range(0, X.shape[0]))  # total sensors.
    know_set = full_set - unknow_set

    training_set_s = training_set[:, list(know_set)]  # get the training data in the sample time period (training_set)
    # ps: the training_set_s -> row: how many time slices; column: how many sensors.

    A_s = A[:, list(know_set)][list(know_set), :]  # get the observed adjacent matrix from the full adjacent matrix,
    # the adjacent matrix are based on pairwise distance,

    # so we `need not` to construct it for each batch, we just use index to find the dynamic adjacent matrix
    return A, X, training_set, test_set, unknow_set, full_set, know_set, training_set_s, A_s


def load_param(dataset_name):
    global n_o_n_m, n_u, n_m, h, z, E_maxvalue, K, learning_rate, batch_size
    if dataset_name == 'pm25_BJ':
        n_o_n_m = 27  # sampled space dimension
        n_u = 6  # target locations, n_u locations will be deleted from the training dataset.
        n_m = 0  # number of missing mask node during training# target locations, n_u locations will be deleted from the training datase
        h = 6  # sampled time dimension [time window = 6 days]
        z = 75  # hidden dimension for graph convolution
        E_maxvalue = 50
        K = 2  # order.
        learning_rate = 0.0001
        batch_size = 4  # batch size

    elif dataset_name == 'pm25_China':
        n_o_n_m = 1200  # sampled space dimension [1200]
        n_u = 300  # target locations, n_u locations will be deleted from the training dataset.
        n_m = 10  # number of missing mask node during training
        h = 15  # Set how many as an integral to be the fully connected;
            # sampled time dimension [time window = 24 hours]
        z = 100  # hidden dimension for graph convolution
        E_maxvalue = 50
        K = 2  # order.
        learning_rate = 0.0001
        batch_size = 6  # batch sizechcp # original [6]

    elif dataset_name == 'synthetic':
        n_o_n_m = 500  # sampled space dimension
        n_u = 100  # target locations, n_u locations will be deleted from the training dataset.
        n_m = 25  # number of missing mask node during training
        h = 4  # sampled time dimension [time intervals]
        z = 100  # hidden units for graph convolution
        E_maxvalue = 1
        K = 2  # order of diffusion convolution
        learning_rate = 0.001
        batch_size = 1  # batch size

    return n_o_n_m, n_u, n_m, h, z, E_maxvalue, K, learning_rate, batch_size

