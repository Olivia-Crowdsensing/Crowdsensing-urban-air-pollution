import torch
import torch.nn as nn
import torch.nn.functional as F

import math


# to construct the diffusion_GCN D_GCN model - block
# This D_GCN layer contains (convolution on nodes& feature + layer)
# This D_GCN actually focus on frequency domain (I think)
class D_GCN(nn.Module):
    def __init__(self, in_channels, out_channels, orders, activation="relu"):
        """
        in_channels: number of time step
        out_channels: desired number of output features at each node in each time step
        """
        super(D_GCN, self).__init__()
        self.orders = orders  # the diffusion step.
        self.activation = activation
        self.num_matrices = 2 * self.orders + 1  # 2 diffusion + 1 self

        # transmission matrix [diffuse process + reverse one] + adds for x itself.
        self.Theta = nn.Parameter(torch.FloatTensor(in_channels * self.num_matrices,
                                                    out_channels))  # <Diffusion GCN> Eqn. 2-3
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        # parameters initialization.
        stdv = 1. / math.sqrt(self.Theta.shape[1])
        self.Theta.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)

    def _concat(self, x, x_):
        x_ = x_.unsqueeze(0)  # 对数据维度进行扩充，在x_的0的位置，加上一个维数为1的维度。
        return torch.cat([x, x_], dim=0)

    # forward() -> message passing
    def forward(self, X, A_q, A_h):
        """
        X: Input features (batch_size, num_nodes, num_timesteps)
        A_q: The forward random walk matrix
        A_h: The backward random walk matrix

        Returns:
            Output data of shape (batch_size, num_nodes, num_features)
        """
        batch_size, num_node = X.shape[0], X.shape[1]
        t_size = X.size(2)  # time_length

        x0 = X.permute(1, 2, 0)  # (num_nodes, num_times, batch_size)
        x0 = torch.reshape(x0, shape=[num_node, t_size * batch_size])
        x = torch.unsqueeze(x0, dim=0)  # [1, nodes, inp_dim*batch]

        for A_w in [A_q, A_h]:
            x1 = torch.mm(A_w, x0)
            x = self._concat(x, x1)
            for k in range(2, self.orders + 1):  # the following gives the chebyshev/ chebnet approximation
                x2 = 2 * torch.mm(A_w, x1) - x0  # This is the T_k(x) in <IGNNK, over Fig.2>
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        x = torch.reshape(x, shape=[self.num_matrices, num_node, t_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, t_size, order)
        x = torch.reshape(x, shape=[batch_size, num_node, t_size * self.num_matrices])
        x = torch.matmul(x, self.Theta)  # (batch_size * self._num_nodes, output_size)
        x += self.bias
        if self.activation == 'relu':
            x = F.relu(x)
        return x


class GSTKriging(nn.Module):

    def __init__(self, t, hd, k):
        super(GSTKriging, self).__init__()
        self.time_dimension = t  # time window
        self.hidden_dimension = hd  # hidden dimension
        self.order = k  # the diffusion step

        self.GNN1 = D_GCN(self.time_dimension, self.hidden_dimension, self.order)
        self.GNN2 = D_GCN(self.hidden_dimension, self.hidden_dimension, self.order)
        self.GNN3 = D_GCN(self.hidden_dimension, self.time_dimension, self.order,
                          activation='linear')

    def forward(self, X, A_q, A_h):
        """
        Args:
            X: input data of shape (batch_size, num_timedim, num_nodes)
            A_q: the forward random walk matrix (num_nodes, num_nodes)
            A_h: the backward random walk matrix (num_nodes, num_nodes)
        Returns:
            Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """
        X_S = X.permute(0, 2, 1)  # to transpose / exchange the input dims

        X_s1 = self.GNN1(X_S, A_q, A_h)  # Here -> (batch_size, num_nodes, num_timedim)
        # The reason why use more than 1 layer: masked nodes only pass 0 -> its neighbors in the first layer.
        # The reason add X_s1 -> X_s2: because, H1 contains information about sensors with missing data
        # print("Hidden layer 1 output: ", X_s1.shape)

        X_s2 = self.GNN2(X_s1, A_q, A_h) # + X_s1  # num_nodes, rank
        # print("Hidden layer 2 output: ", X_s2.shape)

        X_s3 = self.GNN3(X_s2, A_q, A_h)

        X_reset = X_s3.permute(0, 2, 1)  # Turn back: (batch_size, num_timdim, num_nodes)
        return X_s1, X_s2, X_reset
