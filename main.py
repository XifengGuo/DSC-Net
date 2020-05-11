import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from post_clustering import spectral_clustering, acc, nmi
import scipy.io as sio


# class Encoder(nn.Module):
#     def __init__(self, channels, kernels):
#         """
#         :param channels: a list containing all channels including the input image channel (1 for gray, 3 for RGB)
#         :param kernels:  a list containing all kernel sizes, it should satisfy: len(kernels) = len(channels) - 1.
#         """
#         super(Encoder, self).__init__()
#         assert isinstance(channels, list) and isinstance(kernels, list)
#         self.layers = []
#         for i in range(1, len(channels)):
#             self.layers.append(nn.Conv2d(channels[i - 1], channels[i], kernel_size=kernels[i - 1], stride=2, padding=1))
#             self.layers.append(nn.ReLU(True))
#         self.model = nn.Sequential(*self.layers)
#
#     def forward(self, x):
#         return self.model(x)
#
#
# class Decoder(nn.Module):
#     def __init__(self, channels, kernels):
#         """
#         :param channels: a list containing all channels including reconstructed image channel (1 for gray, 3 for RGB)
#         :param kernels: a list containing all kernel sizes, it should satisfy: len(kernels) = len(channels) - 1.
#         """
#         super(Decoder, self).__init__()
#         assert isinstance(channels, list) and isinstance(kernels, list)
#         self.layers = []
#         for i in range(len(channels) - 1):
#             self.layers.append(nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernels[i], stride=2,
#                                                   padding=1, output_padding=1))
#
#     def forward(self, x):
#         for layer in self.layers:
#             x = F.relu(layer(x))
#         return x


class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)

    def forward(self, x):  # shape=[n, d]
        y = torch.matmul(self.Coefficient, x)
        return y


class ConvAE(nn.Module):
    def __init__(self, channels, kernels):
        """
        :param channels: a list containing all channels including the input image channel (1 for gray, 3 for RGB)
        :param kernels:  a list containing all kernel sizes, it should satisfy: len(kernels) = len(channels) - 1.
        """
        super(ConvAE, self).__init__()
        assert isinstance(channels, list) and isinstance(kernels, list)
        self.encoder = nn.Sequential()
        for i in range(1, len(channels)):
            #  Each layer will divide the size of feature map by 2
            self.encoder.add_module(
                'conv%d' % i,
                nn.Conv2d(channels[i - 1], channels[i], kernel_size=kernels[i - 1], stride=2,
                          padding=int(kernels[i - 1] / 2))
            )
            self.encoder.add_module('relu%d' % i, nn.ReLU(True))

        self.decoder = nn.Sequential()
        channels = list(reversed(channels))
        for i in range(len(channels) - 1):
            # Each layer will double the size of feature map
            self.decoder.add_module(
                'deconv%d' % (i + 1),
                nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernels[i], stride=2,
                                   padding=int(kernels[i] / 2), output_padding=1)
            )
            self.decoder.add_module('relud%d' % i, nn.ReLU(True))

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y


class DSCNet(nn.Module):
    def __init__(self, channels, kernels, num_sample):
        super(DSCNet, self).__init__()
        self.n = num_sample
        self.ae = ConvAE(channels, kernels)
        self.self_expression = SelfExpression(self.n)

    def forward(self, x):  # shape=[n, c, w, h]
        z = self.ae.encoder(x)

        # self expression layer, reshape to vectors, multiply Coefficient, then reshape back
        shape = z.shape
        z = z.view(self.n, -1)  # shape=[n, d]
        z_recon = self.self_expression(z)  # shape=[n, d]
        z_recon_reshape = z_recon.view(shape)

        x_recon = self.ae.decoder(z_recon_reshape)  # shape=[n, c, w, h]
        return x_recon, z, z_recon

    def loss_fn(self, x, x_recon, z, z_recon, weight_coef, weight_selfExp):
        loss_ae = F.mse_loss(x_recon, x, reduction='sum')
        loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        loss_selfExp = 0.5 * F.mse_loss(z_recon, z, reduction='sum')
        loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp
        loss /= x.size(0)  # just control the range, does not affect the optimization.
        return loss


def train(model,  # type: DSCNet
          x, y, epochs, lr=1e-3, weight_coef=1.0, weight_selfExp=150, device='cuda',
          alpha=0.04, dim_subspace=12, ro=8):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    if isinstance(y, torch.Tensor):
        y = y.to('cpu').numpy()
    K = len(np.unique(y))
    model.to(device)
    for epoch in range(epochs):
        x_recon, z, z_recon = model(x)
        loss = model.loss_fn(x, x_recon, z, z_recon, weight_coef=weight_coef, weight_selfExp=weight_selfExp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            C = model.self_expression.Coefficient.detach().to('cpu').numpy()
            y_pred = spectral_clustering(C, K, dim_subspace, alpha, ro)
            print('Epoch %02d: loss=%.4f, acc=%.4f, nmi=%.4f' % (epoch, loss.item(), acc(y, y_pred), nmi(y, y_pred)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='DSCNet')
    parser.add_argument('--db', default='coil20',
                        choices=['coil20', 'coil100', 'orl', 'reuters10k', 'stl'])
    parser.add_argument('--ae-weights', default=None)
    parser.add_argument('--save-dir', default='results')
    args = parser.parse_args()
    print(args)
    import os

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    db = args.db
    if db == 'coil20':
        # load data
        data = sio.loadmat('datasets/COIL20.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]

        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 15]
        kernels = [3]
        epochs = 30
        weight_coef = 1.0
        weight_selfExp = 150

        # post clustering parameters
        alpha = 0.04  # threshold of C
        dim_subspace = 12  # dimension of each subspace
        ro = 8  #
    elif db == 'coil100':
        # load data
        data = sio.loadmat('datasets/COIL100.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]

        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 50]
        kernels = [5]
        epochs = 120
        weight_coef = 1.0
        weight_selfExp = 30

        # post clustering parameters
        alpha = 0.04  # threshold of C
        dim_subspace = 12  # dimension of each subspace
        ro = 8  #
    elif db == 'orl':
        # load data
        data = sio.loadmat('datasets/ORL_32x32.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 3, 3, 5]
        kernels = [3, 3, 3]
        epochs = 700
        weight_coef = 1.0
        weight_selfExp = 0.1

        # post clustering parameters
        alpha = 0.2  # threshold of C
        dim_subspace = 3  # dimension of each subspace
        ro = 1  #

    dscnet = DSCNet(num_sample=num_sample, channels=channels, kernels=kernels)

    # load the pretrained weights which are provided by the original author in
    # https://github.com/panji1990/Deep-subspace-clustering-networks
    ae_state_dict = torch.load('pretrained_weights_original/%s.pkl' % db)
    dscnet.ae.load_state_dict(ae_state_dict)
    print("Pretrained ae weights are loaded successfully.")
    print(dscnet.ae.state_dict())
    a = np.arange(32 * 32).reshape((1, 1, 32, 32)).astype(np.float)
    a = torch.tensor(a, dtype=torch.float32)
    print(a.size())
    print(dscnet.ae.encoder(a))
    #
    # train(dscnet, x, y, epochs, weight_coef=weight_coef, weight_selfExp=weight_selfExp,
    #       alpha=alpha, dim_subspace=dim_subspace, ro=ro, device='cpu')
