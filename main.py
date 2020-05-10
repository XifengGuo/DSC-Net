import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from post_clustering import spectral_clustering, acc, nmi


class Encoder(nn.Module):
    def __init__(self, in_channel):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 15, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        return y


class Decoder(nn.Module):
    def __init__(self, out_channel):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(15, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        y = F.relu(self.deconv1(x))
        return y


class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)

    def forward(self, x):  # shape=[n, d]
        y = torch.matmul(self.Coefficient, x)
        return y


class AE(nn.Module):
    def __init__(self, num_channel):
        super(AE, self).__init__()
        self.encoder = Encoder(num_channel)
        self.decoder = Decoder(num_channel)

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)


class DSCNet(nn.Module):
    def __init__(self, num_sample, num_channel):
        super(DSCNet, self).__init__()
        self.n, self.c, = num_sample, num_channel
        self.encoder = Encoder(self.c)
        self.self_expression = SelfExpression(self.n)
        self.decoder = Decoder(self.c)

    def forward(self, x):  # shape=[n, c, w, h]
        z = self.encoder(x)

        # self expression layer, reshape to vectors, multiply Coefficient, then reshape back
        shape = z.shape
        z = z.view(self.n, -1)  # shape=[n, d]
        z_recon = self.self_expression(z)  # shape=[n, d]
        z_recon_reshape = z_recon.view(shape)

        x_recon = self.decoder(z_recon_reshape)  # shape=[n, c, w, h]
        return x_recon, z, z_recon

    def loss_fn(self, x, x_recon, z, z_recon, weight_coef, weight_selfExp):
        loss_ae = F.mse_loss(x_recon, x, reduction='sum')
        loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        loss_selfExp = 0.5 * F.mse_loss(z_recon, z, reduction='sum')
        loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp
        loss /= x.size(0)  # just control the range, does not affect the optimization.
        return loss


def train(model,  # type: DSCNet
          x, y, epochs, lr=1e-3, device='cuda'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    if isinstance(y, torch.Tensor):
        y = y.to('cpu').numpy()
    K = len(np.unique(y))
    model.to(device)
    for epoch in range(epochs):
        x_recon, z, z_recon = model(x)
        loss = model.loss_fn(x, x_recon, z, z_recon, weight_coef=1.0, weight_selfExp=150)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            C = model.self_expression.Coefficient.detach().to('cpu').numpy()
            y_pred = spectral_clustering(C, K, 12, 8, 0.04)
            print('Epoch %02d: loss=%.4f, acc=%.4f, nmi=%.4f' % (epoch, loss.item(), acc(y, y_pred), nmi(y, y_pred)))


# load data
import scipy.io as sio

data = sio.loadmat('datasets/COIL20.mat')
x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
data_shape = x.shape

ae = AE(data_shape[1])

dscnet = DSCNet(num_sample=data_shape[0], num_channel=data_shape[1])

# load the pretrained weights which are provided by the original author in
# https://github.com/panji1990/Deep-subspace-clustering-networks
dscnet_dict = dscnet.state_dict()
pretrain_weights = torch.load('pretrained_weights_original/coil20.pkl')
state_dict = {k:v for k,v in pretrain_weights.items() if k in dscnet_dict.keys()}
dscnet_dict.update(state_dict)  # filter out the Coefficient parameter which is not pretrained.
dscnet.load_state_dict(dscnet_dict)

train(dscnet, x, y, 30, device='cpu')
