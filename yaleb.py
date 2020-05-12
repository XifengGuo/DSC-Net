"""
By Xifeng Guo (guoxifeng1990@163.com), May 13, 2020.
All rights reserved.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from post_clustering import spectral_clustering, acc, nmi
import scipy.io as sio
import math


class Conv2dSamePad(nn.Module):
    """
    Implement Tensorflow's 'SAME' padding mode in Conv2d.
    When an odd number, say `m`, of pixels are need to pad, Tensorflow will pad one more column at right or one more
    row at bottom. But Pytorch will pad `m+1` pixels, i.e., Pytorch always pads in both sides.
    So we can pad the tensor in the way of Tensorflow before call the Conv2d module.
    """
    def __init__(self, kernel_size, stride):
        super(Conv2dSamePad, self).__init__()
        self.kernel_size = kernel_size if type(kernel_size) in [list, tuple] else [kernel_size, kernel_size]
        self.stride = stride if type(stride) in [list, tuple] else [stride, stride]

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        out_height = math.ceil(float(in_height) / float(self.stride[0]))
        out_width = math.ceil(float(in_width) / float(self.stride[1]))
        pad_along_height = ((out_height - 1) * self.stride[0] + self.kernel_size[0] - in_height)
        pad_along_width = ((out_width - 1) * self.stride[1] + self.kernel_size[1] - in_width)
        pad_top = math.floor(pad_along_height / 2)
        pad_left = math.floor(pad_along_width / 2)
        pad_bottom = pad_along_height - pad_top
        pad_right = pad_along_width - pad_left
        return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom], 'constant', 0)


class ConvTranspose2dSamePad(nn.Module):
    """
    This module implements the "SAME" padding mode for ConvTranspose2d as in Tensorflow.
    A tensor with width w_in, feed it to ConvTranspose2d(ci, co, kernel, stride), the width of output tensor T_nopad:
        w_nopad = (w_in - 1) * stride + kernel
    If we use padding, i.e., ConvTranspose2d(ci, co, kernel, stride, padding, output_padding), the width of T_pad:
        w_pad = (w_in - 1) * stride + kernel - (2*padding - output_padding) = w_nopad - (2*padding - output_padding)
    Yes, in ConvTranspose2d, more padding, the resulting tensor is smaller, i.e., the padding is actually deleting row/col.
    If `pad`=(2*padding - output_padding) is odd, Pytorch deletes more columns in the left, i.e., the first ceil(pad/2) and
    last `pad - ceil(pad/2)` columns of T_nopad are deleted to get T_pad.
    In contrast, Tensorflow deletes more columns in the right, i.e., the first floor(pad/2) and last `pad - floor(pad/2)`
    columns are deleted.
    For the height, Pytorch deletes more rows at top, while Tensorflow at bottom.
    In practice, we usually want `w_pad = w_in * stride` or `w_pad = w_in * stride - 1`, i.e., the "SAME" padding mode
    in Tensorflow. To determine the value of `w_pad`, we should pass it to this function.
    So the number of columns to delete:
        pad = 2*padding - output_padding = w_nopad - w_pad
    If pad is even, we can directly set padding=pad/2 and output_padding=0 in ConvTranspose2d.
    If pad is odd, we can use ConvTranspose2d to get T_nopad, and then delete `pad` rows/columns by
    ourselves.
    This module should be called after the ConvTranspose2d module with shared kernel_size and stride values.
    """

    def __init__(self, output_size):
        super(ConvTranspose2dSamePad, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        in_height = x.size(2)
        in_width = x.size(3)
        pad_height = in_height - self.output_size[0]
        pad_width = in_width - self.output_size[1]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return x[:, :, pad_top:in_height - pad_bottom, pad_left: in_width - pad_right]


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
            self.encoder.add_module('pad%d' % i, Conv2dSamePad(kernels[i - 1], 2))
            self.encoder.add_module(
                'conv%d' % i,
                nn.Conv2d(channels[i - 1], channels[i], kernel_size=kernels[i - 1], stride=2)
            )
            self.encoder.add_module('relu%d' % i, nn.ReLU(True))

        self.decoder = nn.Sequential()
        channels = list(reversed(channels))
        kernels = list(reversed(kernels))
        sizes = [[12, 11], [24, 21], [48, 42]]
        for i in range(len(channels) - 1):
            # Each layer will double the size of feature map
            self.decoder.add_module(
                'deconv%d' % (i + 1),
                nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernels[i], stride=2)
            )
            self.decoder.add_module('padd%d' % i, ConvTranspose2dSamePad(sizes[i]))
            self.decoder.add_module('relud%d' % i, nn.ReLU(True))

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y


class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0e-4 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)

    def forward(self, x):  # shape=[n, d]
        y = torch.matmul(self.Coefficient, x)
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
        loss_ae = 0.5 * F.mse_loss(x_recon, x, reduction='sum')
        loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        loss_selfExp = 0.5 * F.mse_loss(z_recon, z, reduction='sum')
        loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp
        loss /= x.size(0)  # just control the range, does not affect the optimization.
        return loss


def train(model,  # type: DSCNet
          x, y, epochs, lr=1e-3, weight_coef=1.0, weight_selfExp=150.0, device='cuda',
          alpha=0.04, dim_subspace=12, ro=8.0, show=10):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    x = x.to(device)
    if isinstance(y, torch.Tensor):
        y = y.to('cpu').numpy()
    K = len(np.unique(y))
    model.to(device)
    acc_, nmi_ = 0., 0.
    for epoch in range(epochs):
        x_recon, z, z_recon = model(x)
        loss = model.loss_fn(x, x_recon, z, z_recon, weight_coef=weight_coef, weight_selfExp=weight_selfExp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % show == 0 or epoch == epochs - 1:
            C = model.self_expression.Coefficient.detach().to('cpu').numpy()
            y_pred = spectral_clustering(C, K, dim_subspace, alpha, ro)
            acc_, nmi_ = acc(y, y_pred), nmi(y, y_pred)
            print('Epoch %02d: loss=%.4f, acc=%.4f, nmi=%.4f' % (epoch, loss.item(), acc_, nmi_))
    return acc_, nmi_


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='DSCNet')
    parser.add_argument('--db', default='yaleb', choices=['yaleb'])
    parser.add_argument('--show-freq', default=100, type=int)
    parser.add_argument('--ae-weights', default=None)
    parser.add_argument('--save-dir', default='results')
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    db = args.db
    if db == 'yaleb':
        # load data
        data = sio.loadmat('datasets/YaleBCrop025.mat')
        img = data['Y']
        I = []
        Label = []
        for i in range(img.shape[2]):
            for j in range(img.shape[1]):
                temp = np.reshape(img[:, j, i], [42, 48])
                Label.append(i)
                I.append(temp)
        I = np.array(I)
        y_total = np.array(Label[:])
        Img = np.transpose(I, [0, 2, 1])
        x_total = np.expand_dims(Img[:], 1)
        print(x_total.shape)
        print(y_total.shape)
        print(np.unique(y_total))

        # network and optimization parameters
        channels = [1, 10, 20, 30]
        kernels = [5, 3, 3]

        # post clustering parameters
        # alpha = 0.04  # threshold of C
        dim_subspace = 10  # dimension of each subspace
        ro = 3.5
    else:
        exit(1)

    all_subjects = [38]  # [10, 15, 20, 25, 30, 35, 38]
    acc_avg, nmi_avg = [], []
    iter_loop = 0
    for iter_loop in range(len(all_subjects)):  # how many subjects to use
        num_class = all_subjects[iter_loop]
        num_sample = num_class * 64
        epochs = 50 + num_class * 25
        weight_coef = 1.0
        weight_selfExp = 1.0 * 10 ** (num_class / 10.0 - 3.0)
        alpha = max(0.4 - (num_class - 1) / 10 * 0.1, 0.1)
        print('='*20, 'Train on %d subjects' % num_class, '='*20)
        acc_subjects, nmi_subjects = [], []
        for i in range(0, 39 - num_class):  # which `num_class` subjects to use
            print('-'*20, 'The %dth / %d group of %d subjects' % (i+1, 39-num_class, num_class), '-'*20)
            x = x_total[64 * i:64 * (i + num_class)].astype(float)
            y = y_total[64 * i:64 * (i + num_class)]
            y = y - y.min()

            dscnet = DSCNet(num_sample=num_sample, channels=channels, kernels=kernels)
            # load the pretrained weights which are provided by the original author in
            # https://github.com/panji1990/Deep-subspace-clustering-networks
            dscnet.ae.load_state_dict(torch.load('pretrained_weights_original/%s.pkl' % db))
            print("Pretrained ae weights are loaded successfully.")

            acc_i, nmi_i = train(dscnet, x, y, epochs, weight_coef=weight_coef, weight_selfExp=weight_selfExp,
                                 alpha=alpha, dim_subspace=dim_subspace, ro=ro, show=args.show_freq, device=device)
            acc_subjects.append(acc_i)
            nmi_subjects.append(nmi_i)
        acc_avg.append(sum(acc_subjects)/len(acc_subjects))
        nmi_avg.append(sum(nmi_subjects)/len(nmi_subjects))
        print(acc_avg, nmi_avg)

    for iter_loop in range(len(all_subjects)):
        num_class = all_subjects[iter_loop]
        print('%d subjects:' % num_class)
        print('Acc: %.4f%%' % (acc_avg[iter_loop] * 100), 'Nmi: %.4f%%' % (nmi_avg[iter_loop] * 100))