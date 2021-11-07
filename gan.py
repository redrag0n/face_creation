import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from math import ceil, floor
#from torchgan.losses.wasserstein import WassersteinGeneratorLoss, WassersteinDiscriminatorLoss
#from conditional_vae import Decoder, Encoder


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Encoder(nn.Module):
    def __init__(self, latent_dim_size, data_shape, label_shape, layer_count=3, base_filters=32, kernel_size=(5, 5),
                 label_resize_shape=100, use_add_layer=False):
        nn.Module.__init__(self)
        self.latent_dim_size = latent_dim_size
        self.data_shape = data_shape
        self.label_shape = label_shape
        self.layer_count = layer_count
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.label_resize_shape = label_resize_shape
        self.use_add_layer = use_add_layer
        self.create_architecture()

    def create_architecture(self):
        self.encoder_layers = list()
        padding = (ceil((self.kernel_size[0] - 1) / 2), floor((self.kernel_size[0] - 1) / 2),
                   ceil((self.kernel_size[1] - 1) / 2), floor((self.kernel_size[1] - 1) / 2))
        for layer_i in range(self.layer_count):
            prev_filters = self.base_filters * 2 ** (layer_i - 1) if layer_i > 0 else 3
            next_filters = self.base_filters * 2 ** layer_i

            self.encoder_layers.append(nn.ConstantPad2d(padding, 0))
            # self.encoder_layers.append(nn.Conv2d(prev_filters, prev_filters, self.kernel_size,
            #                                      padding='same'))
            self.encoder_layers.append(nn.Conv2d(prev_filters, next_filters, self.kernel_size,
                                                 stride=(2, 2), bias=False))
            self.encoder_layers.append(torch.nn.BatchNorm2d(next_filters))
            self.encoder_layers.append(torch.nn.LeakyReLU(negative_slope=0.2))

        self.encoder_layers = nn.ModuleList(self.encoder_layers)
        self.w = self.data_shape[1] // (2 ** self.layer_count)
        self.h = self.data_shape[2] // (2 ** self.layer_count)
        self.c = self.base_filters * (2 ** (self.layer_count - 1))
        # self.c = self.base_filters
        # print(self.w, self.h, self.c)
        if self.label_shape is not None:
            self.fc_e_label = nn.Linear(self.label_shape, self.label_resize_shape)
            self.bn_e_label = torch.nn.BatchNorm1d(self.label_resize_shape)
            self.fc_e = nn.Linear(self.w * self.c * self.h + self.label_resize_shape, int(self.latent_dim_size * 2))
        else:
            self.fc_e = nn.Linear(self.w * self.c * self.h, int(self.latent_dim_size * 2))

    def forward(self, data, label=None):
        res = data
        # print(res.shape)
        for layer in self.encoder_layers:
            # print(layer)
            res = layer(res)
            # print(res.shape)

        res = torch.flatten(res, start_dim=1)

        if self.label_shape is not None:
            label_enc = F.relu(self.bn_e_label(self.fc_e_label(label)))
            res = torch.cat([res, label_enc], dim=1)
        res = self.fc_e(res)
        return res


class Decoder(nn.Module):
    def __init__(self, latent_dim_size, data_shape, label_shape, layer_count=3, base_filters=32, kernel_size=(5, 5),
                 label_resize_shape=100, use_add_layer=False):
        nn.Module.__init__(self)
        self.latent_dim_size = latent_dim_size
        self.data_shape = data_shape
        self.label_shape = label_shape
        self.layer_count = layer_count
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.label_resize_shape = label_resize_shape
        self.use_add_layer = use_add_layer
        self.create_architecture()

    def create_architecture(self):
        self.decoder_layers = list()
        padding = ceil((self.kernel_size[0] - 1) / 2)
        # print(padding)

        self.w = self.data_shape[1] // (2 ** self.layer_count)
        self.h = self.data_shape[2] // (2 ** self.layer_count)
        self.c = self.base_filters * (2 ** (self.layer_count - 1))

        if self.label_shape is not None:
            self.fc_d_label = nn.Linear(self.label_shape, self.label_resize_shape)
            self.bn_d_label = torch.nn.BatchNorm1d(self.label_resize_shape)
            self.fc_d = nn.Linear(self.latent_dim_size + self.label_resize_shape, self.w * self.c * self.h)
        else:
            self.fc_d = nn.Linear(self.latent_dim_size, self.w * self.c * self.h)
        self.bn_d = torch.nn.BatchNorm1d(self.w * self.c * self.h)
        for layer_i in range(self.layer_count - 1, -1, -1):
            #prev_filters = self.base_filters * 2 ** layer_i if layer_i < self.layer_count - 1 else self.latent_dim_size
            prev_filters = self.base_filters * 2 ** layer_i
            next_filters = self.base_filters * 2 ** (layer_i - 1) if layer_i > 0 else 3

            # self.decoder_layers.append(nn.ConvTranspose2d(prev_filters, prev_filters, self.kernel_size,
            #                                               padding=padding))
            self.decoder_layers.append(nn.ConvTranspose2d(prev_filters, next_filters, self.kernel_size,
                                                          stride=(2, 2), padding=padding,
                                                          output_padding=1, bias=False))

            if layer_i != 0:
                self.decoder_layers.append(torch.nn.BatchNorm2d(next_filters))
                self.decoder_layers.append(torch.nn.ReLU())
        self.decoder_layers = nn.ModuleList(self.decoder_layers)

    def forward(self, latent, label=None):
        res = latent
        if self.label_shape is not None:
            label = F.relu(self.bn_d_label(self.fc_d_label(label)))
            res = torch.cat([res, label], dim=1)
        res = F.relu(self.bn_d(self.fc_d(res)))
        res = torch.reshape(res, (-1, self.c, self.w, self.h))
        # print(res.shape)
        # res = torch.unsqueeze(res, -1)
        # res = torch.unsqueeze(res, -1)
        print(res.shape)
        for layer in self.decoder_layers:
            res = layer(res)
            # print(res.shape)

        return res


class Discriminator(nn.Module):
    def __init__(self, data_shape, label_shape, layer_count=3, base_filters=32, kernel_size=(5, 5),
                 label_resize_shape=100, use_add_layer=False, loss_f='bce'):
        nn.Module.__init__(self)
        # self.base = Encoder(1./2, data_shape, label_shape, layer_count, base_filters // 8, kernel_size,
        #                     label_resize_shape // 2, use_add_layer)
        self.base = Encoder(1. / 2, data_shape, label_shape, layer_count, base_filters, kernel_size,
                            label_resize_shape // 2, use_add_layer)
        self.loss_f = loss_f
        self.apply(weights_init)

    def forward(self, x, y):
        if self.loss_f == 'bce':
            return torch.sigmoid(self.base(x, y)).view(-1)
        else:
            return self.base(x, y).view(-1)


class Generator(nn.Module):
    def __init__(self, latent_dim_size, data_shape, label_shape, layer_count=3, base_filters=32, kernel_size=(5, 5),
                 label_resize_shape=100, use_add_layer=False):
        nn.Module.__init__(self)
        self.base = Decoder(latent_dim_size, data_shape, label_shape, layer_count, base_filters, kernel_size,
                            label_resize_shape, use_add_layer)
        self.apply(weights_init)

    def forward(self, x, y):
        #return torch.sigmoid(self.base(x, y)).view(-1)
        return torch.tanh(self.base(x, y))
        #return self.base(x, y)


class GAN(nn.Module):
    def __init__(self, latent_dim_size, data_shape, label_shape, layer_count=3, base_filters=32, kernel_size=(5, 5),
                 label_resize_shape=100, use_add_layer=False, loss_f='bce'):
        nn.Module.__init__(self)
        self.latent_dim_size = latent_dim_size
        self.data_shape = data_shape
        self.label_shape = label_shape
        self.layer_count = layer_count
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.label_resize_shape = label_resize_shape
        self.use_add_layer = use_add_layer
        self.loss_f = loss_f
        self.create_architecture()

    def create_architecture(self):
        self.generator = Generator(self.latent_dim_size, self.data_shape, self.label_shape, self.layer_count, self.base_filters, self.kernel_size,
                                 self.label_resize_shape, self.use_add_layer)
        print(self.generator)
        print('Generator parameters:', sum(p.numel() for p in self.generator.parameters() if p.requires_grad))
        self.discriminator = Discriminator(self.data_shape, self.label_shape, self.layer_count, self.base_filters, self.kernel_size,
                                           self.label_resize_shape, self.use_add_layer, loss_f=self.loss_f)
        print(self.discriminator)
        print('Discriminator parameters:', sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad))

    @staticmethod
    def loss(*args):
        return nn.BCELoss()(*args)

    def generator_loss(self, fake_predicted, backward=True):
        if self.loss_f == 'bce':
            device = 'cuda' if fake_predicted.is_cuda else 'cpu'
            true_ = torch.full((fake_predicted.shape[0],), 1, dtype=torch.float, device=device)
            loss = nn.BCELoss()(fake_predicted, true_)
            if backward:
                loss.backward()
            return loss
        elif self.loss_f == 'wasserstein':
            loss = -torch.mean(fake_predicted)
            if backward:
                loss.backward()
            return loss
        else:
            return None

    def discriminator_loss(self, true_predicted, fake_predicted, backward=True):
        if self.loss_f == 'bce':
            device = 'cuda' if true_predicted.is_cuda else 'cpu'
            real_true = torch.full((fake_predicted.shape[0],), 1, dtype=torch.float, device=device)
            fake_true = torch.full((fake_predicted.shape[0],), 0, dtype=torch.float, device=device)
            real_loss = nn.BCELoss()(true_predicted, real_true)
            if backward:
                real_loss.backward()
            fake_loss = nn.BCELoss()(fake_predicted, fake_true)
            if backward:
                fake_loss.backward()
            return  real_loss + fake_loss
        elif self.loss_f == 'wasserstein':
            loss = torch.mean(fake_predicted) - torch.mean(true_predicted)
            if backward:
                loss.backward()
            return loss
        else:
            return None
