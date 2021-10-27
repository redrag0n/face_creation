import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from math import ceil, floor
#from torchgan.losses.wasserstein import WassersteinGeneratorLoss, WassersteinDiscriminatorLoss
from conditional_vae import Decoder, Encoder


class Discriminator(nn.Module):
    def __init__(self, data_shape, label_shape, layer_count=3, base_filters=32, kernel_size=(5, 5),
                 label_resize_shape=100, use_add_layer=False):
        nn.Module.__init__(self)
        self.base = Encoder(1./2, data_shape, label_shape, layer_count, base_filters // 2, kernel_size,
                            label_resize_shape // 2, use_add_layer)

    def forward(self, x, y):
        #return torch.sigmoid(self.base(x, y)).view(-1)
        return self.base(x, y).view(-1)


class Generator(nn.Module):
    def __init__(self, latent_dim_size, data_shape, label_shape, layer_count=3, base_filters=32, kernel_size=(5, 5),
                 label_resize_shape=100, use_add_layer=False):
        nn.Module.__init__(self)
        self.base = Decoder(latent_dim_size, data_shape, label_shape, layer_count, base_filters, kernel_size,
                            label_resize_shape, use_add_layer)

    def forward(self, x, y):
        #return torch.sigmoid(self.base(x, y)).view(-1)
        return torch.tanh(self.base(x, y))


class GAN(nn.Module):
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
        self.generator = Generator(self.latent_dim_size, self.data_shape, self.label_shape, self.layer_count, self.base_filters, self.kernel_size,
                                 self.label_resize_shape, self.use_add_layer)
        self.discriminator = Discriminator(self.data_shape, self.label_shape, self.layer_count, self.base_filters, self.kernel_size,
                                           self.label_resize_shape, self.use_add_layer)

    @staticmethod
    def loss(*args):
        return nn.BCELoss()(*args)

    @staticmethod
    def generator_loss(fake_predicted):
        return -torch.mean(fake_predicted)

    @staticmethod
    def discriminator_loss(true_predicted, fake_predicted):
        return torch.mean(fake_predicted) - torch.mean(true_predicted)
