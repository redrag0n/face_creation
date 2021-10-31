import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from math import ceil, floor
#from torchgan.losses.wasserstein import WassersteinGeneratorLoss, WassersteinDiscriminatorLoss
from conditional_vae import Decoder, Encoder


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self, data_shape, label_shape, layer_count=3, base_filters=32, kernel_size=(5, 5),
                 label_resize_shape=100, use_add_layer=False, loss_f='bce'):
        nn.Module.__init__(self)
        self.base = Encoder(1./2, data_shape, label_shape, layer_count, base_filters // 8, kernel_size,
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
        #return torch.tanh(self.base(x, y))
        return self.base(x, y)


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
        self.discriminator = Discriminator(self.data_shape, self.label_shape, self.layer_count, self.base_filters, self.kernel_size,
                                           self.label_resize_shape, self.use_add_layer, loss_f=self.loss_f)

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
