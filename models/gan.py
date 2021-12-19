import os
import json
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from math import ceil, floor
#from torchgan.losses.wasserstein import WassersteinGeneratorLoss, WassersteinDiscriminatorLoss
from models.decoder import Decoder
from models.encoder import Encoder


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# class Discriminator(nn.Module):
#     def __init__(self, data_shape, label_shape, layer_count=3, base_filters=32, kernel_size=(5, 5),
#                  label_resize_shape=100, use_add_layer=False, loss_f='bce'):
#         nn.Module.__init__(self)
#         # self.base = Encoder(1./2, data_shape, label_shape, layer_count, base_filters // 8, kernel_size,
#         #                     label_resize_shape // 2, use_add_layer)
#         self.base = Encoder(1. / 2, data_shape, label_shape, layer_count + 1, base_filters * 2, kernel_size,
#                             label_resize_shape // 2, use_add_layer)
#         self.loss_f = loss_f
#         self.apply(weights_init)
#
#     def forward(self, x, y):
#         if self.loss_f == 'bce':
#             return torch.sigmoid(self.base(x, y)).view(-1)
#         else:
#             return self.base(x, y).view(-1)
#
#
# class Generator(nn.Module):
#     def __init__(self, latent_dim_size, data_shape, label_shape, layer_count=3, base_filters=32, kernel_size=(5, 5),
#                  label_resize_shape=100, use_add_layer=False):
#         nn.Module.__init__(self)
#         self.base = Decoder(latent_dim_size, data_shape, label_shape, layer_count, base_filters, kernel_size,
#                             label_resize_shape, use_add_layer)
#         self.apply(weights_init)
#
#     def forward(self, x, y):
#         #return torch.sigmoid(self.base(x, y)).view(-1)
#         #return torch.tanh(self.base(x, y))
#         return self.base(x, y)


class GAN(nn.Module):
    def __init__(self, generator_params=None, discriminator_params=None, loss_f='bce', generator=None,
                 discriminator=None):
        nn.Module.__init__(self)
        self.generator_params = generator_params
        self.discriminator_params = discriminator_params
        self.loss_f = loss_f
        self.generator = generator
        self.discriminator = discriminator
        self.create_architecture()

    def create_architecture(self):
        if self.generator is None:
            self.generator = Decoder(**self.generator_params)
            self.generator.apply(weights_init)
        print(self.generator)
        print('Generator parameters:', sum(p.numel() for p in self.generator.parameters() if p.requires_grad))

        if self.discriminator is None:
            self.discriminator = Encoder(1, **self.discriminator_params)
            self.discriminator.apply(weights_init)
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
            return real_loss + fake_loss
        elif self.loss_f == 'wasserstein':
            loss = torch.mean(fake_predicted) - torch.mean(true_predicted)
            if backward:
                loss.backward()
            return loss
        else:
            return None

    def save(self, model_save_dir, generator_name='generator', discriminator_name='discriminator'):
        if not os.path.isdir(model_save_dir):
            os.mkdir(model_save_dir)
        torch.save(self.generator, f'{model_save_dir}/{generator_name}')
        torch.save(self.discriminator, f'{model_save_dir}/{discriminator_name}')
        params = {
            'generator_params': self.generator_params,
            'discriminator_params': self.discriminator_params,
            'loss_f': self.loss_f
        }
        with open(f'{model_save_dir}/params.json', mode='w') as out:
            json.dump(params, out, ensure_ascii=False)

    @staticmethod
    def load(model_save_dir, device='cpu',
             generator_name='generator', discriminator_name='discriminator'):
        with open(f'{model_save_dir}/params.json', mode='r') as in_:
            params = json.load(in_)
        params['generator'] = torch.load(f'{model_save_dir}/{generator_name}', map_location=torch.device(device))
        params['generator'].eval()
        params['discriminator'] = torch.load(f'{model_save_dir}/{discriminator_name}', map_location=torch.device(device))
        params['discriminator'].eval()
        return GAN(**params)

    def load_generator(self, generator_path, device='cpu'):
        self.generator = torch.load(generator_path, map_location=torch.device(device))

