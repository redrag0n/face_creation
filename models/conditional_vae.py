import os
import json
import numpy as np
import setuptools
import torch
from torch import nn
import torch.nn.functional as F
from math import ceil, floor
from models.encoder import Encoder
from models.decoder import Decoder


class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim_size=None, data_shape=None, label_shape=None,
                 layer_count=3, base_filters=32, kernel_size=(5, 5),
                 label_resize_shape=100, use_add_layer=False, last_non_linearity=None,
                 encoder=None, decoder=None):
        nn.Module.__init__(self)
        self.latent_dim_size = latent_dim_size
        self.data_shape = data_shape
        self.label_shape = label_shape
        self.layer_count = layer_count
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.label_resize_shape = label_resize_shape
        self.use_add_layer = use_add_layer
        self.last_non_linearity = last_non_linearity
        self.encoder = encoder
        self.decoder = decoder
        self.create_architecture()

    def create_architecture(self):
        if self.encoder is None:
            self.encoder = Encoder(self.latent_dim_size * 2, self.data_shape,
                                   self.label_shape, self.layer_count, self.base_filters, self.kernel_size,
                                   self.label_resize_shape, self.use_add_layer, self.last_non_linearity)
        if self.decoder is None:
            self.decoder = Decoder(self.latent_dim_size, self.data_shape, self.label_shape,
                                   self.layer_count, self.base_filters, self.kernel_size,
                                   self.label_resize_shape, self.use_add_layer, self.last_non_linearity)

    def get_cond_t_mean(self, hidden):
        return hidden[:, :self.latent_dim_size]

    def get_cond_t_log_var(self, hidden):
        return hidden[:, self.latent_dim_size:]

    def sampling(self, t_mean, t_log_var):
        """Returns sample from a distribution N(args[0], diag(args[1]))

        The sample should be computed with reparametrization trick.

        The inputs are tf.Tensor
            args[0]: (batch_size x latent_dim) mean of the desired distribution
            args[1]: (batch_size x latent_dim) logarithm of the variance vector of the desired distribution

        Returns:
            A tf.Tensor of size (batch_size x latent_dim), the samples.
        """
        #print(t_mean.is_cuda)
        device = 'cuda' if t_mean.is_cuda else 'cpu'
        epsilon = torch.tensor(np.random.standard_normal(t_mean.shape), requires_grad=False).float().to(device)
        #print(epsilon.is_cuda)
        #print(t_mean.shape, t_log_var.shape)
        latent_t = t_mean + torch.exp(t_log_var / 2) * epsilon
        return latent_t

    def forward(self, x, label=None):
        hidden = self.encoder(x, label)
        cond_t_mean = self.get_cond_t_mean(hidden)
        cond_t_log_var = self.get_cond_t_log_var(hidden)
        t_sampled = self.sampling(cond_t_mean, cond_t_log_var)
        decoded = self.decoder(t_sampled, label)
        return decoded, cond_t_mean, cond_t_log_var, t_sampled

    @staticmethod
    def loss(x, reconstruction, t_mean, t_log_var, reconstruction_weight=1000):
        loss_reconstruction = torch.mean(nn.functional.mse_loss(x, reconstruction))
        # print('loss reconstruction', loss_reconstruction.cpu().item())
        # print('sum x', torch.sum(x).cpu().item())
        # print('sum reconstruction', torch.sum(reconstruction).cpu().item())
        loss_KL = - torch.mean(
            0.5 * torch.sum(1 + t_log_var - torch.square(t_mean) - torch.exp(t_log_var), dim=1)
        )
        # #rint('t_log_var', t_log_var, torch.sum(t_log_var).cpu().item())
        # print('t_log_var', torch.sum(t_log_var).cpu().item())
        # #print('exp t_log_var', torch.exp(t_log_var), torch.sum(torch.exp(t_log_var)).cpu().item())
        # print('exp t_log_var', torch.sum(torch.exp(t_log_var)).cpu().item())
        # #print('t_mean', t_mean)
        # print('square t_mean', torch.sum(torch.square(t_mean)).cpu().item())
        # print('KL loss', loss_KL.cpu().item())
        loss = reconstruction_weight * loss_reconstruction + loss_KL
        #print('loss', loss.cpu().item())
        return loss

    def log_gradients(self):
        print('encoder', self.encoder.encoder_layers[1].weight.grad[0])
        print('decoder', self.decoder.decoder_layers[1].weight.grad[0])

    def save(self, model_save_dir, encoder_name='encoder', decoder_name='generator'):
        if not os.path.isdir(model_save_dir):
            os.mkdir(model_save_dir)
        torch.save(self.decoder, f'{model_save_dir}/{decoder_name}')
        torch.save(self.encoder, f'{model_save_dir}/{encoder_name}')
        params = {
            'latent_dim_size': self.latent_dim_size,
            'data_shape': self.data_shape,
            'label_shape': self.label_shape,
            'layer_count': self.layer_count,
            'base_filters': self.base_filters,
            'kernel_size': self.kernel_size,
            'label_resize_shape': self.label_resize_shape,
            'use_add_layer': self.use_add_layer,
            'last_non_linearity': self.last_non_linearity,
        }
        with open(f'{model_save_dir}/params.json', mode='w') as out:
            json.dump(params, out, ensure_ascii=False)

    @staticmethod
    def load(model_save_dir, device='cpu',
             encoder_name='encoder', decoder_name='generator'):
        with open(f'{model_save_dir}/params.json', mode='r') as in_:
            params = json.load(in_)
        params['encoder'] = torch.load(f'{model_save_dir}/{encoder_name}', map_location=torch.device(device))
        params['encoder'].eval()
        params['decoder'] = torch.load(f'{model_save_dir}/{decoder_name}', map_location=torch.device(device))
        params['decoder'].eval()
        return ConditionalVAE(**params)
