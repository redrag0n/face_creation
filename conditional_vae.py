import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from math import ceil, floor


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
            self.encoder_layers.append(nn.Conv2d(prev_filters, prev_filters, self.kernel_size,
                                                 padding='same'))
            self.encoder_layers.append(nn.Conv2d(prev_filters, next_filters, self.kernel_size,
                                                 stride=(2, 2)))
            self.encoder_layers.append(torch.nn.BatchNorm2d(next_filters))
            self.encoder_layers.append(torch.nn.ReLU())

        self.encoder_layers = nn.ModuleList(self.encoder_layers)
        self.w = self.data_shape[1] // (2 ** self.layer_count)
        self.h = self.data_shape[2] // (2 ** self.layer_count)
        self.c = self.base_filters * (2 ** (self.layer_count - 1))
        # self.c = self.base_filters
        # print(self.w, self.h, self.c)
        self.fc_e_label = nn.Linear(self.label_shape, self.label_resize_shape)
        self.bn_e_label = torch.nn.BatchNorm1d(self.label_resize_shape)
        self.fc_e = nn.Linear(self.w * self.c * self.h + self.label_resize_shape, int(self.latent_dim_size * 2))

    def forward(self, data, label):
        res = data
        # print(res.shape)
        for layer in self.encoder_layers:
            # print(layer)
            res = layer(res)
            # print(res.shape)

        res = torch.flatten(res, start_dim=1)

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

        self.fc_d_label = nn.Linear(self.label_shape, self.label_resize_shape)
        self.bn_d_label = torch.nn.BatchNorm1d(self.label_resize_shape)
        self.fc_d = nn.Linear(self.latent_dim_size + self.label_resize_shape, self.w * self.c * self.h)
        self.bn_d = torch.nn.BatchNorm1d(self.w * self.c * self.h)
        for layer_i in range(self.layer_count - 1, -1, -1):
            prev_filters = self.base_filters * 2 ** layer_i
            next_filters = self.base_filters * 2 ** (layer_i - 1) if layer_i > 0 else 3

            self.decoder_layers.append(nn.ConvTranspose2d(prev_filters, prev_filters, self.kernel_size,
                                                          padding=padding))
            self.decoder_layers.append(nn.ConvTranspose2d(prev_filters, next_filters, self.kernel_size,
                                                          stride=(2, 2), padding=padding,
                                                          output_padding=1))

            if layer_i != 0:
                self.decoder_layers.append(torch.nn.BatchNorm2d(next_filters))
                self.decoder_layers.append(torch.nn.ReLU())
        self.decoder_layers = nn.ModuleList(self.decoder_layers)

    def forward(self, latent, label):
        label = F.relu(self.bn_d_label(self.fc_d_label(label)))
        res = latent
        res = torch.cat([res, label], dim=1)
        res = F.relu(self.bn_d(self.fc_d(res)))
        res = torch.reshape(res, (-1, self.c, self.w, self.h))
        # print(res.shape)

        for layer in self.decoder_layers:
            res = layer(res)
            # print(res.shape)

        return res


class ConditionalVAE(nn.Module):
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
        self.encoder = Encoder(self.latent_dim_size, self.data_shape, self.label_shape, self.layer_count, self.base_filters, self.kernel_size,
                                     self.label_resize_shape, self.use_add_layer)
        self.decoder = Decoder(self.latent_dim_size, self.data_shape, self.label_shape, self.layer_count, self.base_filters, self.kernel_size,
                               self.label_resize_shape, self.use_add_layer)

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
        latent_t = t_mean + torch.exp(t_log_var / 2) * epsilon
        return latent_t

    def forward(self, x, label):
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
