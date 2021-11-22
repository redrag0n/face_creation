import torch
from torch import nn
import torch.nn.functional as F
from math import ceil, floor

from models.encoder import non_linearity_dict


class Decoder(nn.Module):
    def __init__(self, latent_dim_size, data_shape, label_shape, layer_count=3, base_filters=32, kernel_size=(5, 5),
                 label_resize_shape=100, use_add_layer=False, last_non_linearity=None):
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
        self.create_architecture()

    def create_architecture(self):
        self.decoder_layers = list()
        padding = ceil((self.kernel_size[0] - 1) / 2)

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
        for layer in self.decoder_layers:
            res = layer(res)
            # print(res.shape)
        if self.last_non_linearity is not None:
            res = non_linearity_dict[self.last_non_linearity](res)
        return res

