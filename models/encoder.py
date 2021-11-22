import torch
from torch import nn
import torch.nn.functional as F
from math import ceil, floor


non_linearity_dict = {
    'sigmoid': torch.sigmoid,
    'relu': torch.relu,
    'tanh': torch.tanh
}

class Encoder(nn.Module):
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
            self.fc_e = nn.Linear(self.w * self.c * self.h, self.latent_dim_size)

    def forward(self, data, label=None):
        res = data
        # print(res.shape)
        for layer in self.encoder_layers:
            #vprint(layer)
            res = layer(res)
            #vprint(res.shape)

        res = torch.flatten(res, start_dim=1)

        if self.label_shape is not None:
            label_enc = F.relu(self.bn_e_label(self.fc_e_label(label)))
            res = torch.cat([res, label_enc], dim=1)
        res = self.fc_e(res)

        if self.last_non_linearity is not None:
            res = non_linearity_dict[self.last_non_linearity](res)
        return res
