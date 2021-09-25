import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from math import ceil, floor


class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim_size, data_shape, label_shape):
        nn.Module.__init__(self)
        self.latent_dim_size = latent_dim_size
        self.data_shape = data_shape
        self.label_shape = label_shape
        self.create_architecture()

    def create_encoder(self):
        self.conv1 = nn.Conv2d(3, 32, (3, 3), padding='same')
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bn4 = torch.nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bn5 = torch.nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 1, (3, 3), padding='same')
        self.bn6 = torch.nn.BatchNorm2d(1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # self.conv1 = nn.Conv2d(3, 32, (3, 3), padding='same')
        # self.conv2 = nn.Conv2d(32, 32, (3, 3), padding='same')
        # self.pool1 = nn.MaxPool2d(2, 2)
        #
        # self.conv3 = nn.Conv2d(32, 32, (3, 3), padding='same')
        # self.conv4 = nn.Conv2d(32, 32, (3, 3), padding='same')
        # self.pool2 = nn.MaxPool2d(2, 2)
        #
        # self.conv5 = nn.Conv2d(32, 32, (3, 3), padding='same')
        # self.conv6 = nn.Conv2d(32, 1, (3, 3), padding='same')
        # self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(224 * 184 // 64, 500)
        self.fc_label = nn.Linear(self.label_shape, 100)
        self.fc2 = nn.Linear(500 + 100, 100)
        self.fc3 = nn.Linear(100, self.latent_dim_size * 2)

    def create_decoder(self):
        self.fc1_d = nn.Linear(self.latent_dim_size + self.label_shape, 100)
        self.fc2_d = nn.Linear(100, 200)
        self.fc3_d = nn.Linear(200, 224 * 184 // 64)

        # self.unpool1 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        # self.conv1_d = nn.Conv2d(1, 32, (3, 3), padding='same')
        # self.conv2_d = nn.Conv2d(32, 32, (3, 3), padding='same')
        #
        # self.unpool2 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        # self.conv3_d = nn.Conv2d(32, 32, (3, 3), padding='same')
        # self.conv4_d = nn.Conv2d(32, 32, (3, 3), padding='same')
        #
        # self.unpool3 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        # self.conv5_d = nn.Conv2d(32, 32, (3, 3), padding='same')
        # self.conv6_d = nn.Conv2d(32, 3, (3, 3), padding='same')
        self.unpool1 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.conv1_d = nn.Conv2d(1, 32, (3, 3), padding='same')
        self.bn1_d = torch.nn.BatchNorm2d(32)
        self.conv2_d = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bn2_d = torch.nn.BatchNorm2d(32)

        self.unpool2 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.conv3_d = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bn3_d = torch.nn.BatchNorm2d(32)
        self.conv4_d = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bn4_d = torch.nn.BatchNorm2d(32)

        self.unpool3 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.conv5_d = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bn5_d = torch.nn.BatchNorm2d(32)
        self.conv6_d = nn.Conv2d(32, 3, (3, 3), padding='same')
        self.bn6_d = torch.nn.BatchNorm2d(3)

    def create_architecture(self):
        self.create_encoder()
        self.create_decoder()

    def encode(self, data, label):
        # print(data.shape)
        res = self.pool1(F.relu(self.bn2(self.conv2(self.bn1(self.conv1(data))))))
        # print(res.shape)
        res = self.pool2(F.relu(self.bn4(self.conv4(self.bn3(self.conv3(res))))))
        # print(res.shape)
        res = self.pool3(F.relu(self.bn6(self.conv6(self.bn5(self.conv5(res))))))
        # print(res.shape)

        res = torch.flatten(res, start_dim=1)
        # print('!!!')
        # print(res.shape)
        res = self.fc1(res)

        label_enc = self.fc_label(label)

        res = torch.cat([res, label_enc], dim=1)
        res = self.fc2(F.relu(res))
        res = self.fc3(F.relu(res))
        # print('encode end', res.shape)
        return res

    def decode(self, latent, label):
        res = torch.cat([latent, label], dim=1)
        res = F.relu(self.fc1_d(res))
        res = F.relu(self.fc2_d(res))
        res = F.relu(self.fc3_d(res))
        # print(res.shape)

        res = torch.reshape(res, (res.shape[0], 1, 224 // 8, 184 // 8))
        res = self.unpool1(F.relu(self.bn2_d(self.conv2_d(self.bn1_d(self.conv1_d(res))))))
        res = self.unpool2(F.relu(self.bn4_d(self.conv4_d(self.bn3_d(self.conv3_d(res))))))
        res = self.unpool3(self.bn6_d(self.conv6_d(self.bn5_d(self.conv5_d(res)))))
        # print('decode', res.shape)
        return res

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
        hidden = self.encode(x, label)
        cond_t_mean = self.get_cond_t_mean(hidden)
        cond_t_log_var = self.get_cond_t_log_var(hidden)
        t_sampled = self.sampling(cond_t_mean, cond_t_log_var)
        decoded = self.decode(t_sampled, label)
        return decoded, cond_t_mean, cond_t_log_var

    @staticmethod
    def loss(x, reconstruction, t_mean, t_log_var, reconstruction_weight=1000):
        loss_reconstruction = torch.mean(nn.functional.mse_loss(x, reconstruction))
        loss_KL = - torch.mean(
            0.5 * torch.sum(1 + t_log_var - torch.square(t_mean) - torch.exp(t_log_var), dim=1)
        )
        loss = reconstruction_weight * loss_reconstruction + loss_KL
        return loss


class ConditionalVAE1(nn.Module):
    def __init__(self, latent_dim_size, data_shape, label_shape):
        nn.Module.__init__(self)
        self.latent_dim_size = latent_dim_size
        self.data_shape = data_shape
        self.label_shape = label_shape
        self.create_architecture()

    def create_encoder(self):
        self.conv1 = nn.Conv2d(3, 32, (3, 3), padding='same')
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bn4 = torch.nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bn5 = torch.nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 1, (3, 3), padding='same')
        self.bn6 = torch.nn.BatchNorm2d(1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(224 * 184 * 4 // 64, 500)
        self.bn7 = torch.nn.BatchNorm1d(500)
        self.fc_label = nn.Linear(self.label_shape, 100)
        self.fc2 = nn.Linear(500 + 100, 100)
        self.bn8 = torch.nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, self.latent_dim_size * 2)

    def create_decoder(self):
        self.fc1_d = nn.Linear(self.latent_dim_size + self.label_shape, 100)
        self.bn01_d = torch.nn.BatchNorm1d(100)
        self.fc2_d = nn.Linear(100, 200)
        self.bn02_d = torch.nn.BatchNorm1d(200)
        self.fc3_d = nn.Linear(200, 224 * 184 * 4 // 64)
        self.bn03_d = torch.nn.BatchNorm1d(224 * 184 * 4 // 64)

        self.unpool1 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.conv1_d = nn.Conv2d(1, 32, (3, 3), padding='same')
        self.bn1_d = torch.nn.BatchNorm2d(32)
        self.conv2_d = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bn2_d = torch.nn.BatchNorm2d(32)

        self.unpool2 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.conv3_d = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bn3_d = torch.nn.BatchNorm2d(32)
        self.conv4_d = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bn4_d = torch.nn.BatchNorm2d(32)

        self.unpool3 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.conv5_d = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bn5_d = torch.nn.BatchNorm2d(32)
        self.conv6_d = nn.Conv2d(32, 3, (3, 3), padding='same')
        self.bn6_d = torch.nn.BatchNorm2d(3)

    def create_architecture(self):
        self.create_encoder()
        self.create_decoder()

    def encode(self, data, label):
        # print(data.shape)
        res = self.pool1(F.relu(self.bn2(self.conv2(self.bn1(self.conv1(data))))))
        # print(res.shape)
#        res = self.pool2(F.relu(self.conv4(self.conv3(res))))
        # print(res.shape)
        res = self.pool3(F.relu(self.bn6(self.conv6(self.bn5(self.conv5(res))))))
        # print(res.shape)

        res = torch.flatten(res, start_dim=1)
        # print('!!!')
        # print(res.shape)
        res = self.bn7(self.fc1(res))

        label_enc = self.fc_label(label)

        res = torch.cat([res, label_enc], dim=1)
        res = self.bn8(self.fc2(F.relu(res)))
        res = self.fc3(F.relu(res))
        # print('encode end', res.shape)
        return res

    def decode(self, latent, label):
        res = torch.cat([latent, label], dim=1)
        res = self.bn01_d(F.relu(self.fc1_d(res)))
        res = self.bn02_d(F.relu(self.fc2_d(res)))
        res = self.bn03_d(F.relu(self.fc3_d(res)))
        # print(res.shape)

        res = torch.reshape(res, (res.shape[0], 1, 224 * 2 // 8, 184 * 2 // 8))
        res = self.unpool1(F.relu(self.bn2_d(self.conv2_d(self.bn1_d(self.conv1_d(res))))))
        # res = self.unpool2(F.relu(self.conv4_d(self.conv3_d(res))))
        res = self.unpool3(self.bn6_d(self.conv6_d(self.bn5_d(self.conv5_d(res)))))
        # print('decode', res.shape)
        return res

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
        #print(latent_t)
        #latent_t = torch.exp(t_log_var / 2)# * epsilon
        return latent_t

    def forward(self, x, label):
        hidden = self.encode(x, label)
        cond_t_mean = self.get_cond_t_mean(hidden)
        cond_t_log_var = self.get_cond_t_log_var(hidden)
        t_sampled = self.sampling(cond_t_mean, cond_t_log_var)
        decoded = self.decode(t_sampled, label)
        return decoded, cond_t_mean, cond_t_log_var

    @staticmethod
    def loss(x, reconstruction, t_mean, t_log_var, reconstruction_weight=1000):
        loss_reconstruction = torch.mean(nn.functional.mse_loss(x, reconstruction))
        loss_KL = - torch.mean(
            0.5 * torch.sum(1 + t_log_var - torch.square(t_mean) - torch.exp(t_log_var), dim=1)
        )
        loss = reconstruction_weight * loss_reconstruction + loss_KL
        return loss


class ConditionalVAE3(ConditionalVAE):
    def __init__(self, latent_dim_size, data_shape, label_shape, layer_count=3, base_filters=64, kernel_size=(5, 5)):
        self.layer_count = layer_count
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.encoder_layers = list()
        self.decoder_layers = list()
        ConditionalVAE.__init__(self, latent_dim_size, data_shape, label_shape)

    def create_encoder(self):
        padding = (ceil((self.kernel_size[0] - 1) / 2), floor((self.kernel_size[0] - 1) / 2),
                   ceil((self.kernel_size[1] - 1) / 2), floor((self.kernel_size[1] - 1) / 2))
        for layer_i in range(self.layer_count):
            prev_filters = self.base_filters * 2 ** (layer_i - 1) if layer_i > 0 else 3
            next_filters = self.base_filters * 2 ** layer_i
            self.encoder_layers.append(nn.ConstantPad2d(padding, 0))
            self.encoder_layers.append(nn.Conv2d(prev_filters, next_filters, self.kernel_size,
                                                 stride=(2, 2)))
            self.encoder_layers.append(torch.nn.BatchNorm2d(next_filters))
            self.encoder_layers.append(torch.nn.ReLU())

        self.w = self.data_shape[1] // (2 ** self.layer_count)
        self.h = self.data_shape[2] // (2 ** self.layer_count)
        self.c = self.base_filters * (2 ** (self.layer_count - 1))
        # print(self.w, self.h, self.c)
        self.fc_e_label = nn.Linear(self.label_shape, 100)
        self.fc_e = nn.Linear(self.w * self.c * self.h + 100, self.latent_dim_size * 2)

    def create_decoder(self):
        padding = (ceil((self.kernel_size[0] - 1) / 2), floor((self.kernel_size[0] - 1) / 2),
                   ceil((self.kernel_size[1] - 1) / 2), floor((self.kernel_size[1] - 1) / 2))
        # print(padding)
        #print(padding)
        #padding = (0, 0, 0, 0)
        # w = self.data_shape[1] // (2 ** self.layer_count)
        # h = self.data_shape[2] // (2 ** self.layer_count)
        # c = self.base_filters * (2 ** (self.layer_count - 1))

        self.fc_d_label = nn.Linear(self.label_shape, 100)
        self.fc_d = nn.Linear(self.latent_dim_size + 100, self.w * self.c * self.h)
        for layer_i in range(self.layer_count - 1, -1, -1):
            prev_filters = self.base_filters * 2 ** layer_i
            next_filters = self.base_filters * 2 ** (layer_i - 1) if layer_i > 0 else 3
            #self.decoder_layers.append(nn.ConstantPad2d(padding, 0))
            # self.decoder_layers.append(nn.ConvTranspose2d(prev_filters, next_filters, self.kernel_size,
            #                                               stride=(2, 2), padding=0))
            #self.decoder_layers.append(nn.Conv2d(prev_filters, next_filters, self.kernel_size, stride=(2, 2), padding=ceil((self.kernel_size[0] - 1) / 2)))
            self.decoder_layers.append(nn.ConstantPad2d(padding, 0))
            self.decoder_layers.append(nn.Conv2d(prev_filters, next_filters, self.kernel_size, padding=0))
            self.decoder_layers.append(nn.Upsample(scale_factor=2, mode='nearest', align_corners=None))
            #self.decoder_layers.append(nn.ConstantPad2d(padding, 0))
            if layer_i != 0:
                self.decoder_layers.append(torch.nn.BatchNorm2d(next_filters))
                self.decoder_layers.append(torch.nn.ReLU())

    def encode(self, data, label):
        res = data
        # print(res.shape)
        for layer in self.encoder_layers:
            # print(layer)
            res = layer(res)
            # print(res.shape)

        res = torch.flatten(res, start_dim=1)

        label_enc = self.fc_e_label(label)

        res = torch.cat([res, label_enc], dim=1)
        res = self.fc_e(F.relu(res))
        return res

    def decode(self, latent, label):
        label = self.fc_d_label(label)
        res = latent
        res = torch.cat([res, label], dim=1)
        res = self.fc_d(res)
        res = torch.reshape(res, (-1, self.c, self.w, self.h))
        print(res.shape)

        for layer in self.decoder_layers:
            res = layer(res)
            print(res.shape)

        return res

