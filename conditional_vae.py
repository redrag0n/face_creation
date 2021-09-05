import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim_size, data_shape, label_shape):
        self.latent_dim_size = latent_dim_size
        self.data_shape = data_shape
        self.label_shape = label_shape

    def create_encoder(self):
        self.conv1 = nn.Conv2d(3, 32, (3, 3), padding='same')
        self.conv2 = nn.Conv2d(3, 32, (3, 3), padding='same')
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(3, 32, (3, 3), padding='same')
        self.conv4 = nn.Conv2d(3, 32, (3, 3), padding='same')
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(3, 32, (3, 3), padding='same')
        self.conv6 = nn.Conv2d(3, 32, (3, 3), padding='same')
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(218 * 178 // 64, 500)
        self.fc_label = nn.Linear(self.label_shape, 100)
        self.fc2 = nn.Linear(500 + self.label_shape, 100)
        self.fc3 = nn.Linear(100, self.latent_dim_size * 2)

    def create_decoder(self):
        self.fc1_d = nn.Linear(self.latent_dim_size + self.label_shape, 100)
        self.fc2_d = nn.Linear(100, 500)
        self.fc3_d = nn.Linear(500, 218 * 178 // 64)

        self.unpool1 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.conv1_d = nn.Conv2d(3, 32, (3, 3), padding='same')
        self.conv2_d = nn.Conv2d(3, 32, (3, 3), padding='same')

        self.unpool2 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.conv3_d = nn.Conv2d(3, 32, (3, 3), padding='same')
        self.conv4_d = nn.Conv2d(3, 32, (3, 3), padding='same')

        self.unpool3 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.conv5_d = nn.Conv2d(3, 32, (3, 3), padding='same')
        self.conv6_d = nn.Conv2d(3, 32, (3, 3), padding='same')

    def create_architecture(self):
        self.create_encoder()
        self.create_decoder()

    def encode(self, data, label):
        res = self.pool1(F.relu(self.conv2(self.conv1(data))))
        res = self.pool2(F.relu(self.conv4(self.conv3(res))))
        res = self.pool3(F.relu(self.conv4(self.conv3(res))))
        res = self.fc1(torch.flatten(res, 1))
        label_enc = self.fc_label(label)

        res = torch.cat([res, label_enc], dim=1)
        res = self.fc2(F.relu(res))
        res = self.fc3(F.relu(res))
        return res

    def decode(self, latent, label):
        res = torch.cat([latent, label], dim=1)
        res = F.relu(self.fc1_d(res))
        res = F.relu(self.fc2_d(res))
        res = F.relu(self.fc3_d(res))

        res = self.unpool1(F.relu(self.conv2_d(self.conv1_d(res))))
        res = self.unpool2(F.relu(self.conv4_d(self.conv3_d(res))))
        res = self.unpool3(F.relu(self.conv6_d(self.conv5_d(res))))
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
        epsilon = np.random.standard_normal(t_mean.shape)
        latent_t = t_mean + torch.sqrt(torch.exp(t_log_var)) * epsilon
        return latent_t

    def forward(self, x, label):
        hidden = self.encode(x, label)
        cond_t_mean = self.get_cond_t_mean(hidden)
        cond_t_log_var = self.get_cond_t_log_var(hidden)
        t_sampled = self.sampling(cond_t_mean, cond_t_log_var)
        decoded = self.decode(t_sampled, label)
        return decoded, cond_t_mean, cond_t_log_var

    @staticmethod
    def loss(x, reconstruction, t_log_var, t_mean, reconstruction_weight=1000):
        loss_reconstruction = torch.mean(nn.functional.mse_loss(x, reconstruction))
        loss_KL = - torch.mean(
            0.5 * torch.sum(1 + t_log_var - torch.square(t_mean) - torch.exp(t_log_var), dim=1)
        )
        loss = reconstruction_weight * loss_reconstruction + loss_KL
        return loss

