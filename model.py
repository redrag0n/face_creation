import time

import matplotlib.pyplot as plt
import tqdm
import torch
import json
import numpy as np
import pandas as pd
from conditional_vae import ConditionalVAE1, ConditionalVAE, ConditionalVAE3


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Model:
    def __init__(self, cond_vae_params=None, cond_vae_model=None, model_save_path=None, device='cpu',
                 reconstruction_weight=1000):
        if cond_vae_params is not None:
            if model_save_path is not None:
                with open(model_save_path + '/' + 'params.json', mode='w') as out:
                    json.dump(cond_vae_params, out)
            self.conditional_vae = ConditionalVAE3(**cond_vae_params)
        else:
            self.conditional_vae = cond_vae_model
        print('Model parameters:', sum(p.numel() for p in self.conditional_vae.parameters() if p.requires_grad))
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.conditional_vae = torch.nn.DataParallel(self.conditional_vae)
        self.conditional_vae.to(device)
        self.device = device
        self.model_save_path = model_save_path
        self.reconstruction_weight = reconstruction_weight

    def fit(self, train_dataloader, test_dataloader=None, max_epochs=10, lr=0.01, save_model=False):
        train_losses = list()
        test_losses = list()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.conditional_vae.parameters()), lr=lr, weight_decay=0.0001)
        for epoch in range(max_epochs):
            self.conditional_vae.train()
            train_loss, n, start = 0.0, 0, time.time()
            for X, y in tqdm.tqdm(train_dataloader, ncols=50):
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                reconstructed, t_mean, t_log_var, _ = self.conditional_vae(X, y)
                l = self.conditional_vae.__class__.loss(X, reconstructed, t_mean, t_log_var, self.reconstruction_weight).to(self.device)

                #train_loss += l.cpu().item()
                l.backward()

                # print('conv1', self.conditional_vae.conv1.weight.grad[0])
                # print('conv5_d', self.conditional_vae.conv5_d.weight.grad[0])

                #self.conditional_vae.log_gradients()
                optimizer.step()
                train_loss += l.cpu().item()

                n += X.shape[0]

            train_loss /= n
            train_losses.append(train_loss)
            print('epoch %d, train loss %.4f , time %.1f sec'
                  % (epoch, train_loss, time.time() - start))

            if test_dataloader is not None:
                test_losses.append(self.test(test_dataloader))
            if self.model_save_path is not None and save_model:
                torch.save(self.conditional_vae, self.model_save_path + '/' + 'epoch' + str(epoch))
                metric_df = pd.DataFrame({'epoch': list(range(epoch + 1)), 'train': train_losses}).set_index('epoch')
                if test_dataloader is not None:
                    metric_df['test'] = test_losses
                metric_df.to_excel(self.model_save_path + '/' + 'metrics.xlsx')

                plt.figure(figsize=(10, 10))
                plt.plot(list(range(epoch + 1)), train_losses, color='blue', label='train')
                if test_dataloader is not None:
                    plt.plot(list(range(epoch + 1)), test_losses, color='green', label='test')
                plt.legend()
                plt.savefig(self.model_save_path + '/' + 'metrics.png', dpi=300, bbox_inches='tight')

    def test(self, test_dataloader):
        self.conditional_vae.eval()
        test_loss = 0
        n = 0
        with torch.no_grad():
            for X, y in test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                reconstructed, t_mean, t_log_var, _ = self.conditional_vae(X, y)
                test_loss += self.conditional_vae.__class__.loss(X, reconstructed, t_mean, t_log_var, self.reconstruction_weight).to(self.device).cpu().item()
                n += X.shape[0]
        test_loss /= n
        print(f"Test Avg loss: {test_loss:>8f} \n")
        return test_loss

    def transform(self, X, y):
        return self.conditional_vae(X, y)[0]

    def generate_from_t_sampled(self, t_sampled, label):
        label = torch.Tensor(label).float().to(self.device)
        print('label', label.shape)
        t_sampled = torch.Tensor(t_sampled).float()
        print('t_sampled', t_sampled.shape)
        return sigmoid(self.conditional_vae.decode(t_sampled, label).detach().numpy())

    def generate_random_with_label(self, label):
        t_sampled = np.random.normal(size=(len(label), self.conditional_vae.latent_dim_size))
        return self.generate_from_t_sampled(t_sampled, label)

    @staticmethod
    def load(model_path, device='cpu'):
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        return Model(cond_vae_model=model, model_save_path=model_path, device=device)


def train(config):
    from celeba_dataset import CelebaDataset, IMG_SHAPE, data_transforms
    from torch.utils.data import DataLoader, random_split
    dataset = CelebaDataset(config.ANNOTATION_DATA_PATH, config.DATA_PATH,
                            transform=data_transforms)
    config.VAE_PARAMS['label_shape'] = len(dataset[0][1])
    train_size = int(len(dataset) * 0.9)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, config.BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, config.BATCH_SIZE, shuffle=True)
    vae_model = Model(config.VAE_PARAMS, model_save_path=config.MODEL_SAVE_PATH, device=config.DEVICE,
                      reconstruction_weight=config.RECONSTRUCTION_WEIGHT)
    vae_model.fit(train_dataloader, test_dataloader, save_model=True, lr=config.LEARNING_RATE, max_epochs=config.MAX_EPOCHS)