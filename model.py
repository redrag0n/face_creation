import time

import matplotlib.pyplot as plt
import tqdm
import torch
import json
import numpy as np
import pandas as pd
from conditional_vae import ConditionalVAE
from gan import GAN
import torchvision.utils as vutils


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class VaeModel:
    def __init__(self, cond_vae_params=None, cond_vae_model=None, model_save_path=None, device='cpu',
                 reconstruction_weight=1000):
        if cond_vae_params is not None:
            if model_save_path is not None:
                with open(model_save_path + '/' + 'params.json', mode='w') as out:
                    json.dump(cond_vae_params, out)
            self.conditional_vae = ConditionalVAE(**cond_vae_params)
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
        self.example_count = 25

    def fit(self, train_dataloader, test_dataloader=None, max_epochs=10, lr=0.01, save_model=False):
        train_losses = list()
        test_losses = list()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.conditional_vae.parameters()), lr=lr, weight_decay=0.0001)

        self._test_latent = np.random.normal(size=(self.example_count, self.conditional_vae.latent_dim_size))
        self._test_labels = np.random.randint(0, 2, (self.example_count, self.conditional_vae.label_shape))

        self.save_model_data(-1, test_dataloader, test_losses, train_losses)
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
                self.save_model_data(epoch, test_dataloader, test_losses, train_losses)

    def save_model_data(self, epoch, test_dataloader, test_losses, train_losses):
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

        generated = self.generate_from_t_sampled(self._test_latent, self._test_labels)
        fig, ax = plt.subplots(5, 5, figsize=(15, 15))
        for i in range(5):
            for j in range(5):
                ax[i, j].imshow(np.moveaxis(generated[i * 5 + j], 0, -1))
                ax[i, j].axis('off')
        plt.savefig(self.model_save_path + '/' + 'generated' + str(epoch) + '.png')
        plt.show()

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
        #print('label', label.shape)
        t_sampled = torch.Tensor(t_sampled).float().to(self.device)
        #print('t_sampled', t_sampled.shape)
        return sigmoid(self.conditional_vae.decoder(t_sampled, label).detach().cpu().numpy())

    def generate_random_with_label(self, label):
        t_sampled = np.random.normal(size=(len(label), self.conditional_vae.latent_dim_size))
        return self.generate_from_t_sampled(t_sampled, label)

    @staticmethod
    def load(model_path, device='cpu'):
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        return VaeModel(cond_vae_model=model, model_save_path=model_path, device=device)


def train_vae(config):
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
    vae_model = VaeModel(config.VAE_PARAMS, model_save_path=config.MODEL_SAVE_PATH, device=config.DEVICE,
                      reconstruction_weight=config.RECONSTRUCTION_WEIGHT)
    vae_model.fit(train_dataloader, test_dataloader, save_model=True, lr=config.LEARNING_RATE, max_epochs=config.MAX_EPOCHS)


class GanModel:
    def __init__(self, gan_params=None, gan_model=None, model_save_path=None, device='cpu'):
        if gan_params is not None:
            if model_save_path is not None:
                with open(model_save_path + '/' + 'params.json', mode='w') as out:
                    json.dump(gan_params, out)
            self.gan = GAN(**gan_params)
        else:
            self.gan = gan_model
        print('Model parameters:', sum(p.numel() for p in self.gan.parameters() if p.requires_grad))
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.gan = torch.nn.DataParallel(self.gan)
        self.gan.to(device)
        self.device = device
        self.model_save_path = model_save_path
        self.example_count = 25

    def fit(self, train_dataloader, test_dataloader=None, max_epochs=10, lr=0.01, save_model=False):
        d_losses = list()
        g_losses = list()

        d_xs = list()
        d_z1s = list()
        d_z2s = list()
        #img_list = list()
        g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.gan.generator.parameters()), lr=lr, weight_decay=0.0001)
        d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.gan.discriminator.parameters()), lr=lr, weight_decay=0.0001)

        self._test_noise = np.random.normal(size=(self.example_count, self.gan.latent_dim_size))
        self._test_labels = np.random.randint(0, 2, (self.example_count, self.gan.label_shape))

        self.save_model_data(-1, test_dataloader, d_losses, g_losses, d_xs, d_z1s, d_z2s)
        for epoch in range(max_epochs):
            epoch_d_losses = list()
            epoch_g_losses = list()

            # epoch_d_xs = list()
            # epoch_d_z1s = list()
            # epoch_d_z2s = list()
            # self.gan.discriminator.train()
            # self.gan.generator.train()
            train_loss, n, start = 0.0, 0, time.time()
            for X, y in tqdm.tqdm(train_dataloader, ncols=50):

                # Discriminator train
                X, y = X.to(self.device), y.to(self.device)
                b_size = X.shape[0]
                d_optimizer.zero_grad()
                d_predicted = self.gan.discriminator(X, y)
                true_ = torch.full((b_size,), 1, dtype=torch.float, device=self.device)
                d_loss_true = self.gan.__class__.loss(d_predicted, true_).to(self.device)
                d_loss_true.backward()
                d_optimizer.step()
                d_xs.append(d_predicted.mean().cpu().item())  # average discriminator prediction on true images

                ## Train with all-fake batch
                # Generate batch of latent vectors
                #noise = torch.randn(b_size, self.gan.latent_dim_size, 1, 1, device=self.device)
                noise = torch.tensor(np.random.standard_normal((b_size, self.gan.latent_dim_size)), requires_grad=False).float().to(self.device)
                random_labels = torch.tensor(np.random.randint(0, 1, (b_size, self.gan.label_shape)), requires_grad=False).float().to(self.device)
                # Generate fake image batch with generator
                fake = self.gan.generator(noise, random_labels)
                true_.fill_(0)
                # Classify all fake batch with D
                output = self.gan.discriminator(fake.detach(), random_labels)
                # Calculate D's loss on the all-fake batch
                d_loss_fake = self.gan.__class__.loss(output, true_)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                d_loss_fake.backward()
                d_z1s.append(output.mean().cpu().item())
                # Compute error of D as sum over the fake and the real batches
                d_loss = d_loss_true + d_loss_fake
                # Update D
                d_optimizer.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                # self.gan.generator.train()
                # self.gan.discriminator.eval()
                g_optimizer.zero_grad()
                true_.fill_(1)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.gan.discriminator(fake, random_labels)
                # Calculate G's loss based on this output
                g_loss = self.gan.__class__.loss(output, true_)
                # Calculate gradients for G
                g_loss.backward()
                d_z2s.append(output.mean().cpu().item())
                # Update G
                g_optimizer.step()

                # # Output training stats
                # if i % 50 == 0:
                #     print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                #           % (epoch, num_epochs, i, len(dataloader),
                #              errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                epoch_g_losses.append(g_loss.cpu().item())
                epoch_d_losses.append(d_loss.cpu().item())

            # # Check how the generator is doing by saving G's output on fixed_noise
            # with torch.no_grad():
            #     fake = self.gan.generator(self._test_noise).detach().cpu()
            # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            mean_epoch_g_loss = np.mean(epoch_g_losses)
            mean_epoch_d_loss = np.mean(epoch_d_losses)

            g_losses.append(mean_epoch_g_loss)
            d_losses.append(mean_epoch_d_loss)

            print('epoch %d, discriminator train loss %.4f , time %.1f sec'
                  % (epoch, mean_epoch_d_loss, time.time() - start))

            print('epoch %d, generator train loss %.4f , time %.1f sec'
                  % (epoch, mean_epoch_g_loss, time.time() - start))

            # if test_dataloader is not None:
            #     test_losses.append(self.test(test_dataloader))
            if self.model_save_path is not None and save_model:
                self.save_model_data(epoch, test_dataloader, d_losses, g_losses, d_xs, d_z1s, d_z2s)

    def save_model_data(self, epoch, test_dataloader, d_losses, g_losses, d_xs, d_z1s, d_z2s):
        torch.save(self.gan, self.model_save_path + '/' + 'epoch' + str(epoch))

        metric_df = pd.DataFrame({'epoch': list(range(epoch + 1)), 'discriminator': d_losses, 'generator': g_losses}).set_index('epoch')

        metric_df.to_excel(self.model_save_path + '/' + 'metrics.xlsx')

        plt.figure(figsize=(10, 10))
        plt.plot(list(range(epoch + 1)), d_losses, color='blue', label='discriminator')
        plt.plot(list(range(epoch + 1)), g_losses, color='green', label='generator')
        plt.legend()
        plt.savefig(self.model_save_path + '/' + 'metrics.png', dpi=300, bbox_inches='tight')

        plt.figure(figsize=(10, 10))
        plt.plot(list(range(len(d_xs))), d_xs, color='blue', label='true discriminator')
        plt.plot(list(range(len(d_z1s))), d_z1s, color='green', label='fake discriminator first')
        plt.plot(list(range(len(d_z2s))), d_z2s, color='orange', label='fake discriminator last')
        plt.legend()
        plt.savefig(self.model_save_path + '/' + 'answers.png', dpi=300, bbox_inches='tight')

        generated = self.generate_from_t_sampled(self._test_noise, self._test_labels)
        fig, ax = plt.subplots(5, 5, figsize=(15, 15))
        for i in range(5):
            for j in range(5):
                ax[i, j].imshow(np.moveaxis(generated[i * 5 + j], 0, -1))
                ax[i, j].axis('off')
        plt.savefig(self.model_save_path + '/' + 'generated' + str(epoch) + '.png')
        #plt.show()

    # def test(self, test_dataloader):
    #     self.conditional_vae.eval()
    #     test_loss = 0
    #     n = 0
    #     with torch.no_grad():
    #         for X, y in test_dataloader:
    #             X, y = X.to(self.device), y.to(self.device)
    #             reconstructed, t_mean, t_log_var, _ = self.conditional_vae(X, y)
    #             test_loss += self.conditional_vae.__class__.loss(X, reconstructed, t_mean, t_log_var, self.reconstruction_weight).to(self.device).cpu().item()
    #             n += X.shape[0]
    #     test_loss /= n
    #     print(f"Test Avg loss: {test_loss:>8f} \n")
    #     return test_loss

    # def transform(self, X, y):
    #     return self.conditional_vae(X, y)[0]

    def generate_from_t_sampled(self, t_sampled, label):
        label = torch.Tensor(label).float().to(self.device)
        # #print('label', label.shape)
        t_sampled = torch.Tensor(t_sampled).float().to(self.device)
        #print('t_sampled', t_sampled.shape)
        self.gan.generator.eval()
        return sigmoid(self.gan.generator(t_sampled, label).detach().cpu().numpy())

    def generate_random_with_label(self, label):
        t_sampled = np.random.normal(size=(len(label), self.gan.latent_dim_size))
        return self.generate_from_t_sampled(t_sampled, label)

    @staticmethod
    def load(model_path, device='cpu'):
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        return GanModel(gan_model=model, model_save_path=model_path, device=device)


def train_gan(config):
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
    gan_model = GanModel(config.VAE_PARAMS, model_save_path=config.MODEL_SAVE_PATH, device=config.DEVICE)
    gan_model.fit(train_dataloader, test_dataloader, save_model=True, lr=config.LEARNING_RATE, max_epochs=config.MAX_EPOCHS)
