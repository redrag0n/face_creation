import time
import tqdm
import torch
import numpy as np
from conditional_vae import ConditionalVAE1


class Model:
    def __init__(self, cond_vae_params=None, cond_vae_model=None, model_save_path=None, device='cpu'):
        if cond_vae_params is not None:
            self.conditional_vae = ConditionalVAE1(**cond_vae_params)
        else:
            self.conditional_vae = cond_vae_model
        print('Model parameters:', sum(p.numel() for p in self.conditional_vae.parameters() if p.requires_grad))
        self.conditional_vae.to(device)
        self.device = device
        self.model_save_path = model_save_path

    def fit(self, train_dataloader, test_dataloader=None, max_epochs=10, lr=0.01, save_model=False):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.conditional_vae.parameters()), lr=lr, weight_decay=0.0001)
        for epoch in range(max_epochs):
            self.conditional_vae.train()
            train_loss, n, start = 0.0, 0, time.time()
            for X, y in tqdm.tqdm(train_dataloader, ncols=50):
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                reconstructed, t_mean, t_log_var = self.conditional_vae(X, y)
                l = self.conditional_vae.__class__.loss(X, reconstructed, t_mean, t_log_var).to(self.device)

                l.backward()

                # print('conv1', self.conditional_vae.conv1.weight.grad[0])
                # print('conv5_d', self.conditional_vae.conv5_d.weight.grad[0])
                optimizer.step()

                train_loss += l.cpu().item()
                n += X.shape[0]

            train_loss /= n
            print('epoch %d, train loss %.4f , time %.1f sec'
                  % (epoch, train_loss, time.time() - start))

            if test_dataloader is not None:
                self.test(test_dataloader)
            if self.model_save_path is not None and save_model:
                torch.save(self.conditional_vae, self.model_save_path + '/' + 'epoch' + str(epoch))

    def test(self, test_dataloader):
        self.conditional_vae.eval()
        test_loss = 0
        n = 0
        with torch.no_grad():
            for X, y in test_dataloader:
                X, y = X.to(self.device), y.to(self.device)
                reconstructed, t_mean, t_log_var = self.conditional_vae(X, y)
                test_loss += self.conditional_vae.__class__.loss(X, reconstructed, t_mean, t_log_var).to(self.device).cpu().item()
                n += X.shape[0]
        test_loss /= n
        print(f"Test Avg loss: {test_loss:>8f} \n")
        return test_loss

    def transform(self, X, y):
        return self.conditional_vae(X, y)[0]

    def generate_from_t_sampled(self, t_sampled, label):
        return self.conditional_vae.decode(t_sampled, label)

    def generate_random_with_label(self, label):
        t_sampled = torch.Tensor([np.random.random(self.conditional_vae.latent_dim_size)]).float()
        label = torch.Tensor(label).float().to(self.device)
        return self.generate_from_t_sampled(t_sampled, label)

    @staticmethod
    def load(model_path, device='cpu'):
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        return Model(cond_vae_model=model, model_save_path=model_path, device=device)
