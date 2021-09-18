import torch
from model import Model
from celeba_dataset import CelebaDataset, IMG_SHAPE, data_transforms
from torch.utils.data import DataLoader, random_split

#PROJECT_DIR = '/Users/valeriamozarova/PycharmProjects/face_creation'
PROJECT_DIR = '/root/face_creation'
#DATA_DIR = f'{PROJECT_DIR}/data'
DATA_DIR = f'../data'


ANNOTATION_DATA_PATH = f'{DATA_DIR}/list_attr_celeba.csv'
DATA_PATH = f'{DATA_DIR}/img_align_celeba'

VAE_PARAMS = {
    'latent_dim_size': 8,
    'data_shape': IMG_SHAPE,
    'label_shape': None
}

MODEL_SAVE_PATH = f'{PROJECT_DIR}/models/'

BATCH_SIZE = 32
MAX_EPOCHS = 10
LEARNING_RATE = 0.01
DEVICE = 'cpu'


def train():
    dataset = CelebaDataset(ANNOTATION_DATA_PATH, DATA_PATH,
                            transform=data_transforms)
    VAE_PARAMS['label_shape'] = len(dataset[0][1])
    train_size = int(len(dataset) * 0.9)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True)
    vae_model = Model(VAE_PARAMS, model_save_path=MODEL_SAVE_PATH, device=DEVICE)
    vae_model.fit(train_dataloader, test_dataloader, save_model=True, lr=LEARNING_RATE, max_epochs=MAX_EPOCHS)


if __name__ == '__main__':
    train()