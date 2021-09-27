from celeba_dataset import IMG_SHAPE

#PROJECT_DIR = '/Users/valeriamozarova/PycharmProjects/face_creation'
PROJECT_DIR = '/root/face_creation'
#DATA_DIR = f'{PROJECT_DIR}/data'
DATA_DIR = f'../data'


ANNOTATION_DATA_PATH = f'{DATA_DIR}/list_attr_celeba.csv'
DATA_PATH = f'{DATA_DIR}/img_align_celeba'

VAE_PARAMS = {
    'latent_dim_size': 8,
    'data_shape': IMG_SHAPE,
    'label_shape': None,
    'layer_count': 4,
    'base_filters': 64,
    'kernel_size': (5, 5)
}
RECONSTRUCTION_WEIGHT = 100

MODEL_SAVE_PATH = f'{PROJECT_DIR}/models/'

BATCH_SIZE = 32
MAX_EPOCHS = 15
LEARNING_RATE = 0.001
DEVICE = 'cpu'
#DEVICE = 'cuda'