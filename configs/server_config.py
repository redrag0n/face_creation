import copy
from celeba_dataset import IMG_SHAPE


PROJECT_DIR = '/root/face_creation'
DATA_DIR = f'{PROJECT_DIR}/data'


ANNOTATION_DATA_PATH = f'{DATA_DIR}/list_attr_celeba.csv'
DATA_PATH = f'{DATA_DIR}/img_align_celeba'

# VAE_PARAMS = {  # good for vae
#     'latent_dim_size': 32,
#     'data_shape': IMG_SHAPE,
#     'label_shape': None,
#     'layer_count': 4,
#     'base_filters': 64,
#     'kernel_size': (5, 5),
#     'label_resize_shape': 64,
#     'use_add_layer': True
# }
VAE_PARAMS = {
    'latent_dim_size': 100,
    'data_shape': IMG_SHAPE,
    'label_shape': None,
    'layer_count': 4,
    'base_filters': 64,
    'kernel_size': (3, 3),
    'label_resize_shape': 64,
    'use_add_layer': False
}

GENERATOR_PARAMS = copy.deepcopy(VAE_PARAMS)

DISCRIMINATOR_PARAMS = {
    # 'latent_dim_size': 1,
    'data_shape': IMG_SHAPE,
    'label_shape': None,
    'layer_count': 4,
    'base_filters': 64,
    'kernel_size': (3, 3),
    'label_resize_shape': 32,
    'use_add_layer': True,
    'last_non_linearity': 'sigmoid'
}

GAN_PARAMS = {
    'generator_params': GENERATOR_PARAMS,
    'discriminator_params': DISCRIMINATOR_PARAMS,
    'loss_f': 'bce'
}
RECONSTRUCTION_WEIGHT = 1000
USE_LABELS = False

MODEL_SAVE_PATH = f'{PROJECT_DIR}/trained_models/'

BATCH_SIZE = 32
MAX_EPOCHS = 100
LEARNING_RATE = 0.0002
DEVICE = 'cuda'
NUM_WORKERS = 4
BETA = (0.5, 0.999)

SCHEDULER_PARAMS = {
    'cycle_size': 100,
    'base_lr': LEARNING_RATE,
    'down_coef': 2,
    'dif_threshold': 0.2
}