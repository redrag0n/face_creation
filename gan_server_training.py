from configs import server_config
from model import train_gan, train_gan_with_pretrain_generator


#train_gan_with_pretrain_generator(server_config)
train_gan(server_config)
