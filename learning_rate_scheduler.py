import numpy as np


class BaseLearningRateScheduler:
    def __init__(self, cycle_size=100):
        self.cycle_size = cycle_size

    def update(self, old_generator_lr, old_discriminator_lr,
               real_discriminator_preds, fake_discriminator_preds):
        return old_generator_lr, old_discriminator_lr


class CompareLearningRateScheduler(BaseLearningRateScheduler):
    def __init__(self, cycle_size, base_lr, down_coef=2, dif_threshold=0.2):
        BaseLearningRateScheduler.__init__(self, cycle_size)
        self.base_lr = base_lr
        self.down_coef = down_coef
        self.dif_threshold = dif_threshold

    def update(self, old_generator_lr, old_discriminator_lr,
               real_discriminator_preds, fake_discriminator_preds):
        if np.mean(real_discriminator_preds) > np.mean(fake_discriminator_preds) + self.dif_threshold:
            if old_discriminator_lr == old_generator_lr:
                new_generator_lr = self.base_lr
                new_discriminator_lr = self.base_lr / self.down_coef
            elif old_discriminator_lr > old_generator_lr:
                new_generator_lr = old_generator_lr * self.down_coef
                new_discriminator_lr = old_discriminator_lr
            else:
                new_generator_lr = self.base_lr
                new_discriminator_lr = new_generator_lr / (self.down_coef * (old_generator_lr / old_discriminator_lr))

        elif ((np.mean(real_discriminator_preds) + self.dif_threshold < np.mean(fake_discriminator_preds)) or \
              (np.mean(fake_discriminator_preds) - 0.5 > self.dif_threshold) or
              (np.mean(real_discriminator_preds)) < 0.5):
            if old_discriminator_lr == old_generator_lr:
                new_generator_lr = self.base_lr / self.down_coef
                new_discriminator_lr = self.base_lr
            elif old_discriminator_lr < old_generator_lr:
                new_generator_lr = old_generator_lr
                new_discriminator_lr = old_discriminator_lr * self.down_coef
            else:
                new_discriminator_lr = self.base_lr
                new_generator_lr = new_discriminator_lr / (self.down_coef * (old_discriminator_lr / old_generator_lr))
        else:
            new_generator_lr = old_generator_lr
            new_discriminator_lr = old_discriminator_lr

        return new_generator_lr, new_discriminator_lr
