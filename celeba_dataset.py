import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from torchvision.io import read_image
from torch.utils.data import Dataset


IMG_SHAPE = (3, 218, 178)


class CelebaDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels_df = pd.read_csv(annotations_file, delimiter=r"\s+")
        self.labels = self.img_labels_df.columns
        self.img_labels_df.replace(-1, 0, inplace=True)
        self.img_labels_df.index.name = 'id'
        self.img_labels_df.reset_index(inplace=True)

        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels_df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels_df.iloc[idx]['id'])
        image = read_image(img_path)
        label = np.array(self.img_labels_df.iloc[idx])[1:]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == '__main__':
    dataset = CelebaDataset('data/list_attr_celeba.csv', 'data/img_align_celeba')
    img = np.moveaxis(np.array(dataset[0][0]), 0, -1)
    print(img.shape)

    img = np.moveaxis(np.array(dataset[1][0]), 0, -1)
    print(img.shape)

    # plt.imshow(img)
    # plt.show()
    # print(dataset[0][1])
