import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from torchvision.io import read_image
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset


IMG_SHAPE_OLD = (3, 218, 178)
IMG_SHAPE = (3, 224, 184)


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
        image = read_image(img_path) / 255.
        label = torch.tensor(self.img_labels_df.iloc[idx][1:]).float()
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


data_transforms = transforms.Compose([
    transforms.Pad(padding=((IMG_SHAPE[2] - IMG_SHAPE_OLD[2]) // 2, (IMG_SHAPE[1] - IMG_SHAPE_OLD[1]) // 2),
                   padding_mode='edge'),
    #transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

# target_transforms = transforms.Compose([
#     transforms.ToTensor()
# ])


if __name__ == '__main__':
    dataset = CelebaDataset('data/list_attr_celeba.csv', 'data/img_align_celeba',
                            transform=data_transforms)
    # img = np.moveaxis(np.array(dataset[10][0]), 0, -1)
    # print(img.shape)

    # img = np.moveaxis(np.array(dataset[1][0]), 0, -1)
    # print(img.shape)

    # plt.imshow(img)
    # plt.show()
    print(dataset[0][0])
