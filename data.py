import torch
from tqdm import tqdm
import time
import torch.nn
import os
import random
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import config
from PIL import Image
import cv2
from torchvision import transforms
import albumentations as A


class DegDataset(Dataset):
    def __init__(self, clear_imgs_dir, deg_imgs_dir, patch_size=None):
        self.clear_imgs_dir = clear_imgs_dir
        self.deg_imgs_dir = deg_imgs_dir
        self.patch_size = patch_size

        self.clear_images = [
            file for file in os.listdir(clear_imgs_dir) if
            file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg')
        ]

        self.deg_images = [
            file for file in os.listdir(deg_imgs_dir) if
            file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg')
        ]

        self.both_transform = transforms.Compose(
            [
                A.RandomCrop(width=config.PATCH_SIZE, height=config.PATCH_SIZE),
                transforms.ToTensor(),
            ]
        )

    def transform(self, clear_image, deg_image, crop_size):
        # Resize
        if self.patch_size is not None:
            resize = transforms.Resize(size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
            clear_image = resize(clear_image)
            deg_image = resize(deg_image)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(clear_image, output_size=(crop_size, crop_size))

        clear_image = TF.crop(clear_image, i, j, h, w)
        deg_image = TF.crop(deg_image, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            clear_image = TF.hflip(clear_image)
            deg_image = TF.hflip(deg_image)

        # Random vertical flipping
        if random.random() > 0.5:
            clear_image = TF.vflip(clear_image)
            deg_image = TF.vflip(deg_image)

        # Transform to tensor
        clear_image = TF.to_tensor(clear_image)
        deg_image = TF.to_tensor(deg_image)

        return deg_image, clear_image

    def __getitem__(self, index):

        clear_img = Image.open(self.clear_imgs_dir + self.clear_images[index]).convert("RGB")
        deg_img = Image.open(self.deg_imgs_dir + self.deg_images[index]).convert("RGB")

        img_deg, img_clear = self.transform(clear_img, deg_img, crop_size=config.PATCH_SIZE)

        return img_deg.type(torch.float32), img_clear.type(torch.float32)

    def __len__(self):
        return len(self.clear_images)


def test():
    def process(arr):
        arr = arr.squeeze(0).permute(1, 2, 0).numpy()
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return arr

    def show_images(clear_img, deg_img):
        cv2.imshow("clear", clear_img)
        cv2.imshow("deg", deg_img)

        cv2.waitKey(-1)

    bs = 5
    dataset = DegDataset(clear_imgs_dir=config.TRAIN_CLEAR_DIR, deg_imgs_dir=config.TRAIN_DEG_DIR,
                         patch_size=config.PATCH_SIZE)
    loader = DataLoader(dataset, batch_size=bs, num_workers=1)

    examples = next(iter(loader))
    deg, clear = examples

    for i in range(bs):
        print(clear[i].shape)
        print(deg[i].shape)

        deg_img = process(deg[i])
        clear_img = process(clear[i])

        show_images(clear_img, deg_img)


if __name__ == "__main__":
    test()
