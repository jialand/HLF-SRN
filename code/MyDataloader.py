import os
from random import shuffle
import imageio
import random
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional
import time

class ImageFolder(data.Dataset):
    def __init__(self, root_path, input_size, patch_size, scale=4, valid_div=30, mode='train', dataset_len=-1):
        self.root = root_path
        self.input_size = input_size
        self.scale = scale
        self.mode = mode
        self.valid_div = valid_div
        self.dataset_len = None
        self.patch_size = patch_size
        if dataset_len > 0:
            self.dataset_len = dataset_len

        self.lr_paths = []
        cnt = 0
        if self.mode == "test":
            for path, _, file in os.walk(self.root):
                for file_name in file:
                    lr = os.path.join(path, file_name)
                    self.lr_paths.append(lr)
        else:
            self.lr_root = os.path.join(self.root, str(input_size) + "X" + str(input_size), "LR", "X" + str(scale))
            self.hr_paths = []
            for path, _, file in os.walk(self.lr_root):
                for file_name in file:
                    lr = os.path.join(path, file_name)
                    hr = lr.replace(os.path.join("LR", "X" + str(scale)), "HR")
                    if os.path.exists(hr):
                        if (cnt % 100 > valid_div and self.mode == "train") \
                                or (cnt % 100 < valid_div and self.mode == "valid"):
                            self.lr_paths.append(lr)
                            self.hr_paths.append(hr)
                        cnt += 1
                    else:
                        print("Find LR image: " + lr + ", but no corresponding HR image: " + hr)

        if not self.dataset_len is None:
            print("Fixed dataset length: " + str(self.dataset_len))
            if len(self.lr_paths) < self.dataset_len:
                print("not enough, use all dataset: " + str(self.dataset_len))
        else:
            if self.mode == 'test':
                print("Image count in {} path :{}".format(self.root, len(self.lr_paths)))
            else:
                print("Image count for {} :{}".format(self.mode, len(self.lr_paths)))

    def __getitem__(self, index):
        lr_path = self.lr_paths[index]
        lr = imageio.v2.imread(lr_path)

        if self.mode == "test":
            lr = np.ascontiguousarray(lr.transpose((2, 0, 1)))
            lr = torch.FloatTensor(lr).cuda()
            return lr, lr_path
        else:
            hr_path = self.hr_paths[index]
            hr = imageio.v2.imread(hr_path)
            lr, hr = self.get_patch(lr, hr)
            lr = np.ascontiguousarray(lr.transpose((2, 0, 1)))
            lr = torch.FloatTensor(lr).cuda()
            hr = np.ascontiguousarray(hr.transpose((2, 0, 1)))
            hr = torch.FloatTensor(hr).cuda()
            return lr, hr, lr_path, hr_path

    def __len__(self):
        if self.dataset_len is None:
            return len(self.lr_paths)
        else:
            if self.mode == "train":
                return min(self.dataset_len, len(self.lr_paths))
            else:
                return min(self.dataset_len * self.valid_div // 100, len(self.lr_paths))

    def get_patch(self, lr, hr):
        h, w = lr.shape[:2]
        xx = random.randrange(0, h - self.patch_size + 1)
        yy = random.randrange(0, w - self.patch_size + 1)

        hr_size = self.patch_size * self.scale
        hx = xx * self.scale
        hy = yy * self.scale

        plr = lr[xx:xx+self.patch_size, yy:yy+self.patch_size, :]
        phr = hr[hx:hx+hr_size, hy:hy+hr_size, :]

        return plr, phr




def get_loader(root_path, batch_size, input_size, scale, patch_size,
               num_workers=0, valid_div=30, mode='train', is_shuffle=False, dataset_len=-1):
    dataset = ImageFolder(root_path=root_path, input_size=input_size, patch_size=patch_size, scale=scale,
                          valid_div=valid_div, mode=mode, dataset_len=dataset_len)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=is_shuffle,
                                  num_workers=num_workers,
                                  drop_last=True)
    return data_loader
