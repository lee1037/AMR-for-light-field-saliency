import os

import numpy as np
import PIL.Image
import torch
from torchvision import transforms
from torch.utils import data
import random
import pdb
from PIL import Image
from transform_images import RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, Normalize, Resize, Rotation
import glob

class LFdata(data.Dataset):
    """
    Light field dataset.
    root: director/to/images/
    structure:
    - root
        - focus_tack
        - ground_truth
    """
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    def __init__(self, root, datatag='train', is_training=True):
        super(LFdata, self).__init__()
        self.root = root
        self.is_training = is_training
        self.focus_root = os.path.join(self.root, 'focus_stack')
        self.gt_root = os.path.join(self.root, 'ground_truth')
        self.datatag = datatag
        filenames = os.listdir(self.gt_root)

        # Read the test fold.
        with open(self.root + '1fold.txt', 'r') as f:
            testnames = f.read().splitlines()

        if datatag == 'train':
            self.names = [i for i in filenames if i not in testnames]
        elif datatag == 'test':
            self.names = testnames

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        img_stack = []
        focus_list = glob.glob(self.focus_root + '/' + self.names[index][:-4] +'*.jpg')
        # load the focus stack.
        for focus_file in focus_list:
            if not focus_file[-4:] in ['.jpg','.png']:
                continue
            focus = PIL.Image.open(focus_file)
            focus = focus.resize((224, 224))
            img_stack.append(focus)
        gt_file = os.path.join(self.gt_root, self.names[index][:-4]+'.png')
        gt = PIL.Image.open(gt_file).convert('L')
        img_size= gt.size
        gt = gt.resize((224, 224))
        img_stack.append(gt)

        # Data augmentation. Randomly flip, rotation.
        if self.is_training:
            img_stack = RandomVerticalFlip()(img_stack)
            img_stack = RandomHorizontalFlip()(img_stack)
            img_stack = Rotation()(img_stack)

        focus_stack = Normalize(self.mean, self.std)(img_stack[:-1])
        focus_stack = torch.from_numpy(np.array(focus_stack))
        gt = torch.from_numpy(np.array(img_stack[-1])/255)

        return focus_stack, gt, self.names[index][:-4], img_size


class RGBdata(data.Dataset):
    """
    RGB dataset
    root: director/to/images/
    structure:
    - root
        - Imgs
        - GT (ground truth)
    """

    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    def __init__(self, root, datatag='train', is_training=True):
        super(RGBdata, self).__init__()
        self.root = root
        self.is_training = is_training

        self.img_root = os.path.join(self.root, 'Imgs')
        self.gt_root = os.path.join(self.root, 'GT')

        file_names = os.listdir(self.img_root)
        self.names = file_names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # Load image.
        img = PIL.Image.open(os.path.join(self.img_root, self.names[index]))
        img = img.resize((224, 224))
        gt_file = os.path.join(self.gt_root, self.names[index][:-4] + '.png')
        gt = PIL.Image.open(gt_file).convert('L')
        gt = gt.resize((224, 224))

        if self.is_training:
            img, gt = RandomVerticalFlip()([img, gt])
            img, gt = RandomHorizontalFlip()([img, gt])
            img, gt = Rotation()([img, gt])

        img = np.array(img, dtype=np.float32) / 255
        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(np.array(img))
        img_size = img.shape

        gt = np.array(gt, dtype=np.float32)
        gt /= 255
        gt = torch.from_numpy(gt)

        return img, gt, self.names[index], img_size

