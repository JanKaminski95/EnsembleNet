import matplotlib.pyplot as plt

from matplotlib import image as mpimg
import numpy as np
from PIL import Image
import random
from random import uniform
import glob
import torch
import math
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
import torchvision.transforms.functional as TF
from torch.utils.data.sampler import SubsetRandomSampler
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage import io,transform,filters,exposure
from skimage.measure import label, regionprops
from skimage.morphology import dilation
from skimage.util.shape import view_as_windows
from skimage.color import label2rgb
torch.manual_seed(95)
np.random.seed(95)

class MyDataset(Dataset):
    def __init__(self, root, n_slice, contrast, hflip=False, vflip=False, rotation=False, p=0.5):
        super().__init__()
        self.root = root
        self.hflip = hflip
        self.vflip = vflip
        self.n_slice = n_slice
        self.contrast = contrast
        self.rot = rotation
        self.p = p
        self.rotation = rotation
        targets_path = sorted(glob.glob('/home/janek/cluster/data/ensemble/' + root + '/targets/' + n_slice + '.tif'))
        self.targets = [io.imread(i) for i in targets_path]
        images_path = sorted(glob.glob('/home/janek/cluster/data/ensemble/' + root + '/images/' + n_slice + '.tif'))
        self.images = [io.imread(i) for i in images_path]
    def __getitem__(self, index):

        image = self.images[index]
        image = image/image.max()
        
        if len(self.targets) == 0:
            mask = np.zeros((512, 512))
        else:
            mask = self.targets[index]
        
        mask = mask/np.max(mask)
        
        do_hflip = random.random() > self.p

        do_vflip = random.random() > self.p

        do_rotation = random.random() > self.p

        rot = round(uniform(0, 3.6), 2)
        rot = rot * 100

        if self.contrast != 0:
            image = exposure.equalize_adapthist(image, clip_limit=self.contrast)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        # image = image.unsqueeze(0)
        # mask= mask.unsqueeze(0)

        if self.hflip and do_hflip:
            image, mask = TF.hflip(image), TF.hflip(mask)

        if self.vflip and do_vflip:
            image, mask = TF.vflip(image), TF.vflip(mask)

        if self.rotation and do_rotation:
            image, mask = TF.rotate(image, rot), TF.rotate(mask, rot)

        return image, mask

    def __len__(self):
        return len(self.images)
