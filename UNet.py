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



def double_conv(in_chan ,out_chan):
    conv = nn.Sequential(
        nn.BatchNorm2d(in_chan),
        nn.Conv2d(in_chan ,out_chan, kernel_size = 3, padding = 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_chan ,out_chan, kernel_size = 3, padding = 1),
        nn.ReLU(inplace=True)
    )
    return conv


def crop(tensor, target):
    """ Crop tensor to target size """
    _, _, tensor_height, tensor_width = tensor.size()
    _, _, crop_height, crop_width = target.size()
    left = (tensor_width - crop_height) // 2
    top = (tensor_height - crop_width) // 2
    right = left + crop_width
    bottom = top + crop_height
    cropped_tensor = tensor[:, :, top: bottom, left: right]
    return cropped_tensor

class UNet(nn.Module):
    def __init__(self, init_chan, input_chan):
        super(UNet, self).__init__() # UNet inherits all methods in the class nn.Module

        self.input_chan = input_chan
        self.init_chan = init_chan

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_conv1 = double_conv(self.input_chan ,self.init_chan)
        self.down_conv2 = double_conv(self.init_chan ,self.init_chan *2)
        self.down_conv3 = double_conv(self.init_chan *2 ,self.init_chan *4)

        self.down_conv4 = double_conv(self.init_chan *4 ,self.init_chan *8)
        self.down_conv5 = double_conv(self.init_chan *8 ,self.init_chan *16)

        self.trans_conv1 = nn.ConvTranspose2d(in_channels=self.init_chan *16, out_channels=self.init_chan *8, kernel_size=2, stride=2)
        self.trans_conv2 = nn.ConvTranspose2d(in_channels=self.init_chan *8, out_channels=self.init_chan *4, kernel_size=2, stride=2)
        self.trans_conv3 = nn.ConvTranspose2d(in_channels=self.init_chan *4, out_channels=self.init_chan *2, kernel_size=2, stride=2)
        self.trans_conv4 = nn.ConvTranspose2d(in_channels=self.init_chan *2, out_channels=self.init_chan, kernel_size=2, stride=2)

        self.up_conv1 = double_conv(self.init_chan *16 ,self.init_chan *8)
        self.up_conv2 = double_conv(self.init_chan *8 ,self.init_chan *4)
        self.up_conv3 = double_conv(self.init_chan *4 ,self.init_chan *2)

        self.up_conv4 = double_conv(self.init_chan *2 ,self.init_chan)


        self.out = nn.Conv2d(in_channels = self.init_chan, out_channels = 1, kernel_size=1)
        self.Sigmoid = nn.Sigmoid()


    def path(self, image):
        x1 = self.down_conv1(image)
        x = self.max_pool(x1)
        x2 = self.down_conv2(x)
        x = self.max_pool(x2)
        x3 = self.down_conv3(x)
        x = self.max_pool(x3)
        x4 = self.down_conv4(x)
        x = self.max_pool(x4)
        x5 = self.down_conv5(x)
        x = self.trans_conv1(x5)
        y = crop(x4 ,x)
        x = self.up_conv1(torch.cat([y ,x], 1))
        x = self.trans_conv2(x)
        y = crop(x3 ,x)
        x = self.up_conv2(torch.cat([y ,x], 1))
        x = self.trans_conv3(x)
        y = crop(x2 ,x)
        x = self.up_conv3(torch.cat([y ,x], 1))
        x = self.trans_conv4(x)
        y = crop(x1 ,x)
        x = self.up_conv4(torch.cat([y ,x], 1))
        x = self.out(x)
        x = self.Sigmoid(x)
        return x
