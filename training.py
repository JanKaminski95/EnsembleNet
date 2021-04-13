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

from Dataset import *
from training import *
from UNet import *


def F1(y_pred, y_true):
    y_pred = (y_pred > 0.5).float()
    tp = (y_true * y_pred).sum()
    tn = ((1 - y_true) * (1 - y_pred)).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def dice_loss(pred, target):
  preds_sq = pred**2
  target_sq = target**2
  return 1 - (2. * (torch.sum(pred * target))) / (preds_sq.sum() + target_sq.sum())


def train_UNet(mode,train_data, val_data, init_chan, batch_size, train_idx, val_idx, num, lr=1e-4, n_epochs=400,patience=30):
    log_file = '/home/Jan/Insitu/UNet_models/log/training_Unet.txt'
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            print('init_chan\tbatch_size\tnum\ttrain_loss \ttrain_acc\tval_loss\tval_acc', file=f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loss_final_models = []
    train_acc_final_models = []
    val_loss_final_models = []
    val_acc_final_models = []
    models = []
    print("NEW MODEL!!!!!!!!!!!!!!!!!!!!")
    net = UNet(init_chan, 1)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0)
    train = torch.utils.data.Subset(train_data, train_idx)
    val = torch.utils.data.Subset(val_data, val_idx)
    traindataload = DataLoader(train, batch_size=batch_size, shuffle=True)
    valdataload = DataLoader(val, batch_size=1, shuffle=False)
    best_score = 0
    counter = 0
    train_loss_final = []
    train_acc_final = []
    val_loss_final = []
    val_acc_final = []
    for epoch in range(n_epochs):
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        for step, (image, target) in enumerate(traindataload):
            if torch.sum(image) == 0:
                continue
            else:
                image = image.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                pred = net.path(image)
                loss = dice_loss(pred, target)
                loss.backward()
                optimizer.step()
                acc = F1(pred, target)
                if math.isnan(acc.item()):
                    continue
                else:
                    train_loss.append(loss.item())
                    train_acc.append(acc.item())
                print("Epoch [{}/{}], Step [{}/{}], training_loss:{:.4f}, training_acc:{:.4f}".format(epoch + 1, n_epochs,step+1, len(traindataload), loss.item(), acc.item()))
        with torch.no_grad():
            for idx, (val_image, val_target) in enumerate(valdataload):
                val_target=val_target.to(device)
                pred = net.path(val_image)
                loss = dice_loss(pred, val_target)
                acc = F1(pred, val_target)
                if math.isnan(acc.item()):
                    continue
                else:
                    val_loss.append(loss.item())
                    val_acc.append(acc.item())
        train_loss_final.append(np.mean(train_loss))
        train_acc_final.append(np.mean(train_acc))
        val_loss_final.append(np.mean(val_loss))
        val_acc_final.append(np.mean(val_acc))
        print('Epoch [{}/{}], TrLoss: {:.4f}, TrAcc: {:.4f}, VlLoss: {:.4f}, VlAcc: {:.4f}'.format(epoch + 1, n_epochs,np.mean(train_loss),np.mean(train_acc),np.mean(val_loss), np.mean(val_acc)))
        val_mean = np.mean(val_loss)
        train_mean = np.mean(train_loss)
        if best_score == 0:
            print("START!!!")
            best_score = val_mean
            counter += 1
        elif math.isnan(val_mean):
            print("NAN", best_score)
            continue
        elif val_mean >= best_score:
            print("BAD!!!!")
            counter += 1
            if counter >= patience:
                print("Model starts to overfit")
                break
        else:
            best_score = val_mean
            best_train = train_mean
            val_acc_best = np.mean(val_acc)
            train_acc_best = np.mean(train_acc)
            # final_opt_thr = optThresh
            print("GOOD!!!")
            torch.save(net.state_dict(),'/home/Jan/Insitu/UNet_models/mode_' + mode + '_best_model_' + str(num) + '_unet_initchan_' + str(init_chan) + '_batch_size' + str(batch_size) + '.pt')
            counter = 0
        print("model added")
        train_loss_final_models.append(train_loss_final)
        train_acc_final_models.append(train_acc_final)
        val_loss_final_models.append(val_loss_final)
        val_acc_final_models.append(val_acc_final)
    print(best_score, 'VAL LOSS')
    print(best_train, 'TRAIN LOSS')

    fig = plt.figure(figsize=(10, 10))
    plt.plot(val_loss_final_models[0], label='Validation loss')
    plt.plot(train_loss_final_models[0], label='Training loss')
    plt.legend()
    plt.savefig('/home/Jan/Insitu/plots_learning_curves/by_loss/learning_curve_unet' + str(num) + '_unet_' + str(init_chan) + '_batch_size_' + str(batch_size) + '.pdf')
    with open(log_file, 'a') as f:
        print('%f\t%f\t%f\t%f\t%f\t%f\t%f' % (
        init_chan, batch_size, num, best_train, train_acc_best, best_score, val_acc_best), file=f)


def cross_val_training():
    train_data_P1 = MyDataset('training', '*P1*', 0, True, True, True, 0.5)
    val_data_P1 = MyDataset('training', '*P1*', 0, False, False, False, 0.5)

    train_data_P28 = MyDataset('training', '*P28*', 0, True, True, True, 0.5)
    val_data_P28 = MyDataset('training', '*P28*', 0, False, False, False, 0.5)

    train_data_pulled = MyDataset('training', '*P*', 0, True, True, True, 0.5)
    val_data_pulled = MyDataset('training', '*P*', 0, False, False, False, 0.5)

    idx_list = [i for i in range(52)]
    idx_list1 = [i for i in range(52)]
    val_indexes = []
    for k in range(4):
        test = random.sample(idx_list, 13)
        test = sorted(test)
        val_indexes.append(test)
        for i in test:
            idx_list.remove(i)
    train_indexes = []
    for i in val_indexes:
        train = [x for x in idx_list1 if x not in i]
        train = sorted(train)
        train_indexes.append(train)
    for i in range(4):
        train_idx = train_indexes[i]
        val_idx = val_indexes[i]
        train_UNet(mode='P1',train_data=train_data_P1, val_data=val_data_P1, init_chan=32, batch_size=2, train_idx=train_idx,
                   val_idx=val_idx, num=i)

    idx_list = [i for i in range(20)]
    idx_list1 = [i for i in range(20)]
    val_indexes = []
    for k in range(5):
        test = random.sample(idx_list, 4)
        test = sorted(test)
        val_indexes.append(test)
        for i in test:
            idx_list.remove(i)
    train_indexes = []
    for i in val_indexes:
        train = [x for x in idx_list1 if x not in i]
        train = sorted(train)
        train_indexes.append(train)
    for i in range(5):
        train_idx = train_indexes[i]
        val_idx = val_indexes[i]
        train_UNet(mode='P28',train_data=train_data_P28, val_data=val_data_P28, init_chan=32, batch_size=2, train_idx=train_idx,
                   val_idx=val_idx, num=i)

    idx_list = [i for i in range(72)]
    idx_list1 = [i for i in range(72)]
    val_indexes = []
    for k in range(6):
        test = random.sample(idx_list, 12)
        test = sorted(test)
        val_indexes.append(test)
        for i in test:
            idx_list.remove(i)
    train_indexes = []
    for i in val_indexes:
        train = [x for x in idx_list1 if x not in i]
        train = sorted(train)
        train_indexes.append(train)
    for i in range(6):
        train_idx = train_indexes[i]
        val_idx = val_indexes[i]
        train_UNet(mode='P1+P28',train_data=train_data_pulled, val_data=val_data_pulled, init_chan=32, batch_size=2,
                   train_idx=train_idx, val_idx=val_idx, num=i)



