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

test = MyDataset('test','*P*',0,False,False,False,0.5)
test_loader = DataLoader(test,batch_size=4,shuffle=False)
test_im, test_mask = next(iter(test_loader))

m_P1 = glob.glob('/home/Jan/Insitu/UNet_models/mode_P1*.pt')
m_P28 = glob.glob('/home/Jan/Insitu/UNet_models/mode_P28*.pt')
m_pulled = glob.glob('/home/Jan/Insitu/UNet_models/mode_P1+P28*.pt')

models = [m_P1,m_P28,m_pulled]

ensemble_predictions = []
for i in range(len(models)):
    for j in range(len(models[i])):
        model = UNet(32,1)
        model.load_state_dict(torch.load(models[i][j]))
        model = model.cuda()
        with torch.no_grad():
            pred = model.path(test_im)
        if j == 0:
            comb_pred = pred
        else:
            comb_pred = comb_pred + pred
    ensemble_predictions.append(comb_pred)

thresholds = [0,0.25,0.5,0.75,1]

thr_ensemble_predictions = []
for i in range(len(ensemble_predictions)):
    for j in thresholds:
        if j == 1:
            thr_prediction = (ensemble_prediction[i] >= j).float()
        else:
            thr_prediction = (ensemble_prediction[i] > j).float()
        thr_ensemble_predictions.append(thr_prediction)

for i in range(len(thr_ensemble_predictions)):
    if i == 0:
        final_ensemble_prediction = thr_ensemble_predictions[i]
    else:
        final_ensemble_prediction = final_ensemble_prediction + thr_ensemble_predictions[i]

lowest_thresh = 1/15
final_ensemble_prediction = final_ensemble_prediction/15
final_ensemble_prediction_thr = (final_ensemble_prediction > lowest_thresh).float()


F1_test_ensemble = []
for i in range(4):
    F1_test_ensemble.append(F1(final_ensemble_prediction_thr[i,0].detach().numpy(),test_mask[i,0].detach().numpy()))


print('Ensemble mean F1 score: %f'%(np.mean(F1_test_ensemble)))
print('Ensemble std F1 score: %f'%(np.std(F1_test_ensemble)))

#### Baseline U-net calculation
unet_prediction_prauc = []
unet_predictions = []
for i in range(len(m_pulled)):
    model = UNet(32, 1)
    model.load_state_dict(torch.load(m_pulled[i]))
    model = model.cuda()
    with torch.no_grad():
        pred = model.path(test_im)
        pred1 = (pred > 0.5).float()
    unet_predictions_prauc.append(pred)
    unet_predictions.append(pred1)

F1_for_models = []
for i in range(len(unet_predictions)):
    temp = []
    for j in range(4):
        temp.append(F1(unet_predictions[i][j,0].detach().numpy(),test_mask[j,0].detach().numpy()))
    F1_for_models.append(np.mean(temp))

print('Baseline mean F1 score: %f'%(np.mean(F1_for_models)))
print('Baseline std F1 score: %f'%(np.std(F1_for_models)))


#### PRAUC for ensemble


prauc_ensemble = []
for i in range(4):
    precision, recall, thr = precision_recall_curve(test_mask[i,0].detach().numpy().flatten(),final_ensemble_prediction[i,0].detach().numpy().flatten())
    prauc = auc(recall,precision)
    prauc_ensemble.append(prauc)

print('Ensemble PRAUC: %f' % (np.mean(prauc_ensemble)))

#### PRAUC for baseline u-net

prauc_unet = []
for i in range(len(unet_predictions_prauc)):
    temp = []
    for j in range(4):
        precision, recall, thr = precision_recall_curve(test_mask[j,0].detach().numpy().flatten(),unet_predictions_prauc[i][j,0].detach().numpy().flatten())
        prauc = auc(recall,precision)
        temp.append(prauc)
    prauc_unet.append(np.mean(temp))

print('Ensemble PRAUC: %f' % (np.mean(prauc_unet)))
