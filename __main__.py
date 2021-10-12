import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import random
from random import uniform
import glob
import torch
import math
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
import torchvision.transforms.functional as TF
from skimage import io,transform,filters,exposure
from Dataset import MyDataset
from sklearn import metrics
from UNet import UNet
import pdb
from training import *
torch.manual_seed(95)
np.random.seed(95)


P1_train_data = MyDataset('training','*P1*',0,True,True,True,0.5)
P1_val_data = MyDataset('training','*P1*',0,False,False,False,0.5)

P28_train_data = MyDataset('training','*P28*',0,True,True,True,0.5)
P28_val_data = MyDataset('training','*P28*',0,False,False,False,0.5)

# We now train the ensemble candidates with training mode consisting of P1 slices

idx_list = [i for i in range(52)]
idx_list1 = [i for i in range(52)]
val_indexes = []
for k in range(4): # I did 4-fold CV because it allowed for equal division of the dataset
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
    train_UNet(train_data=P1_train_data, val_data=P1_val_data, init_chan=32, batch_size=4, train_idx=train_idx,
               val_idx=val_idx, num=i, name='P1/best_model', lr=1e-5, n_epochs=500, patience=100,
               tversky_loss=False, beta=beta_range[i])


# Now we train the ensemble candidates with training mode consisting of P28 slices



idx_list = [i for i in range(20)]
idx_list1 = [i for i in range(20)]
val_indexes = []
for k in range(5): # I did 5-fold CV because it allowed for equal division of the dataset
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
    train_UNet(train_data=P28_train_data, val_data=P28_val_data, init_chan=32, batch_size=4, train_idx=train_idx,
               val_idx=val_idx, num=i, name='P28/best_model', lr=1e-5, n_epochs=500, patience=100,
               tversky_loss=False, beta=beta_range[i])


test = MyDataset('test','*P*',0,False,False,False,0.5)
test_loader = DataLoader(test,batch_size=4,shuffle=False)
test_img, test_mask = next(iter(test_loader))


GPU = False
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# m_P1 = glob.glob('/home/janek/cluster/deep_ensemble/models/P1/*.pt')
# m_P28 = glob.glob('/home/janek/cluster/deep_ensemble/models/P28/*.pt')

m_P1 = glob.glob('/home/janek/erda/personal/ensemble/models/P1/*.pt')
m_P28 = glob.glob('/home/janek/erda/personal/ensemble/models/P28/*.pt')

models = [m_P1,m_P28]

ensemble_predictions = []
for i in range(len(models)):
    for j in range(len(models[i])):
        model = UNet(32,1)
        if GPU:
            model.load_state_dict(torch.load(models[i][j]))
            model = model.to(device)
        else:
            model.load_state_dict(torch.load(models[i][j],map_location=torch.device('cpu')))
            model = model.to(device)
        with torch.no_grad():
            test_img = test_img.to(device)
            pred = model.path(test_img.float())
            pred = (pred > 0.5).float()
            ensemble_predictions.append(pred)

final_ensemble = sum(ensemble_predictions)/len(ensemble_predictions)

final_ensemble_thr = (final_ensemble > 0).float()

F1_scores_deep_ensemble = [F1(final_ensemble_thr[i,0].cpu(),test_mask[i,0].cpu()) for i in range(4)]
print(np.mean(F1_scores_deep_ensemble))