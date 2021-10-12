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
from UNet import *
from sklearn.metrics import precision_recall_curve, auc
import os
from datetime import datetime
torch.manual_seed(95)
np.random.seed(95)

def tversky(y_pred, y_true,beta):
    tp = (y_true*y_pred).sum()
    denom = ((y_true*y_pred) + (beta*y_true*(1-y_pred)) + ((1-beta)*(1-y_true)*y_pred)).sum()
    score = tp/denom
    return 1 - score

def F1(y_pred, y_true):
  y_pred = (y_pred > 0.5).float()
  tp = (y_true*y_pred).sum()
  tn = ((1 - y_true) * (1 - y_pred)).sum()
  fp = ((1 - y_true)*y_pred).sum()
  fn = (y_true*(1-y_pred)).sum()
  precision = tp/(tp + fp)
  recall = tp/(tp + fn)
  f1 = 2*(precision*recall)/(precision + recall)
  return f1


# def tversky_loss_func(true, logits, alpha, beta, eps=1e-7):
#     """Computes the Tversky loss [1].
#     Args:
#         true: a tensor of shape [B, H, W] or [B, 1, H, W].
#         logits: a tensor of shape [B, C, H, W]. Corresponds to
#             the raw output or logits of the model.
#         alpha: controls the penalty for false positives.
#         beta: controls the penalty for false negatives.
#         eps: added to the denominator for numerical stability.
#     Returns:
#         tversky_loss: the Tversky loss.
#     Notes:
#         alpha = beta = 0.5 => dice coeff
#         alpha = beta = 1 => tanimoto coeff
#         alpha + beta = 1 => F beta coeff
#     References:
#         [1]: https://arxiv.org/abs/1706.05721
#     """
#     num_classes = logits.shape[1]
#     if num_classes == 1:
#         true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
#         true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
#         true_1_hot_f = true_1_hot[:, 0:1, :, :]
#         true_1_hot_s = true_1_hot[:, 1:2, :, :]
#         true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
#         pos_prob = torch.sigmoid(logits)
#         neg_prob = 1 - pos_prob
#         probas = torch.cat([pos_prob, neg_prob], dim=1)
#     else:
#         true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
#         true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
#         probas = F.softmax(logits, dim=1)
#     true_1_hot = true_1_hot.type(logits.type())
#     dims = (0,) + tuple(range(2, true.ndimension()))
#     intersection = torch.sum(probas * true_1_hot, dims)
#     fps = torch.sum(probas * (1 - true_1_hot), dims)
#     fns = torch.sum((1 - probas) * true_1_hot, dims)
#     num = intersection
#     denom = intersection + (alpha * fps) + (beta * fns)
#     tversky_loss = (num / (denom + eps)).mean()
#     return (1 - tversky_loss)

def dice_loss(pred, target):
  preds_sq = pred**2
  target_sq = target**2
  return 1 - (2. * (torch.sum(pred * target))) / (preds_sq.sum() + target_sq.sum())


def train_UNet(train_data, val_data, init_chan, batch_size,train_idx,val_idx, num,name, lr=1e-4, n_epochs=400, patience = 30, tversky_loss = False, beta = 1):
    models_path = '/home/jlw351/deep_ensemble/models'
    lc_path = '/home/jlw351/deep_ensemble/learning_curves'
    if os.path.exists(os.path.join(models_path,name)) == False:
        os.mkdir(os.path.join(models_path,name))
        os.mkdir(os.path.join(lc_path,name))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loss_final_models  = []
    train_acc_final_models  = []
    val_loss_final_models  = []
    val_acc_final_models = []
    models =[]
    print("NEW MODEL!!!!!!!!!!!!!!!!!!!!")
    net = UNet(init_chan,1)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0)
    train = torch.utils.data.Subset(train_data,train_idx)
    val = torch.utils.data.Subset(val_data,val_idx)
    traindataload = DataLoader(train,batch_size=batch_size,shuffle=True)
    valdataload = DataLoader(val,batch_size=1,shuffle=False)
    best_score = 0
    counter = 0
    train_loss_final  = []
    train_acc_final  = []
    val_loss_final  = []
    val_acc_final = []
    for epoch in range(n_epochs):
      start = datetime.now()
      start_time = start.strftime('%H:%M:%S')
      print('Start time = ' + str(start_time))
      train_loss = []
      train_acc = []
      val_loss = []
      val_acc = []
      for step, (image,target) in enumerate(traindataload):
        if torch.sum(image) == 0:
          continue
        else:
          image = image.to(device)
          target = target.to(device)
          optimizer.zero_grad()
          pred = net.path(image.float())
          if tversky_loss:
            loss = tversky(pred,target,beta=beta)
          else:
            loss = dice_loss(pred,target)
          loss.backward()
          optimizer.step()
          acc = F1(pred,target)
          if math.isnan(acc.item()):
            continue
          else:
            train_loss.append(loss.item())
            train_acc.append(acc.item())
          print("Epoch [{}/{}], Step [{}/{}], training_loss:{:.4f}, training_acc:{:.4f}".format(epoch+1, n_epochs, step+1, len(traindataload), loss.item(), acc.item() ))
      with torch.no_grad():
        for idx, (val_image, val_target) in enumerate(valdataload):
          val_image = val_image.to(device)
          val_target = val_target.to(device)
          pred = net.path(val_image.float())
          if tversky_loss: 
            loss = tversky(pred,val_target,beta=beta)
          else:
            loss = dice_loss(pred,val_target)
          acc = F1(pred,val_target)
          if math.isnan(acc.item()):
            continue
          else:
            val_loss.append(loss.item())
            val_acc.append(acc.item())
      train_loss_final.append(np.mean(train_loss))
      train_acc_final.append(np.mean(train_acc))
      val_loss_final.append(np.mean(val_loss))
      val_acc_final.append(np.mean(val_acc))
      print('Epoch [{}/{}], TrLoss: {:.4f}, TrAcc: {:.4f}, VlLoss: {:.4f}, VlAcc: {:.4f}'.format(epoch+1, n_epochs, np.mean(train_loss),
                                                                                                np.mean(train_acc), np.mean(val_loss),np.mean(val_acc)))
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
        print("GOOD!!!")
        torch.save(net.state_dict(),'/home/jlw351/deep_ensemble/models/'+str(name)+'/best_model_'+str(num)+'_unet_initchan_'+str(init_chan)+'_batch_size'+str(batch_size)+'.pt')
        counter = 0
      end = datetime.now()
      end_time = end.strftime('%H:%M:%S')
      print('End time = ' + str(end_time))
    print("model added")
    train_loss_final_models.append(train_loss_final)
    train_acc_final_models.append(train_acc_final)
    val_loss_final_models.append(val_loss_final)
    val_acc_final_models.append(val_acc_final)
    print(best_score,'VAL LOSS')
    print(best_train,'TRAIN LOSS')

    fig = plt.figure(figsize=(10, 10))
    plt.plot(val_loss_final_models[0], label='Validation loss')
    plt.plot(train_loss_final_models[0], label='Training loss')
    plt.legend()
    plt.savefig('/home/jlw351/deep_ensemble/learning_curves/'+str(name)+'/learning_curve_unet_'+str(num)+'_unet_'+str(init_chan)+'_batch_size_'+str(batch_size)+'.pdf')
