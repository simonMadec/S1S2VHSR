import torch
import numpy as np
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import os
import sys
from torch.utils.data import DataLoader
# from torch.utils.data import Dataset as BaseDataset
import logging
import torch.nn as nn
from time import time
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score as kappa
os.environ["CUDA_VISIBLE_DEVICES"] ="2"


from dataset import DatasetSpot

split = 0

path_Y_train = f"/mnt/DATA/JE/data_reunion/Ground_truth/Training/Ground_truth_Training_split_{split}.npy"
path_X_train_pan = f"/mnt/DATA/JE/data_reunion/Spot-P/Training/Spot-P_Training_split_{split}.npy"
path_X_train_ms = f"/mnt/DATA/JE/data_reunion/Spot-MS/Training/Spot-MS_Training_split_{split}.npy"

path_Y_valid = f"/mnt/DATA/JE/data_reunion/Ground_truth/Validation/Ground_truth_Validation_split_{split}.npy"
path_X_valid_pan = f"/mnt/DATA/JE/data_reunion/Spot-P/Validation/Spot-P_Validation_split_{split}.npy"
path_X_valid_ms = f"/mnt/DATA/JE/data_reunion/Spot-MS/Validation/Spot-MS_Validation_split_{split}.npy"

path_Y_test = f"/mnt/DATA/JE/data_reunion/Ground_truth/Test/Ground_truth_Test_split_{split}.npy"
path_X_test_pan = f"/mnt/DATA/JE/data_reunion/Spot-P/Test/Spot-P_Test_split_{split}.npy"
path_X_test_ms = f"/mnt/DATA/JE/data_reunion/Spot-MS/Test/Spot-MS_Test_split_{split}.npy"

# fig = plt.figure()
# plt.hist(train[:,2],density=1, bins=1000)
# plt.savefig("dummy_name_pan.png")

train_dataset = DatasetSpot(x_pan_numpy_dir=path_X_train_pan,x_ms_numpy_dir=path_X_train_ms,y_numpy_dir=path_Y_train)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=10)

valid_dataset = DatasetSpot(x_pan_numpy_dir=path_X_valid_pan,x_ms_numpy_dir=path_X_valid_ms,y_numpy_dir=path_Y_valid)
valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=True, num_workers=10)

test_dataset = DatasetSpot(x_pan_numpy_dir=path_X_test_pan,x_ms_numpy_dir=path_X_test_ms,y_numpy_dir=path_Y_test)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=10)


def validateALL(net, loader):
    y_pred = []
    y_true = []
    net = net.cuda()
    net.eval()
    with torch.no_grad():
        acc = []
        f1 = []
        ka = []
        for i, sample in enumerate(loader):
            PAN, MS, Target = sample[0].to(torch.float32), sample[1].to(torch.float32), sample[2].to(torch.float32) #torch.long todo change 0 1 2 to MS PAN Target see hao lu codes
            PAN, MS, Target = PAN.cuda(), MS.cuda(), Target.cuda() 

            # generate targets
            outputs_is = net(PAN, MS)
            correct_pred = (outputs_is.argmax(-1) == Target).float()
            val = correct_pred.sum() / len(correct_pred)

            f1.append(f1_score(outputs_is.argmax(-1).cpu().numpy(), Target.cpu().numpy(),average='micro'))
            ka.append(kappa(outputs_is.argmax(-1).cpu().numpy(), Target.cpu().numpy()))
            acc.append(val.cpu().numpy())
             #faire une accuracy

    return np.mean(np.asarray(acc)),np.mean(np.asarray(ka)),np.mean(np.asarray(f1))

def validate(net, loader):
    y_pred = []
    y_true = []
    net = net.cuda()
    net.eval()
    with torch.no_grad():
        acc = []
        for i, sample in enumerate(loader):
            PAN, MS, Target = sample[0].to(torch.float32), sample[1].to(torch.float32), sample[2].to(torch.float32) #torch.long todo change 0 1 2 to MS PAN Target see hao lu codes
            PAN, MS, Target = PAN.cuda(), MS.cuda(), Target.cuda() 

            # generate targets
            outputs_is = net(PAN, MS)
            correct_pred = (outputs_is.argmax(-1) == Target).float()
            val = correct_pred.sum() / len(correct_pred)
            acc.append(val.cpu().numpy())
             #faire une accuracy
    return np.mean(np.asarray(acc))


def train(net,train_loader, valid_loader,test_loader, num_epochs=50,csv_name=f"result.csv"):
    start = time()
    net = net.cuda()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=0.0001)
    best_f1 = 0
    best_metrics = []

    df = pd.DataFrame(columns=['epoch', 'train_acc', 'val_acc','time'])

    with tqdm(range(num_epochs), unit="epoch",leave=False) as tqdmEpoch:
        for e in tqdmEpoch:

            for i, sample in enumerate(train_loader):
                # zero the parameter gradients
                optimizer.zero_grad()

                PAN, MS, Target = sample[0].to(torch.float32), sample[1].to(torch.float32), sample[2].to(torch.float32) #torch.long todo change 0 1 2 to MS PAN Target see hao lu codes
                PAN, MS, Target = PAN.cuda(), MS.cuda(), Target.cuda() 
                
                # generate targets
                outputs_is = net(PAN, MS)

                # OkTarget = np.zeros([256,11])
                # for it,classe in enumerate(Target.cpu().numpy()):
                loss = loss_function(outputs_is,Target.long()) # ! source d'erreur
                loss.backward() # compute the total loss
                optimizer.step() # make the updates for each parameter
                
            end = time()

            val_acc = validate(net, valid_loader)
            train_acc = validate(net, train_loader)

            new_row = {"epoch":e+1, "train_acc":train_acc, "val_acc":val_acc, "time (s)":end - start}
            df = df.append(new_row, ignore_index=True)
            tqdmEpoch.set_description(f"[{e+1:d}/{num_epochs:d}] Loss: {loss:.3f}  Train Acc: {train_acc:.3f}  Val Acc: {val_acc:.3f}", refresh=False)

    test_acc,test_kaa,test_f1 = validateALL(net, test_loader)

    df.to_csv(path_or_buf=csv_name,index=False)

    df2 = pd.DataFrame(columns=["test_acc", "test_kaa", "test_f1"])
    new_row = {"test_acc":test_acc, "test_kaa":test_kaa, "test_f1":test_f1}

    df2 = df2.append(new_row, ignore_index=True)
    df2.to_csv(path_or_buf="test_" + csv_name,index=False)
    

train(Model_SPOT(n_classes=11),train_loader,valid_loader,test_loader)








