
import torch
#from spot import *
import numpy as np
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import os
import sys
#from torch.utils.data import DataLoader
# from torch.utils.data import Dataset as BaseDataset
import logging
import torch.nn as nn
from time import time
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score as kappa


def validateALL(net, loader,test=True):
    # compute all metrics
    y_pred = []
    y_true = []
    net = net.cuda()
    net.eval()
    with torch.no_grad():
        acc = []
        f1 = []
        ka = []
        for i, sample in enumerate(loader):

            if len(sample)==2:
                #c'est pas spot #todo
                S2, Target = sample[0].to(torch.float32), sample[1].to(torch.float32) #torch.long todo change 0 1 2 to MS PAN Target avec S2 S1.. see hao lu codes
                if test==True:
                    Target=Target+1
                S2, Target = S2.cuda(), Target.cuda() 
                # generate targets
                outputs_is = net(S2)
            else:
                PAN, MS, Target = sample[0].to(torch.float32), sample[1].to(torch.float32), sample[2].to(torch.float32) #torch.long todo change 0 1 2 to MS PAN Target avec S2 S1.. see hao lu codes
                if test==True:
                    Target=Target+1
                PAN, MS, Target = PAN.cuda(), MS.cuda(), Target.cuda() 
                outputs_is = net(PAN, MS)

            # generate targets
            correct_pred = (outputs_is.argmax(-1) == Target).float()
            val = correct_pred.sum() / len(correct_pred)

            f1.append(f1_score(outputs_is.argmax(-1).cpu().numpy(), Target.cpu().numpy(),average='micro'))
            ka.append(kappa(outputs_is.argmax(-1).cpu().numpy(), Target.cpu().numpy()))
            acc.append(val.cpu().numpy())
             #faire une accuracy
    return np.mean(np.asarray(acc)),np.mean(np.asarray(ka)),np.mean(np.asarray(f1))


def validate(net, loader):
    #compute just accuracy
    y_pred = []
    y_true = []
    net = net.cuda()
    net.eval()
    with torch.no_grad():
        acc = []
        for i, sample in enumerate(loader):

            if len(sample)==2:
                #c'est pas spot #todo
                S2, Target = sample[0].to(torch.float32), sample[1].to(torch.float32) #torch.long todo change 0 1 2 to MS PAN Target avec S2 S1.. see hao lu codes
                S2, Target = S2.cuda(), Target.cuda() 
                # generate targets
                outputs_is = net(S2)
            else:
                PAN, MS, Target = sample[0].to(torch.float32), sample[1].to(torch.float32), sample[2].to(torch.float32) #torch.long todo change 0 1 2 to MS PAN Target avec S2 S1.. see hao lu codes
                PAN, MS, Target = PAN.cuda(), MS.cuda(), Target.cuda() 
                outputs_is = net(PAN, MS)


            correct_pred = (outputs_is.argmax(-1) == Target).float()
            val = correct_pred.sum() / len(correct_pred)
            acc.append(val.cpu().numpy())
             #faire une accuracy
    return np.mean(np.asarray(acc))


def train(net,train_loader, valid_loader,test_loader, num_epochs=50,csv_name=f"result.csv",model_file="model-split"):
    start = time()
    net = net.cuda()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=0.0001)
    best_acc = 0
    count=0

    df = pd.DataFrame(columns=['epoch', 'train_acc', 'val_acc','time'])

    with tqdm(range(num_epochs), unit="epoch",leave=False) as tqdmEpoch:
        for e in tqdmEpoch:

            for i, sample in enumerate(train_loader):
                # zero the parameter gradients
                optimizer.zero_grad()

                if len(sample)==2:
                    #c'est pas spot #todo
                    S2, Target = sample[0].to(torch.float32), sample[1].to(torch.float32) #torch.long todo change 0 1 2 to MS PAN Target avec S2 S1.. see hao lu codes
                    S2, Target = S2.cuda(), Target.cuda() 
                    # generate targets
                    outputs_is = net(S2)
                else:
                    PAN, MS, Target = sample[0].to(torch.float32), sample[1].to(torch.float32), sample[2].to(torch.float32) #torch.long todo change 0 1 2 to MS PAN Target avec S2 S1.. see hao lu codes
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
            if val_acc > best_acc: # minimum 5 epoch to save best model..
                print("saving models .. ")
                best_acc = val_acc
                torch.save(net.state_dict(), model_file)
                # compute accuracy ..
                test_acc,test_kaa,test_f1 = validateALL(net, test_loader)
                count=0
            else:
                #for early stopping
                count=count+1
                if count>10:
                    break

            new_row = {"epoch":e+1, "train_acc":train_acc, "val_acc":val_acc, "time (s)":end - start}
            df = df.append(new_row, ignore_index=True)

            if not e % 10:
                df.to_csv(path_or_buf=csv_name,index=False)

            tqdmEpoch.set_description(f"[{e+1:d}/{num_epochs:d}] Loss: {loss:.3f}  Train Acc: {train_acc:.3f}  Val Acc: {val_acc:.3f}", refresh=False)

    
    df.to_csv(path_or_buf=csv_name,index=False)

    df2 = pd.DataFrame(columns=["test_acc", "test_kaa", "test_f1"])
    new_row = {"test_acc":test_acc, "test_kaa":test_kaa, "test_f1":test_f1}

    df2 = df2.append(new_row, ignore_index=True)
    df2.to_csv(path_or_buf="test_" + csv_name,index=False)