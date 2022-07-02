
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import logging
import torch.nn as nn
from time import time
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score as kappa
from sklearn.metrics import confusion_matrix
from .util_metric import validate, validateALL


pp = Path("result") / "temp" / "split_metric_result"
pp.mkdir(parents=True, exist_ok=True)
pp = Path("result") / "training"
pp.mkdir(parents=True, exist_ok=True)
pp = Path("result") / "temp" / "split_confusion_result"
pp.mkdir(parents=True, exist_ok=True)


def train(net,
    train_loader,
    valid_loader,
    test_loader, 
    num_epochs=50,
    csv_name=f"result.csv",
    model_file="model.pth",
    save_model=False,
    batch_size=256):
    
    if num_epochs<1:
        print(f"wrong number epochs = {num_epochs} please put at least 1")
        breakpoint()

    print(f"Training for :  {Path(csv_name).stem}")
    start = time()
    net = net.cuda()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=0.0001,eps=1e-07) # lr=0.0001
    best_acc = 0
    count=0
    df = pd.DataFrame(columns=['epoch', 'train_acc', 'val_acc','time'])

    with tqdm(range(num_epochs), unit="epoch",leave=True) as tqdmEpoch:
        for e in tqdmEpoch:
            for i, sample in enumerate(tqdm(train_loader)):
                # zero the parameter gradients
                optimizer.zero_grad()
                for x in sample:
                    sample[x] = sample[x].to('cuda:0', non_blocking=True)
                outputs_is = net(sample)
                
                if type(outputs_is) is dict: # fusion
                    loss = loss_function(outputs_is["fusion"],sample["Target"].long()) # ! source d'erreur
                    for s_ in outputs_is.keys():
                        if s_ != "fusion":
                            loss += 0.15*loss_function(outputs_is[s_],sample["Target"].long())
                            loss += 0.15*loss_function(outputs_is[s_],outputs_is["fusion"].argmax(-1))
                else:
                    loss = loss_function(outputs_is,sample["Target"].long())
                
                loss.backward() # compute the total loss
                optimizer.step() # make the updates for each parameter
                
            end = time()
            
            val_acc, val_acc_classe = validate(net, valid_loader)
            train_acc, train_acc_classe = validate(net, train_loader)

            if val_acc > best_acc: # minimum 0 epoch to save best model.. 
                best_acc = val_acc
                if e > -1:
                    # compute accuracy and save models
                    test_acc, y_true, y_pred = validateALL(net, test_loader)
                    if save_model == True:
                        torch.save(net.state_dict(), model_file)
                    
                    if len(np.unique(y_pred))>9:
                        x_axis_labels = ['Sugarcane', 'Pasture and fodder', 'Market gardening','Grenhouse and shaded crops', 
                                        'Orchards','Wooded areas','Moor and Savannah','Rocks and natural bare soil',
                                        'Relief shadow','Water','Urbanized areas']
                    elif len(np.unique(y_pred)) == 4:    
                        x_axis_labels = ['Rice','Cotton','Water','Other']          
                    else:
                        x_axis_labels = ['Urbanized areas', 'Water', 'Forest', 'Moor', 'Orchards', 'Vineyards', 'Other crops']

                    cc = confusion_matrix(y_true,y_pred)/np.sum(confusion_matrix(y_true,y_pred),axis=1,keepdims=True)
                    np.save(str( Path("result") / "temp" / "split_confusion_result" /  f"Confusion_{Path(csv_name).stem}.npy"), cc)
                count=0
            else:
                count=count+1
                if count>10:
                    break

            new_row = {"epoch":e+1, "train_acc":train_acc, "val_acc":val_acc, "time (s)":end - start}
            df = df.append(new_row, ignore_index=True)

            if not e % 9:
                df.to_csv(path_or_buf="result/training/" + csv_name,index=False)
                if e>0:
                    optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/2
                    print(f"reducing lr to {optimizer.param_groups[0]['lr']}")

            tqdmEpoch.write(f"[{e+1:d}/{num_epochs:d}] Loss: {loss:.3f}  Train Acc: {train_acc:.3f}  Val Acc: {val_acc:.3f}")
            tqdmEpoch.update(1)

    df.to_csv(path_or_buf="result/training/" + csv_name,index=False)
    df2 = pd.DataFrame(columns=["test_acc", "test_kaa", "test_f1"])
    new_row = {"test_acc":test_acc, "test_kaa":kappa(y_true,y_pred), "test_f1":f1_score(y_true,y_pred,average='weighted')}
    df2 = df2.append(new_row, ignore_index=True)
    df2.to_csv(path_or_buf="result/temp/split_metric_result/test_" + csv_name,index=False)