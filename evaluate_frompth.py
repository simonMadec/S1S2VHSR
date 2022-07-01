import torch
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
from sklearn.metrics import confusion_matrix
from src.util_metric import validate, validateALL
from src.datasetbig import  DatasetS1S2VHSRbig2
from torch.utils.data import DataLoader
from src.MS import Model_MultiSource # import the model
from src.util_study import merge_csv, merge_result

os.environ["CUDA_VISIBLE_DEVICES"] ="2"


method = "v2Epoch5"

batch_size = 256
# for sensor in [["S1"],["S2"],["Spot"],["S1","S2"],["S2","Spot"],["S1","S2","Spot"]]:
for site in ["data_dordogne_origin_out150cmGSD_v2","data_reunion_origin_out150cmGSD_v2"]:
    root = f"/home/simon/DATA/land_use_classification/data/{site}"
    for sensor in [["Spot","S1","S2"]]:  
        print("hello")
        model_file = f"{method}_{'-'.join(sensor)}_site-{site}.pth"

        csv_name = f"{method}_{'-'.join(sensor)}_site-{site}_result.csv"

        merge_csv(csv_name)
        print("hello")
        merge_result(csv_name)













method = "v2Epoch5"

batch_size = 256
# for sensor in [["S1"],["S2"],["Spot"],["S1","S2"],["S2","Spot"],["S1","S2","Spot"]]:
for site in ["data_dordogne_origin_out150cmGSD_v2","data_reunion_origin_out150cmGSD_v2"]:
    root = f"/home/simon/DATA/land_use_classification/data/{site}"
    for sensor in [["Spot","S1","S2"]]:  
        test_dataset = DatasetS1S2VHSRbig2(root=root,dataset="Test",sensor=sensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,pin_memory=True, num_workers=1)
        print("hello")
        model_file = f"{method}_{'-'.join(sensor)}_site-{site}.pth"
        net = Model_MultiSource(n_classes=test_dataset.numtarget(),sensor=sensor,auxloss=True)
        net = net.cuda()
        breakpoint()
        # net = torch.load(net.state_dict(), model_file)
        net.load_state_dict(torch.load(model_file))
        net.eval()
        df2 = pd.DataFrame(columns=["test_acc", "test_kaa", "test_f1"])
        test_acc, y_true, y_pred = validateALL(net, test_loader)
        csv_name = f"{method}_{'-'.join(sensor)}_site-{site}_result.csv"
        new_row = {"test_acc":test_acc, "test_kaa":kappa(y_true,y_pred), "test_f1":f1_score(y_true,y_pred,average='weighted')}
        df2.to_csv(path_or_buf="result/temp/split_metric_result/test_" + csv_name,index=False)

        merge_csv(csv_name)
        print("hello")
        merge_result(csv_name)