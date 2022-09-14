# Imports
from re import S
import torch
import pandas as pd
from torch.utils.data import Dataset as BaseDataset
import numpy as np
from pathlib import Path
import glob
from bisect import bisect
import glob
import random

class DatasetS1S2VHSRbig(BaseDataset):
    """Read mutiples numpy
        Used for training ..
        a solution is : https://stackoverflow.com/questions/60127632/load-multiple-npy-files-size-10gb-in-pytorch
        inspired from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """
    def __init__(
            self,
            root: str,
            dataset: str,
            sensor: list, #todo
            num_target = None,
            split = 0,
    ):

        #déclarer ici
        # tread qui load et convertit en tensor torch (miner buffer)
        #consumer buffer recupere ce qu'il y a dans le miner buffer
        if dataset not in ["Training","Validation","Test"]:
            print(f"Error training should be Training Validation Test not {dataset}")
            breakpoint()
        if len([item for item in sensor if item not in ["Spot","S2","S1"]])>0:
            print(f"Wrong sensor elements {sensor}")
            breakpoint()

        self.root = root
        self.dataset = dataset
        list_y_numpy_dir = glob.glob( str(Path(root) / "Ground_truth" / dataset  /  f"Ground_truth_{dataset}_*.npy"))
        self.sensor = sensor

        # change the k number for more samples ...
        self.list_y_numpy_dir = list_y_numpy_dir
        # self.list_y_numpy_dir = random.choices(list_y_numpy_dir, k=500)

        if num_target is None:
            if "reunion" in Path(root).stem:
                self.num_target = 11
            elif "dordogne" in Path(root).stem:
                self.num_target = 7
            else:
                self.num_target = 4
                print("unknown dataset number of class so num_target is set to 4")

        dic_path = {}
        for x in self.sensor:
            if x == "S1":
                print(f"reading {dataset} S1 in mmap memory")
                dic_path["S1"]= [xx.replace("Ground_truth","Sentinel-1") for xx in self.list_y_numpy_dir]
                self.data_memmaps_S1 = [np.load(path, mmap_mode='r') for path in dic_path["S1"]]
                print(f"converting {dataset} S1 mmap in torch tensor")
                self.data_memmaps_S1 = [torch.as_tensor(np.array(x_.transpose(0,-1,1,2)).astype('float32')) for x_ in self.data_memmaps_S1]

            if x == "S2":
                print(f"reading {dataset} S2 in mmap memory")
                dic_path["S2"]= [xx.replace("Ground_truth","Sentinel-2") for xx in self.list_y_numpy_dir]
                # dic_path["S2"]= glob.glob( str(Path(root) / "Sentinel-2" / dataset /  f"Sentinel-2_{dataset}_*.npy"))
                self.data_memmaps_S2 = [np.load(path, mmap_mode='r') for path in dic_path["S2"]]
                center = int(self.data_memmaps_S2[0].shape[1]/2)
                self.data_memmaps_S2 = [x_[:,center,center,:] for x_ in self.data_memmaps_S2]
                self.data_memmaps_S2 = [torch.as_tensor(np.array(x_.reshape(x_.shape[0],int(x_.shape[1]/6),6).transpose(0,-1,1)).astype('float32')) for x_ in self.data_memmaps_S2]

            if x == "Spot":
                print(f"reading {dataset} Spot in mmap memory")
                dic_path["PAN"]= [xx.replace("Ground_truth","Spot-P") for xx in self.list_y_numpy_dir]
                self.data_memmaps_PAN = [np.load(path, mmap_mode='r') for path in dic_path["PAN"]]
                self.data_memmaps_PAN = [torch.as_tensor(np.array(x_.transpose(0,-1,1,2)[:,:,:-1,:-1]).astype('float32'))  for x_ in self.data_memmaps_PAN]

                dic_path["MS"]= [xx.replace("Ground_truth","Spot-MS") for xx in self.list_y_numpy_dir]
                self.data_memmaps_MS = [np.load(path, mmap_mode='r') for path in dic_path["MS"]]
                self.data_memmaps_MS = [torch.as_tensor(np.array(x_.transpose(0,-1,1,2)[:,:,:-1,:-1]).astype('float32')) for x_ in self.data_memmaps_MS]

        print(f"reading {dataset} Ground truth in mmap memory")
        self.target_memmaps = [np.load(path, mmap_mode='r') for path in self.list_y_numpy_dir]

        self.start_indices = [0] * len(self.list_y_numpy_dir)
        self.data_count = 0
        for index, memmap in enumerate(self.target_memmaps):
            self.start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]


    def __getitem__(self, index):
        # https://gitlab.irstea.fr/remi.cresson/otbtf/-/blob/develop/otbtf/dataset.py
        #regarder le def read_one_sample(self):
        # tread qui load et convertu en tensor torch (miner buffer)
        #consumer buffer recupere ce qu'il y a dans le miner buffer
        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]
        target = self.target_memmaps[memmap_index][index_in_memmap]
        dicts = {}

        for x in self.sensor:
            if x == "S1":
                dicts[x] = self.data_memmaps_S1[memmap_index][index_in_memmap]
            elif x == "S2":
                dicts[x] = self.data_memmaps_S2[memmap_index][index_in_memmap]
            elif x == "Spot":
                dicts["PAN"] = self.data_memmaps_PAN[memmap_index][index_in_memmap]
                dicts["MS"] = self.data_memmaps_MS[memmap_index][index_in_memmap]

        if target[1]==-1:
            dicts["Target"] = torch.as_tensor(np.array(0))  
        else:
            dicts["Target"] = torch.as_tensor(np.array(target[1]))  

        return dicts

    def __len__(self):
        return self.data_count

    def numtarget(self):
        return self.num_target