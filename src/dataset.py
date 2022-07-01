# Imports
from re import S
import torch
import pandas as pd
# from sklearn.model_selection import train_test_split
import logging
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
loggerupds = logging.getLogger('update')
from pathlib import Path
import glob

def sampler_create(root: str,dataset: str,split):
    #Let there be 9 samples and 1 sample in class 0 and 1 respectively
    y_numpy_dir = Path(root) / "Ground_truth" / dataset /  f"Ground_truth_{dataset}_split_{split}.npy"
    Y_train = np.load(y_numpy_dir)
    num_target = len(np.unique(Y_train[:,1]))

    class_counts = [np.sum((Y_train[:,1]==x)) for x in range(num_target)]
    num_samples = Y_train.shape[0]
    labels = Y_train[:,1].tolist()
    class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
    class_weights = [np.sqrt(x) for x in class_weights]

    weights = [class_weights[labels[i]] for i in range(int(num_samples))]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
    return sampler

class DatasetS1S2VHSR_debug(BaseDataset):
    """Read numpy
        Used for training ..
    """
    def __init__(
            self,
            root: str,
            dataset: str,
            sensor: list, #todo
            split=0,
    ):
        if dataset not in ["Training","Validation","Test"]:
            print(f"Error training should be Training Validation Test not {dataset}")
            breakpoint()
        if len([item for item in sensor if item not in ["Spot","S2","S1"]])>0:
            print(f"Wrong sensor elements {sensor}")
            breakpoint()

        y_numpy_dir = Path(root) / "Ground_truth" / dataset /  f"Ground_truth_{dataset}_split_{split}.npy"
        self.Y_train = np.load(y_numpy_dir)
        self.size = self.Y_train.shape[0]
        self.sensor = sensor
        self.num_target = len(np.unique(self.Y_train[:,1]))

        
        if "S2" in sensor:
            X_train_S2 = np.load(Path(root) / "Sentinel-2" / dataset /  f"Sentinel-2_{dataset}_split_{split}.npy")
            center = int(X_train_S2.shape[1]/2)
            self.X_train_S2 = X_train_S2[:,center,center,:,:].transpose(2,1,0)

        if "S1" in sensor:
            X_train_S1 = np.load(Path(root) / "Sentinel-1" / dataset /  f"Sentinel-1_{dataset}_split_{split}.npy")
            X_train_S1 = np.concatenate((X_train_S1[:,:,:,:26], X_train_S1[:,:,:,31:57]), axis=3)
            self.X_train_S1 = X_train_S1.reshape(X_train_S1.shape[0],X_train_S1.shape[1],X_train_S1.shape[2],-1).transpose(0,3,1,2)

        if "Spot" in sensor:
            X_train_pan = np.load(Path(root) / "Spot-P" / dataset /  f"Spot-P_{dataset}_split_{split}.npy")
            try :
                self.X_train_pan = X_train_pan.transpose(0,3,1,2)
            except IndexError:
                print("index error")
                breakpoint()
            #self.X_train_pan = X_train_pan[:,:,:,0].transpose(0,3,1,2)
            X_train_ms = np.load(Path(root) / "Spot-MS" / dataset /  f"Spot-MS_{dataset}_split_{split}.npy")

            if X_train_ms.shape[3]==4:
                self.X_train_ms = X_train_ms.transpose(0,3,1,2)
            else:
                breakpoint()
                self.X_train_ms = X_train_ms[:,:,:,0,:].transpose(0,3,1,2)
            
    def __getitem__(self, i):
        dicts = {}
        dicts["Target"] = torch.as_tensor(np.array(self.Y_train[i,1]).astype('float32'))
        for x in self.sensor:
            if x == "S1":
                dicts[x] = torch.as_tensor(np.array(self.X_train_S1[i,:,:]).astype('float32'))
            elif x == "S2":
                dicts[x] = torch.as_tensor(np.array(self.X_train_S2[:,:,i]).astype('float32'))
            elif x == "Spot":
                dicts["PAN"] = torch.as_tensor(np.array(self.X_train_pan[i,:,:]).astype('float32'))
                dicts["MS"] = torch.as_tensor(np.array(self.X_train_ms[i,:,:]).astype('float32'))
        return dicts

    def __len__(self):
        return self.size

    def numtarget(self):
        return self.num_target

class DatasetS1S2VHSR(BaseDataset):
    """Read numpy
        Used for training ..
    """
    def __init__(
            self,
            root: str,
            dataset: str,
            sensor: list, #todo
            split=0,
    ):
        if dataset not in ["Training","Validation","Test"]:
            print(f"Error training should be Training Validation Test not {dataset}")
            breakpoint()
        if len([item for item in sensor if item not in ["Spot","S2","S1"]])>0:
            print(f"Wrong sensor elements {sensor}")
            breakpoint()

        y_numpy_dir = Path(root) / "Ground_truth" / dataset /  f"Ground_truth_{dataset}_split_{split}.npy"
        self.Y_train = np.load(y_numpy_dir)
        self.size = self.Y_train.shape[0]
        self.sensor = sensor
        self.num_target = len(np.unique(self.Y_train[:,1]))

        if "S2" in sensor:
            X_train_S2 = np.load(Path(root) / "Sentinel-2" / dataset /  f"Sentinel-2_{dataset}_split_{split}.npy")
            center = int(X_train_S2.shape[1]/2)
            self.X_train_S2 = X_train_S2[:,center,center,:,:].transpose(2,1,0)

        if "S1" in sensor:
            X_train_S1 = np.load(Path(root) / "Sentinel-1" / dataset /  f"Sentinel-1_{dataset}_split_{split}.npy")
            self.X_train_S1 = X_train_S1.reshape(X_train_S1.shape[0],X_train_S1.shape[1],X_train_S1.shape[2],-1).transpose(0,3,1,2)

        if "Spot" in sensor:
            X_train_pan = np.load(Path(root) / "Spot-P" / dataset /  f"Spot-P_{dataset}_split_{split}.npy")
            try :
                self.X_train_pan = X_train_pan.transpose(0,3,1,2)
            except IndexError:
                print("index error")
                breakpoint()
            #self.X_train_pan = X_train_pan[:,:,:,0].transpose(0,3,1,2)
            X_train_ms = np.load(Path(root) / "Spot-MS" / dataset /  f"Spot-MS_{dataset}_split_{split}.npy")

            if X_train_ms.shape[3]==4:
                self.X_train_ms = X_train_ms.transpose(0,3,1,2)
            else:
                breakpoint()
                self.X_train_ms = X_train_ms[:,:,:,0,:].transpose(0,3,1,2)
            
    def __getitem__(self, i):
        dicts = {}
        dicts["Target"] = torch.as_tensor(np.array(self.Y_train[i,1]).astype('float32'))
        for x in self.sensor:
            if x == "S1":
                dicts[x] = torch.as_tensor(np.array(self.X_train_S1[i,:,:]).astype('float32'))
            elif x == "S2":
                dicts[x] = torch.as_tensor(np.array(self.X_train_S2[:,:,i]).astype('float32'))
                
            elif x == "Spot":
                dicts["PAN"] = torch.as_tensor(np.array(self.X_train_pan[i,:,:]).astype('float32'))
                print('PAN')
                print(dicts["PAN"].shape)
                dicts["MS"] = torch.as_tensor(np.array(self.X_train_ms[i,:,:]).astype('float32'))
                print('MS')
                print(dicts["MS"].shape)
        return dicts

    def __len__(self):
        return self.size

    def numtarget(self):
        return self.num_target


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
    ):
        if dataset not in ["Training","Validation","Test"]:
            print(f"Error training should be Training Validation Test not {dataset}")
            breakpoint()
        if len([item for item in sensor if item not in ["Spot","S2","S1"]])>0:
            print(f"Wrong sensor elements {sensor}")
            breakpoint()

        self.root = root
        self.dataset = dataset
        self.list_y_numpy_dir = glob.glob( str(Path(root) / "Ground_truth" / dataset /  f"Ground_truth_{dataset}_*.npy"))
        self.sensor = sensor
        if "reunion" in Path(root).stem:
            self.num_target = 11
        elif "dordogne" in Path(root).stem:
            self.num_target = 7
        else:
            print("unknown dataset")

        self.size = 256*(len(self.list_y_numpy_dir)-1)  + np.load(self.list_y_numpy_dir[len(self.list_y_numpy_dir)-1]).shape[0]

        # for i in range(0,self.size):
        #     dicts = {}
        #     idnpy = i//256
        #     idx = i - (i//256)*256
            
        #     Y_train = np.load(Path(self.root) / "Ground_truth" / self.dataset /  f"Ground_truth_{self.dataset}_split_{idnpy}.npy")[idx]
        #     if Y_train[1]==-1:
        #         breakpoint()
        #     dicts["Target"] = torch.as_tensor(np.array(Y_train[1]).astype('float32'))

        #     for x in self.sensor:
        #         if x == "S1":
        #             X_train_S1 = np.load(Path(self.root) / "Sentinel-1" / self.dataset /  f"Sentinel-1_{self.dataset}_split_{idnpy}.npy")[idx].transpose(2,0,1)
        #             dicts[x] = torch.as_tensor(np.array(X_train_S1).astype('float32'))
        #         elif x == "S2":
        #             X_train_S2 = np.load(Path(self.root) / "Sentinel-2" / self.dataset /  f"Sentinel-2_{self.dataset}_split_{idnpy}.npy")[idx]
        #             center = int(X_train_S2.shape[1]/2)
        #             dicts[x] = torch.as_tensor(np.array( X_train_S2[center,center,:].reshape(21,6).transpose(1,0)).astype('float32'))

        #         elif x == "Spot":
        #             breakpoint()
        #             X_train_Spot = np.load(Path(self.root) / "Spot-P" / self.dataset /  f"Spot-P_{self.dataset}_split_{idnpy}.npy")[idx].transpose(2,0,1)[:,:-1,:-1]
        #             dicts["PAN"] = torch.as_tensor(np.array(X_train_Spot).astype('float32'))
        #             X_train_Spot = np.load(Path(self.root) / "Spot-MS" / self.dataset /  f"Spot-MS_{self.dataset}_split_{idnpy}.npy")[idx].transpose(2,0,1)[:,:-1,:-1]
        #             dicts["MS"] = torch.as_tensor(np.array(X_train_Spot).astype('float32'))

    def __getitem__(self, i):

        dicts = {}
        idnpy = i//256
        idx = i - (i//256)*256

        Y_train = np.load(Path(self.root) / "Ground_truth" / self.dataset /  f"Ground_truth_{self.dataset}_split_{idnpy}.npy")[idx]
        
        
        if Y_train[1]==-1:
            print("Bad value for y train")
            dicts["Target"] = torch.as_tensor(np.array(0).astype('float32'))
        else:
            dicts["Target"] = torch.as_tensor(np.array(Y_train[1]).astype('float32'))

        
        
        for x in self.sensor:
            if x == "S1":
                X_train_S1 = np.load(Path(self.root) / "Sentinel-1" / self.dataset /  f"Sentinel-1_{self.dataset}_split_{idnpy}.npy")[idx].transpose(2,0,1)
                dicts[x] = torch.as_tensor(np.array(X_train_S1).astype('float32'))
            elif x == "S2":
                X_train_S2 = np.load(Path(self.root) / "Sentinel-2" / self.dataset /  f"Sentinel-2_{self.dataset}_split_{idnpy}.npy")[idx]
                center = int(X_train_S2.shape[1]/2)
                # X_train_S2 = np.expand_dims(X_train_S2[center,center,:], axis=0)
                dicts[x] = torch.as_tensor(np.array( X_train_S2[center,center,:].reshape(21,6).transpose(1,0)).astype('float32'))

            elif x == "Spot":
                X_train_Spot = np.load(Path(self.root) / "Spot-P" / self.dataset /  f"Spot-P_{self.dataset}_split_{idnpy}.npy")[idx].transpose(2,0,1)[:,:-1,:-1]
                dicts["PAN"] = torch.as_tensor(np.array(X_train_Spot).astype('float32'))
                X_train_Spot = np.load(Path(self.root) / "Spot-MS" / self.dataset /  f"Spot-MS_{self.dataset}_split_{idnpy}.npy")[idx].transpose(2,0,1)[:,:-1,:-1]
                dicts["MS"] = torch.as_tensor(np.array(X_train_Spot).astype('float32'))
        return dicts

    def __len__(self):
        return self.size

    def numtarget(self):
        return self.num_target