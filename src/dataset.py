# Imports
import torch
import pandas as pd
# from sklearn.model_selection import train_test_split
import logging
from torch.utils.data import Dataset as BaseDataset
import numpy as np
loggerupds = logging.getLogger('update')

class DatasetSpot(BaseDataset):
    """Read numpy
    .. can be used for augmentation ..
    """
    def __init__(
            self,
            x_pan_numpy_dir,
            x_ms_numpy_dir,
            y_numpy_dir,
            # we can add the authors sources of information here
            classes=None, #todo

    ):
        self.classes = classes

        self.Y_train = np.load(y_numpy_dir)
        self.size=len(self.Y_train)
        self.X_train_pan = np.load(x_pan_numpy_dir)
        self.X_train_ms = np.load(x_ms_numpy_dir)


        
    def __getitem__(self, i):
        return self.X_train_pan[i,:,:,0].reshape([1,32,32]).astype("float32"), \
            self.X_train_ms[i,:,:,:,0].reshape([4,8,8]).astype("float32"), self.Y_train[i,1]-1

    def __len__(self):
        return self.size



class DatasetS2(BaseDataset):
    """Read numpy
    .. can be used for augmentation ..
    """
    def __init__(
            self,
            x_S2_numpy_dir,
            y_numpy_dir,
            # we can add the authors sources of information here
            classes=None, #todo

    ):
        self.classes = classes

        self.Y_train = np.load(y_numpy_dir)
        self.size=len(self.Y_train)
        self.X_train_S2 = np.load(x_S2_numpy_dir)
        
        
    def __getitem__(self, i):
        center = int(self.X_train_S2.shape[1]/2)
        return self.X_train_S2[i,center,center,:,:].reshape([6,21]).astype("float32"), self.Y_train[i,1]-1

    def __len__(self):
        return self.size
