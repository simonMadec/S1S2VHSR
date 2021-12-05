
import glob
import numpy as np
import matplotlib.pyplot as plt 
import glob
from pathlib import Path

site = "data_dordogne"

for split in range(0,5):
    for pathin in ['Training','Test','Validation']:
        namenp = f"/home/simon/DATA/{site}/Ground_truth/{pathin}/Ground_truth_{pathin}_split_{split}.npy"
        path_Y_train = np.load(namenp)
        print(np.min(path_Y_train[:,1]))
        if np.min(path_Y_train[:,1]) == 1:
            path_Y_train[:,1]= path_Y_train[:,1]-1
            print(Path(namenp).name)
            np.save(f"/home/simon/DATA/{site}/Ground_truth2/{pathin}/Ground_truth_{pathin}_split_{split}.npy", path_Y_train)

breakpoint()        

for split in range(0,5):
    for pathin in ['Training']:
        namenp = f"{pathin}/Ground_truth_{pathin}_split_{split}.npy"
        path_Y_train = np.load(namenp)
        
        if np.min(path_Y_train[:,1]) == 1:
            path_Y_train[:,1]= path_Y_train[:,1]-1
            print(Path(namenp).name)
            np.save(Path(namenp).name, path_Y_train)
            



