
import glob
import numpy as np
import matplotlib.pyplot as plt 
import glob
from pathlib import Path

for site in ["data_dordogne","data_reunion"]:
    for split in range(1,6):
        for pathin in ['Training','Test','Validation']:
            namenp = f"/home/simon/DATA/land_use_classification/{site}/Ground_truth/{pathin}/Ground_truth_{pathin}_split_{split}.npy"
            path_Y_train = np.load(namenp)
            print(np.min(path_Y_train[:,1]))
            if np.min(path_Y_train[:,1]) == 1:
                path_Y_train[:,1]= path_Y_train[:,1]-1
                print(Path(namenp).name)
                np.save(f"/home/simon/DATA/land_use_classification/{site}/Ground_truth/{pathin}/Ground_truth_{pathin}_split_{split}.npy", path_Y_train)


                



