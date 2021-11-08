

import glob
import numpy as np
import matplotlib.pyplot as plt 

sensor = "Spot-P"
ms = "Spot-MS"
site = "data_reunion"

split = 0
for pathin in ['Training']: #,'Validation','Test'
    path_Y_train = np.load(f"/mnt/DATA/JE/{site}/Ground_truth/{pathin}/Ground_truth_{pathin}_split_{split}.npy")
    path_X_train_Spot = np.load(f"/mnt/DATA/JE/{site}/{sensor}/{pathin}/{sensor}_{pathin}_split_{split}.npy")
    path_X_train_MS = np.load(f"/mnt/DATA/JE/{site}/{ms}/{pathin}/{ms}_{pathin}_split_{split}.npy")

    for ii in range(0,12):
        WaterY = path_Y_train == ii
        Water = WaterY[:,1]
        paa = path_X_train_Spot[Water,:,:,0]
        print(paa.shape[0])



    paa = path_X_train_Spot[Water,:,:,0]
    mss = path_X_train_MS[Water,:,:,:,0]

    breakpoint()
    for i in range(0,paa.shape[0],2000):
        plt.close()
        mosaic = """
            AAAA
            BCDE"""
        fig = plt.figure(constrained_layout=True)
        ax_dict = fig.subplot_mosaic(mosaic)

        A = ax_dict["A"].imshow(paa[i,:,:,0].astype("float32"),cmap='binary')
        plt.colorbar(A,ax=ax_dict["A"])
        B = ax_dict["B"].imshow(mss[i,:,:,0].astype("float32"),cmap='binary')
        ax_dict["C"].imshow(mss[i,:,:,1].astype("float32"),cmap='binary')
        ax_dict["D"].imshow(mss[i,:,:,2].astype("float32"),cmap='binary')
        E = ax_dict["E"].imshow(mss[i,:,:,3].astype("float32"),cmap='binary')
        plt.colorbar(E,ax=ax_dict["E"])

        plt.savefig(f"Debug_Water{i}.png")

