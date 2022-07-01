

import glob
import numpy as np
import matplotlib.pyplot as plt 

sensor = "Spot-P"

ms = "Spot-MS"

site = "data_dordogne"
site = "data_reunion"
split = 0
for pathin in ['Test']:
    path_Y_train = np.load(f"/home/simon/DATA/land_use_classification/{site}/Ground_truth/{pathin}/Ground_truth_{pathin}_split_{split}.npy")
    path_X_train_Spot = np.load(f"/home/simon/DATA/land_use_classification/{site}/{sensor}/{pathin}/{sensor}_{pathin}_split_{split}.npy")
    path_X_train_MS = np.load(f"/home/simon/DATA/land_use_classification/{site}/{ms}/{pathin}/{ms}_{pathin}_split_{split}.npy")

    ii = 9
    WaterY = path_Y_train == ii
    Water = WaterY[:,1]
    paa = path_X_train_Spot[Water,:,:,0]
    mss = path_X_train_MS[Water,:,:,:,0]

    paa_m = np.mean(paa,(1,2))
    mss_m = np.mean(path_X_train_MS,(1,2))

    plt.close()
    n, bins, patches = plt.hist(paa_m[:,0],bins=[x/400 for x in range(0,200)],density = True,facecolor='green', alpha=0.75, stacked=True)
    plt.show()
    plt.savefig(f"his_debg_{pathin}.png")

    for k in range(0,4):
        plt.close()
        n, bins, patches = plt.hist(mss_m[:,k,0],bins=[x/400 for x in range(0,200)],density = True,facecolor='green', alpha=0.75, stacked=True)
        plt.show()
        plt.savefig(f"his_debg_MSS-{k}_{pathin}.png")

    plot = 1
    if plot:
        for i in range(0,paa.shape[0],2000):
            breakpoint()
            paai = paa[i,:,:,0].reshape([1,32,32])

            mssi = mss[i,:,:,:].transpose((-1,0,1))

            plt.close()
            mosaic = """
                 AAAA
                 AAAA
                 AAAA
                 AAAA
                 BCDE"""
            fig = plt.figure(constrained_layout=True)
            ax_dict = fig.subplot_mosaic(mosaic,subplot_kw={'xticks': [], 'yticks': []})
            A = ax_dict["A"].imshow(paai[0,:,:].astype("float32"),cmap='binary')
            plt.colorbar(A,ax=ax_dict["A"])
            B = ax_dict["B"].imshow(mssi[0,:,:].astype("float32"),cmap='binary')
            ax_dict["C"].imshow(mssi[1,:,:].astype("float32"),cmap='binary')
            ax_dict["D"].imshow(mssi[2,:,:].astype("float32"),cmap='binary')
            E = ax_dict["E"].imshow(mssi[3,:,:].astype("float32"),cmap='binary')

            plt.colorbar(E,ax=ax_dict["E"])
            plt.colorbar(E,ax=ax_dict["E"])
            plt.savefig(f"Debug_Watera{i}.png")


