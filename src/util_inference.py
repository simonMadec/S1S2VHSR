
import os
import torch
from MS import Model_MultiSource
from dataset_inference import Dataset_inference  
from torch.utils.data import DataLoader
from tqdm import tqdm
import time 
import numpy as np
import rasterio
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

sensor = ["S1","S2"]

sensor = ["S1","S2","Spot"]
sensor = ["S1","S2"]
sensor = ["S1","S2","Spot"]

sensor = ["S1","S2","Spot"]

sensor = ["S1","S2"]
root = "/home/simon/DATA/land_use_classification" 


# datasedataset = "data_reunion"t = "data_ToyReunion"


dataset = "data_reunion"
dataset = "data_ToyReunion"
os.environ["CUDA_VISIBLE_DEVICES"] ="1"

batchsize = 2048
for sensor in [["S1"],["S2"],["Spot"]]: #,["S2"],["Spot"],["S1","S2"],["S1","S2","Spot"]
    with torch.no_grad():
        # Loading the saved model
        predict_dataset = Dataset_inference(root=root,dataset=dataset,sensor=sensor,ref=sensor[0])
        predict_loader = DataLoader(predict_dataset, batch_size=batchsize,pin_memory=True,num_workers=1)
        final_map = np.empty(predict_dataset.shape())
        final_map[:] = np.nan

        save_path = f"model/inf2401_{'-'.join(sensor)}_site-data_reunion.pth"

        model = Model_MultiSource(n_classes=predict_dataset.numtarget(),sensor=sensor)
        model.load_state_dict(torch.load(save_path))
        model.eval()
        model.cuda()
        
        for i, sample in tqdm(enumerate(predict_loader)):
            for x in sample:
                for x in sample:
                    if x!="ind":
                        sample[x] = sample[x].to('cuda:0', non_blocking=True)

                outputs_is = model(sample)
                # outputs_is = outputs_is[0]
                pred = outputs_is.argmax(-1)
                row = sample["ind"]["r"].numpy()
                col = sample["ind"]["c"].numpy()
                final_map[row,col] = pred.cpu().numpy()

        # to QGIS now
        with predict_dataset.get_source() as src:
            # breakpoint()
            # data = src.read()
            with rasterio.open(
                'final_map.tif',
                'w',
                driver='GTiff',
                height=src.shape[0],
                width=src.shape[1],
                count=1,
                dtype="uint16",
                crs=src.crs,
                # transform=dat.transform,
            ) as dst:
                dst.write(final_map[np.newaxis,:,:]) 


    x_axis_labels = ['Sugarcane', 'Pasture and fodder', 'Market gardening','Grenhouse and shaded crops', 
                    'Orchards','Wooded areas','Moor and Savannah','Rocks and natural bare soil',
                    'Relief shadow','Water','Urbanized areas']

    li_color = ["#ffff00","#7ff416","#7f7f16","#c31616","#f5a205","#126b02","#75c500","#572f1c","#524f4d","#33d9e5","#96a6a7"]
    color_map = ListedColormap([hex_to_rgb(x) for x in li_color] ,'indexed')

    # define color map 
    # color_map = {0: "#c2ed41", # jaune
    #              1: "#00af00", # vert
    #              2: "#32dcda", # cyan
    #              3: "#dddcda", # blanc gris clair
    #              4: "#25e03e", # vert clair
    #              5: "#206829"), # vert fonce
    #              6: "#d38e2c"), # orange
    #              7: "#b8afb7", # gris
    #              8: "#300533", # noir
    #              9: "#0719b1", # bleu fonce
    #              10: "#8c1094"} # violet 
    color_map = ListedColormap([np.array(hex_to_rgb(x))/255 for x in li_color] ,'indexed')

    plt.imshow(final_map,cmap=color_map)

    # color_map = ListedColormap([np.array(hex_to_rgb(x))/255 for x in li_color] ,'indexed')

    proxy = [plt.Rectangle((0,0),1,1,fc = color_map(0)[0:3]) , plt.Rectangle((0,0),1,1,fc = color_map(1)[0:3]) , plt.Rectangle((0,0),1,1,fc = color_map(2)[0:3]),
        plt.Rectangle((0,0),1,1,fc = color_map(3)[0:3]), plt.Rectangle((0,0),1,1,fc = color_map(4)[0:3]), plt.Rectangle((0,0),1,1,fc = color_map(5)[0:3]),
        plt.Rectangle((0,0),1,1,fc = color_map(6)[0:3]), plt.Rectangle((0,0),1,1,fc = color_map(7)[0:3]), plt.Rectangle((0,0),1,1,fc = color_map(8)[0:3]),
        plt.Rectangle((0,0),1,1,fc = color_map(9)[0:3]), plt.Rectangle((0,0),1,1,fc = color_map(10)[0:3])]

    plt.legend(proxy, x_axis_labels , title=f"Inference {'-'.join(sensor)}", bbox_to_anchor=(1.04,1), loc="upper left") #
    plt.axis('off')
    #plt.legend(x_axis_labels)
    plt.savefig(f"Inference  {'-'.join(sensor)}",bbox_inches = 'tight')
breakpoint()
                
            
        