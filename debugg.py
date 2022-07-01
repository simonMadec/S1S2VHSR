
import os
import torch
from src.MS import Model_MultiSource
from src.dataset_inference import Dataset_inference  
from torch.utils.data import DataLoader
from tqdm import tqdm
import time 
import numpy as np
import rasterio
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from src.datasetbig import  DatasetS1S2VHSRbig2

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

root = "/home/simon/DATA/land_use_classification/data" 

os.environ["CUDA_VISIBLE_DEVICES"] ="2"
debug = 0
batchsize = 8192

method = "v2branch"
method = "model"
#left,bottom,right,top
# bottom left : -21.323411, 55.483137
# top right : -21.314224, 55.495880

#bbox = [-21.323411, 55.483137, -21.314224, 55.495880]

# EPSG:32740
bbox = []
bbox = [343508.54,7641629.47, 345892.49, 7642825.10]
bbox = [341902.88,7639369.55,343065.43,7639558.30]

bbox = []

bbox = [341902.88,7639369.55,343065.43,7639558.30]

namemodel = "model1 10m ref Spot smallpart ref Spot"

for dataset in ["data_reunion_origin"]: # ,]: # "data_ToyReunion","data_ToyDordogne",,"data_dordogne_origin","data_ToyDordogne"
    for sensor in [["Spot","S1","S2"]]: #["S2"],["S1"],["Spot","S2"],["S1","S2","Spot"],["Spot","S2","S1"],["S1"],["S1","S2"],["Spot"],["S2"]]:
        # for sensor in [["Spot","S2","S1"]]:
        with torch.no_grad():

            if dataset in ["data_ToyReunion","data_reunion_origin"]:
                dataset_model = "data_reunion"
            elif dataset in ["data_ToyDordogne","data_dordogne_origin"]:
                dataset_model = "data_dordogne"
            else:
                dataset_model = dataset
            
            predict_dataset = Dataset_inference(root=root,dataset=dataset,bbox=bbox,sensor=sensor,ref=sensor[0])
            # Loading the saved model
            sensort = ["S1","S2","Spot"]
            save_path = f"{method}_{'-'.join(sensort)}_site-{dataset_model}.pth"
            model = Model_MultiSource(n_classes=predict_dataset.numtarget(),sensor=sensor)
            print(f"lodad model {save_path}")
            model.load_state_dict(torch.load(save_path))

            predict_loader = DataLoader(predict_dataset, batch_size=batchsize,pin_memory=True,num_workers=8)

            temp = "data_reunion_origin_out150cmGSD_v2"
            root = f"/home/simon/DATA/land_use_classification/data/{temp}"
            test_dataset = DatasetS1S2VHSRbig2(root=root,dataset="Test",sensor=sensor)
            test_loader = DataLoader(test_dataset, batch_size=256,pin_memory=True, num_workers=1)

            # A tester ,pin_memory=True, num_workers=10 .. 
            final_map = np.empty(predict_dataset.shape())
            final_map[:] = np.nan

            model.eval()
            model.cuda()
            print("model loaded")
            if debug ==1:
                print("only debug no prediction")
            else:   
                for i, sample in enumerate(tqdm(test_loader,desc= f"inference for model_{'-'.join(sensor)} for {dataset}")):
                    # for x in sample:
                    for x in sample:
                        if x!="ind":
                            sample[x] = sample[x].to('cuda:0', non_blocking=True)

                    outputs_is = model(sample)
                    pred = outputs_is.argmax(-1)
                    
                    correct_pred = (outputs_is.argmax(-1) == sample["Target"]).float()
                    val = correct_pred.sum() / len(correct_pred)
                    breakpoint()
                    row = sample["ind"]["r"].numpy()
                    col = sample["ind"]["c"].numpy()
                    final_map[row,col] = pred.cpu().numpy().astype('uint8')+1
            breakpoint()
            #profile = predict_dataset.get_source(bbox).profile
            profile = predict_dataset.get_profile(bbox)
            profile.update({'count': 1, 'dtype' : 'uint8'})

            outname = f"InfMap/Inf {method} {dataset} {'-'.join(sensor)} bbox {namemodel}"

            with rasterio.open(
                f"{outname}.tif",
                'w', **profile
                # transform=dat.transform,
            ) as dst:
                dst.write(final_map[np.newaxis,:,:].astype('uint8')) 

        if "reunion" in dataset_model:
            x_axis_labels = ["no data",'Sugarcane', 'Pasture and fodder', 'Market gardening','Grenhouse and shaded crops', 
                            'Orchards','Wooded areas','Moor and Savannah','Rocks and natural bare soil',
                            'Relief shadow','Water','Urbanized areas']
            li_color = ["#ffffff","#ffff00","#7ff416","#7f7f16","#c31616","#f5a205","#126b02","#75c500","#572f1c","#524f4d","#33d9e5","#96a6a7"]
        else:
            x_axis_labels = ["no data",'Urbanized areas', 'Water', 'Forest','Moor', 
            'Orchards','Vineyards','Other crops']
            li_color = ["#ffffff","#96a6a7","#33d9e5","#126b02","#c31616","#dea607","#8f063b","#25e03e"]
        # to consider : https://jingwen-z.github.io/how-to-draw-a-map-with-folium-module-in-python/
        color_map = ListedColormap([hex_to_rgb(x) for x in li_color] ,'indexed')
        color_map = ListedColormap([np.array(hex_to_rgb(x))/255 for x in li_color] ,'indexed')

        if final_map.shape[1]>2**15:
            print("image too large to be saved in png")
        else:
            plt.figure(figsize=tuple(int(x/30) for x in final_map.shape))
            plt.imshow(final_map,cmap=color_map)
            # color_map = ListedColormap([np.array(hex_to_rgb(x))/255 for x in li_color] ,'indexed')
            proxy = [plt.Rectangle((0,0),1,1,fc = color_map(0)[0:3]) , plt.Rectangle((0,0),1,1,fc = color_map(1)[0:3]) , plt.Rectangle((0,0),1,1,fc = color_map(2)[0:3]),
                plt.Rectangle((0,0),1,1,fc = color_map(3)[0:3]), plt.Rectangle((0,0),1,1,fc = color_map(4)[0:3]), plt.Rectangle((0,0),1,1,fc = color_map(5)[0:3]),
                plt.Rectangle((0,0),1,1,fc = color_map(6)[0:3]), plt.Rectangle((0,0),1,1,fc = color_map(7)[0:3]), plt.Rectangle((0,0),1,1,fc = color_map(8)[0:3]),
                plt.Rectangle((0,0),1,1,fc = color_map(9)[0:3]), plt.Rectangle((0,0),1,1,fc = color_map(10)[0:3])]
            legend = plt.legend(proxy, x_axis_labels , title=f"Inference {method} {'-'.join(sensor)}", bbox_to_anchor=(1.04,1), loc="upper left",prop={'size':int(final_map.shape[0]/30)}) #
            legend.get_title().set_fontsize(str(int(final_map.shape[0]/30)))
            plt.axis('off')
            plt.savefig(f"{outname}", dpi=30,bbox_inches = 'tight')
            plt.close()
             
        print(f"we are here")
        print("cc")
    breakpoint()

                    
                
            