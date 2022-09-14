
from torch.utils.data import DataLoader
from src.dataset import DatasetS1S2VHSRbig
from src.MS import Model_MultiSource # import the model
from src.util_train import train
import os 
from src.util_study import merge_csv, merge_result
from pathlib import Path

# specify cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"] ="1"

# give path root of the location of your dataset
rootdataset = "/home/simon/DATA/land_use_classification/data"

# Method name to write report results csv
method = "dd2"
batch_size = 256
cc =0

for site in ["data_dordogne_origin_out150cmGSD_v2"]:
    root = Path(rootdataset) / site
    for sensor in [["S1"]]:#["S1","S2","Spot"]
        for split in range(0,5):
            cc=cc+1
            csv_name = f"{method}_{'-'.join(sensor)}_site-{site}__split-{split}_result.csv"
            print(f"training for {csv_name}")
            # see ReadMe for Training / Validation / Split
            train_dataset = DatasetS1S2VHSRbig(root=root,dataset="Training",sensor=sensor,split=split) 
            # you cann add here argument path_size and num_target argument here
            validation_dataset = DatasetS1S2VHSRbig(root=root,dataset="Validation",sensor=sensor,split=split)
            test_dataset = DatasetS1S2VHSRbig(root=root,dataset="Test",sensor=sensor,split=split)

            train_loader = DataLoader(train_dataset, batch_size=batch_size,pin_memory=True, num_workers=1)
            valid_loader = DataLoader(validation_dataset, batch_size=batch_size,pin_memory=True, num_workers=1)
            test_loader = DataLoader(test_dataset, batch_size=batch_size,pin_memory=True, num_workers=1)

            print(f"starting training")
            train(Model_MultiSource(n_classes=train_dataset.numtarget(),sensor=sensor,auxloss=True),train_loader,valid_loader,test_loader,
                save_model=True,num_epochs=1,csv_name=csv_name,model_file= f"{method}_{'-'.join(sensor)}_site-{site}_split-{split}.pth")
            
            # If we have multiple csv results this function merge them
            #care full this might not work .. or need to be changed
        if (cc)>1:
            merge_csv(csv_name)
            merge_result(csv_name)