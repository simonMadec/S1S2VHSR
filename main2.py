
from torch.utils.data import DataLoader
from src.dataset import DatasetS1S2VHSRbig
from src.MS import Model_MultiSource # import the model
from src.util_train import train
import os 
from src.util_study import merge_csv, merge_result
from pathlib import Path

# specify cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"] ="0"

# give path root of the location of your dataset
rootdataset = "/home/simon/DATA/land_use_classification/data"

# Method name to write report results csv
method = "v2Epoch5"

batch_size = 256

for site in ["data_reunion_origin_out150cmGSD_v2","data_dordogne_origin_out150cmGSD_v2"]:
    root = Path(rootdataset) / site
    for sensor in [["S2"],["Spot","S1","S2"],["Spot"],["S1"]]:
        
        csv_name = f"{method}_{'-'.join(sensor)}_site-{site}_result.csv"
        print(f"training for {csv_name}")
        # see ReadMe for Training / Validation / Split
        train_dataset = DatasetS1S2VHSRbig(root=root,dataset="Training",sensor=sensor)
        validation_dataset = DatasetS1S2VHSRbig(root=root,dataset="Validation",sensor=sensor)
        test_dataset = DatasetS1S2VHSRbig(root=root,dataset="Test",sensor=sensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,pin_memory=True, num_workers=1)
        valid_loader = DataLoader(validation_dataset, batch_size=batch_size,pin_memory=True, num_workers=1)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,pin_memory=True, num_workers=1)

        print(f"starting training")
        train(Model_MultiSource(n_classes=train_dataset.numtarget(),sensor=sensor,auxloss=True),train_loader,valid_loader,test_loader,save_model=True,num_epochs=10,csv_name=csv_name,model_file= f"{method}_{'-'.join(sensor)}_site-{site}.pth")
        
        # If we have multiple csv results this function merge them
        if (len(sensor)+len(site))>2:
            merge_csv(csv_name)
            merge_result(csv_name)