import torch

from torch.utils.data import DataLoader
from src.dataset import DatasetS1S2VHSR, DatasetS1S2VHSRbig, sampler_create
from src.datasetbig import  DatasetS1S2VHSRbig2

from src.MS import Model_MultiSource # import the model
from src.util_train import train
import os 
from src.util_study import merge_csv, merge_result

os.environ["CUDA_VISIBLE_DEVICES"] ="0"
for method in ["split1","split2","split3","split4","split5"]:
    batch_size = 256
    # for sensor in [["S1"],["S2"],["Spot"],["S1","S2"],["S2","Spot"],["S1","S2","Spot"]]:
    for site in ["data_reunion_origin_out150cmGSD_v2"]:
        root = f"/home/simon/DATA/land_use_classification/data/{site}"
        for sensor in [["Spot","S2","S1"]]:  
            csv_name = f"{method}_{'-'.join(sensor)}_site-{site}_result.csv"
            print(f"training for {csv_name}")
            train_dataset = DatasetS1S2VHSRbig2(root=root,dataset="Training",sensor=sensor)
            validation_dataset = DatasetS1S2VHSRbig2(root=root,dataset="Validation",sensor=sensor)
            test_dataset = DatasetS1S2VHSRbig2(root=root,dataset="Test",sensor=sensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size,pin_memory=True, num_workers=1)
            valid_loader = DataLoader(validation_dataset, batch_size=batch_size,pin_memory=True, num_workers=1)
            test_loader = DataLoader(test_dataset, batch_size=batch_size,pin_memory=True, num_workers=1)
            print(f"starting training")
            train(Model_MultiSource(n_classes=train_dataset.numtarget(),sensor=sensor,auxloss=True),train_loader,valid_loader,test_loader,save_model=True,num_epochs=100,csv_name=csv_name,model_file= f"{method}_{'-'.join(sensor)}_site-{site}.pth")
            # todo envlever comm ici et tester avec 5 epoch changer epocj 
            merge_csv(csv_name)
            print("hello")
            merge_result(csv_name)