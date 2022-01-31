import torch

from torch.utils.data import DataLoader
from src.dataset import DatasetS1S2VHSR, sampler_create
from src.MS import Model_MultiSource # import the model
from src.util_train import train
import os 
from src.util_study import merge_csv, merge_result

os.environ["CUDA_VISIBLE_DEVICES"] ="2"
method = "inf2601"

# for sensor in [["S1"],["S2"],["Spot"],["S1","S2"],["S2","Spot"],["S1","S2","Spot"]]:
for site in ["data_reunion","data_dordogne"]:
    root = f"/home/simon/DATA/land_use_classification/{site}"
    for sensor in [["S1"],["S2"],["Spot"]]:
        for split in range(0,5):
            print(f"split {split}")
            train_dataset = DatasetS1S2VHSR(root=root,dataset="Training",sensor=sensor,split=split)
            sampler = sampler_create(root=root,dataset="Training",split=split)
            train_loader = DataLoader(train_dataset, batch_size=256,pin_memory=True,num_workers=1,sampler=sampler) #shuffle=True 
            test_dataset = DatasetS1S2VHSR(root=root,dataset="Validation",sensor=sensor,split=split)
            valid_loader = DataLoader(test_dataset, batch_size=256,pin_memory=True, num_workers=1) #too load sur GPU (pin_memory=True)
            test_dataset = DatasetS1S2VHSR(root=root,dataset="Test",sensor=sensor,split=split)
            test_loader = DataLoader(test_dataset, batch_size=256,pin_memory=True, num_workers=1)

            csv_name = f"{method}_{'-'.join(sensor)}_site-{site}_result_split-{split}.csv"
            train(Model_MultiSource(n_classes=train_dataset.numtarget(),sensor=sensor,auxloss=True),train_loader,valid_loader,test_loader,save_model=True,num_epochs=100,csv_name=csv_name,model_file= f"{method}_{'-'.join(sensor)}_site-{site}.pth")

        merge_csv(csv_name)
        merge_result(csv_name)