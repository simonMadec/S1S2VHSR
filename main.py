import torch

from torch.utils.data import DataLoader
from src.dataset import DatasetS1S2VHSR
from src.MS import Model_MultiSource # import the model
from src.util_train import train
import os 
from util_study import merge_csv

os.environ["CUDA_VISIBLE_DEVICES"] ="0"

 # attention changer le nombre de classes et test -1 dans validate all
site = "data_dordogne"
method = "0512_epoch1"

root = f"/home/simon/DATA/land_use_classification/{site}"

for sensor in [["S1"],["S2"],["Spot"]]:
    for split in range(0,5):
        print(f"split {split}")

        train_dataset = DatasetS1S2VHSR(root=root,dataset="Training",sensor=sensor,split=split)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,pin_memory=False, num_workers=1)

        test_dataset = DatasetS1S2VHSR(root=root,dataset="Validation",sensor=sensor,split=split)
        valid_loader = DataLoader(test_dataset, batch_size=256, shuffle=True,pin_memory=False, num_workers=1) #too load sur GPU (pin_memory=True)

        test_dataset = DatasetS1S2VHSR(root=root,dataset="Test",sensor=sensor,split=split)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True,pin_memory=False, num_workers=1)

        csv_name = f"{method}_{'-'.join(sensor)}_site-{site}_result_split-{split}.csv"

        train(Model_MultiSource(n_classes=train_dataset.numtarget(),branch=sensor),train_loader,valid_loader,test_loader,num_epochs=2,csv_name=csv_name)

    merge_csv(csv_name)