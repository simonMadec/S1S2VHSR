

from torch.utils.data import DataLoader
from src.dataset import DatasetS2
from src.S2 import Model_S2 # import the model
from src.util_train import train
import os 

os.environ["CUDA_VISIBLE_DEVICES"] ="1"
sensor = "Sentinel-2"

for split in range(0,5):

    path_Y_train = f"/mnt/DATA/JE/data_reunion/Ground_truth/Training/Ground_truth_Training_split_{split}.npy"
    path_X_train_S2 = f"/mnt/DATA/JE/data_reunion/{sensor}/Training/{sensor}_Training_split_{split}.npy"

    path_Y_valid = f"/mnt/DATA/JE/data_reunion/Ground_truth/Validation/Ground_truth_Validation_split_{split}.npy"
    path_X_valid_S2 = f"/mnt/DATA/JE/data_reunion/{sensor}/Validation/{sensor}_Validation_split_{split}.npy"

    path_Y_test = f"/mnt/DATA/JE/data_reunion/Ground_truth/Test/Ground_truth_Test_split_{split}.npy"
    path_X_test_S2 = f"/mnt/DATA/JE/data_reunion/{sensor}/Test/{sensor}_Test_split_{split}.npy"


    train_dataset = DatasetS2(x_S2_numpy_dir=path_X_train_S2,y_numpy_dir=path_Y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,pin_memory=False, num_workers=10)

    valid_dataset = DatasetS2(x_S2_numpy_dir=path_X_valid_S2,y_numpy_dir=path_Y_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=True,pin_memory=False, num_workers=10) #too load sur GPU (pin_memory=True)

    test_dataset = DatasetS2(x_S2_numpy_dir=path_X_test_S2,y_numpy_dir=path_Y_test)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True,pin_memory=False, num_workers=10)

    train(Model_S2(n_classes=11),train_loader,valid_loader,test_loader,num_epochs=300,csv_name=f"result-S1_split-{split}.csv",model_file=f"model-S1_split-{split}.pth")
