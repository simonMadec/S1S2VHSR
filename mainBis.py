from torch.utils.data import DataLoader
from src.dataset import DatasetSpot
from src.spot import Model_SPOT # import the model
from src.util_train import train
import os 

os.environ["CUDA_VISIBLE_DEVICES"] ="1"

site = "data_reunion"
site = "data_dordogne" # attention changer le nombre de classes et test -1 dans validate all


for rep in range(3,5):
    for split in range(0,5):

        path_Y_train = f"/mnt/DATA/JE/{site}/Ground_truth/Training/Ground_truth_Training_split_{split}.npy"
        path_X_train_pan = f"/mnt/DATA/JE/{site}/Spot-P/Training/Spot-P_Training_split_{split}.npy"
        path_X_train_ms = f"/mnt/DATA/JE/{site}/Spot-MS/Training/Spot-MS_Training_split_{split}.npy"

        path_Y_valid = f"/mnt/DATA/JE/{site}/Ground_truth/Validation/Ground_truth_Validation_split_{split}.npy"
        path_X_valid_pan = f"/mnt/DATA/JE/{site}/Spot-P/Validation/Spot-P_Validation_split_{split}.npy"
        path_X_valid_ms = f"/mnt/DATA/JE/{site}/Spot-MS/Validation/Spot-MS_Validation_split_{split}.npy"

        path_Y_test = f"/mnt/DATA/JE/{site}/Ground_truth/Test/Ground_truth_Test_split_{split}.npy"
        path_X_test_pan = f"/mnt/DATA/JE/{site}/Spot-P/Test/Spot-P_Test_split_{split}.npy"
        path_X_test_ms = f"/mnt/DATA/JE/{site}/Spot-MS/Test/Spot-MS_Test_split_{split}.npy"

        train_dataset = DatasetSpot(x_pan_numpy_dir=path_X_train_pan,x_ms_numpy_dir=path_X_train_ms,y_numpy_dir=path_Y_train)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,pin_memory=False, num_workers=10)

        valid_dataset = DatasetSpot(x_pan_numpy_dir=path_X_valid_pan,x_ms_numpy_dir=path_X_valid_ms,y_numpy_dir=path_Y_valid)
        valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=True,pin_memory=False, num_workers=10) #too load sur GPU (pin_memory=True)

        test_dataset = DatasetSpot(x_pan_numpy_dir=path_X_test_pan,x_ms_numpy_dir=path_X_test_ms,y_numpy_dir=path_Y_test)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True,pin_memory=False, num_workers=10)

        train(Model_SPOT(n_classes=7),train_loader,valid_loader,test_loader,num_epochs=200,csv_name=f"Spot-withrelu_site-{site}_rep-{rep}_result_split-{split}.csv",model_file=f"model_split-{split}.pth")
