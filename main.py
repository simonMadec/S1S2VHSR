from torch.utils.data import DataLoader
from src.dataset import DatasetSpot
from src.spot import Model_SPOT # import the model
from src.util_train import train
import os 

os.environ["CUDA_VISIBLE_DEVICES"] ="3"

split = 2

path_Y_train = f"/mnt/DATA/JE/data_reunion/Ground_truth/Training/Ground_truth_Training_split_{split}.npy"
path_X_train_pan = f"/mnt/DATA/JE/data_reunion/Spot-P/Training/Spot-P_Training_split_{split}.npy"
path_X_train_ms = f"/mnt/DATA/JE/data_reunion/Spot-MS/Training/Spot-MS_Training_split_{split}.npy"

path_Y_valid = f"/mnt/DATA/JE/data_reunion/Ground_truth/Validation/Ground_truth_Validation_split_{split}.npy"
path_X_valid_pan = f"/mnt/DATA/JE/data_reunion/Spot-P/Validation/Spot-P_Validation_split_{split}.npy"
path_X_valid_ms = f"/mnt/DATA/JE/data_reunion/Spot-MS/Validation/Spot-MS_Validation_split_{split}.npy"

path_Y_test = f"/mnt/DATA/JE/data_reunion/Ground_truth/Test/Ground_truth_Test_split_{split}.npy"
path_X_test_pan = f"/mnt/DATA/JE/data_reunion/Spot-P/Test/Spot-P_Test_split_{split}.npy"
path_X_test_ms = f"/mnt/DATA/JE/data_reunion/Spot-MS/Test/Spot-MS_Test_split_{split}.npy"


train_dataset = DatasetSpot(x_pan_numpy_dir=path_X_train_pan,x_ms_numpy_dir=path_X_train_ms,y_numpy_dir=path_Y_train)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=10)

valid_dataset = DatasetSpot(x_pan_numpy_dir=path_X_valid_pan,x_ms_numpy_dir=path_X_valid_ms,y_numpy_dir=path_Y_valid)
valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=True, num_workers=10)

test_dataset = DatasetSpot(x_pan_numpy_dir=path_X_test_pan,x_ms_numpy_dir=path_X_test_ms,y_numpy_dir=path_Y_test)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=10)

train(Model_SPOT(n_classes=11),train_loader,valid_loader,test_loader,num_epochs=2,csv_name=f"result_split-{split}.csv")
