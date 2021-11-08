
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, dropout, stride=1): #padding_mode='circular'
        super().__init__(
            nn.Conv1d(in_channels,out_channels,kernel_size=kernel_size, stride=stride), #padding_mode=padding_mode
            nn.BatchNorm1d(out_channels),
            nn.Dropout(p=dropout),
            )


class CNN1D_Encoder(nn.Module): #s2 
    def __init__(self,in_channels=1,n_filters=128,drop=0.4,use_batchnorm=True): #todo batchnorm pas utiliser encore   
        super().__init__()
        self.block1 = Conv1DBlock(in_channels=6,out_channels=n_filters,kernel_size=5,dropout=drop) #todo
        self.block2 = Conv1DBlock(in_channels=n_filters,out_channels=n_filters*2,kernel_size=3,dropout=drop,stride=2)
        self.block3 = Conv1DBlock(in_channels=n_filters*2,out_channels=n_filters*2,kernel_size=3,dropout=drop)
        self.block4 = Conv1DBlock(in_channels=n_filters*2,out_channels=n_filters*2,kernel_size=1,dropout=drop)
        self.avg = nn.AvgPool1d(6)
        self.act = nn.ReLU()
        
    def forward(self,inputs): #[256, 6, 21])
        x=self.block1(inputs) # [256, 6, 21])
        x = self.act(x)
        x=self.block2(x) # [256, 256, 8]
        x = self.act(x)
        x=self.block3(x) # [256, 256, 6] 
        x = self.act(x)
        x=self.block4(x) # [256, 256, 6]
        x = self.act(x)
        x=self.avg(x) # [256, 256, 6] todo normal ?
        x=torch.squeeze(x)  # [256, 256, 1]
        return x #[256, 256]

class Model_S2(nn.Module):
    def __init__(self,n_classes,drop=0.4,n_filters=128,num_units=512):
        super().__init__()
        self.s1_branch = CNN1D_Encoder(n_filters=n_filters,drop=drop)
        self.dense1 = nn.Linear(in_features = n_filters*2, out_features = num_units) 
        self.dense2 = nn.Linear(in_features = num_units, out_features = n_classes) 
        #self.softmax = torch.nn.Softmax(dim=1)

    def forward(self,x_s2):
        x = self.s1_branch(x_s2) # rajouter is_training ici ?  #todo
        x = self.dense1(x)
        x = self.dense2(x)
        #x = self.softmax(x)
        return x

