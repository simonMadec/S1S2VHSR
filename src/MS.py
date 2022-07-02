
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Model_MultiSource"]

class Conv2DBlock(nn.Sequential):
    def __init__(self, out_channels, kernel_size, dropout, stride=1): #padding_mode='circular'
        super().__init__(
            nn.LazyConv2d(out_channels,kernel_size=kernel_size, stride=stride), #padding_mode=padding_mode
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=dropout),
            )

class Conv2DBlockPooling(nn.Sequential):
    def __init__(self, out_channels, kernel_size, dropout, kernel_pooling, stride_pooling):
        super().__init__(
            nn.LazyConv2d(out_channels,kernel_size=kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_pooling,stride=stride_pooling),
            nn.Dropout(p=dropout),
            )

class CNN2D_Encoder(nn.Module): #s2 
    def __init__(self,n_filters=128,drop=0.4):    
        super().__init__()
        self.block1 = Conv2DBlock(out_channels=n_filters,kernel_size=3,dropout=drop) #todo
        self.block2 = Conv2DBlock(out_channels=n_filters*2,kernel_size=3,dropout=drop)
        self.block3 = Conv2DBlock(out_channels=n_filters*2,kernel_size=3,dropout=drop)
        self.block4 = Conv2DBlock(out_channels=n_filters*2,kernel_size=1,dropout=drop)
        self.avg = nn.AvgPool2d(3)
        self.act = nn.ReLU()
    def forward(self,inputs): #[256, 52, 9, 9]
        x=self.block1(inputs) #[256, 128, 7, 7]
        x = self.act(x)
        x=self.block2(x) # [256, 256, 5, 5]
        x = self.act(x)
        x=self.block3(x) # [256, 256, 3, 3]
        x = self.act(x)
        x=self.block4(x) # [256, 256, 3, 3]
        x = self.act(x)
        x=self.avg(x)  # [256, 256, 1, 1]
        x=torch.squeeze(x)  
        return x 

class SpotEncoder(nn.Module): #spot 
    def __init__(
            self,
            n_filters=128,
            channel_concat = 4,
            drop=0.4,
    ):
        super().__init__()
        self.block1 = Conv2DBlockPooling(out_channels=n_filters,kernel_size=7,dropout=drop, kernel_pooling=3, stride_pooling=2)
        self.block2 = Conv2DBlock(out_channels=n_filters*2,kernel_size=5,dropout=drop)
        self.block3 = Conv2DBlockPooling(out_channels=n_filters*2,kernel_size=3,dropout=drop, kernel_pooling=3, stride_pooling=2)
        self.block4 = Conv2DBlock(out_channels=n_filters*2,kernel_size=3,dropout=drop)
        self.block5 = Conv2DBlock(out_channels=n_filters*2,kernel_size=1,dropout=drop)
        self.avg = nn.AvgPool2d(1)
        self.act = nn.ReLU()
        
    def forward(self,input_pan,input_ms):
        x=self.block1(input_pan)
        x = self.act(x)
        x=self.block2(x) # 
        x = self.act(x)
        x = torch.cat((x, input_ms), 1) # 
        x = F.pad(x, (0, 2, 2, 0)) # 256 256 10 10 
        x=self.block3(x) # 256 256 3 3
        x = self.act(x)
        x=self.block4(x) # 256 256 1 1
        x = self.act(x)
        x=self.block5(x) # 256 256 1 1
        x = self.act(x)
        x=self.avg(x)
        return torch.squeeze(x)
class Conv1DBlock(nn.Sequential):
    def __init__(self, out_channels, kernel_size, dropout, stride=1): #padding_mode='circular'
        super().__init__(
            nn.LazyConv1d(out_channels,kernel_size=kernel_size, stride=stride), #padding_mode=padding_mode
            nn.BatchNorm1d(out_channels),
            nn.Dropout(p=dropout),
            )
class CNN1D_Encoder(nn.Module): #s2 
    def __init__(self,n_filters=128,drop=0.4,use_batchnorm=True): 
        super().__init__()
        self.block1 = Conv1DBlock(out_channels=n_filters,kernel_size=5,dropout=drop) #todo
        self.block2 = Conv1DBlock(out_channels=n_filters*2,kernel_size=3,dropout=drop,stride=2)
        self.block3 = Conv1DBlock(out_channels=n_filters*2,kernel_size=3,dropout=drop)
        self.block4 = Conv1DBlock(out_channels=n_filters*2,kernel_size=1,dropout=drop)

        self.avg = nn.AvgPool1d(6)
        self.act = nn.ReLU()
        
    def forward(self,inputs): #[256, 6, 21]) haa debug todo
        x=self.block1(inputs) # [256, 6, 21]) haa debug todo
        x = self.act(x)
        x=self.block2(x) # [256, 256, 8]
        x = self.act(x)
        x=self.block3(x) # [256, 256, 6] 
        x = self.act(x)
        x=self.block4(x) # [256, 256, 6]
        x = self.act(x)
        x=self.avg(x) # [256, 256, 6] 
        x=torch.squeeze(x)  # [256, 256, 1]
        return x #[256, 256]
class Model_MultiSource(nn.Module):
    def __init__(self,n_classes,sensor,drop=0.4,n_filters=128,num_units=512,auxloss=False):
        super().__init__()
        self.sensor = sensor
        self.aux = auxloss
        if "Spot" in sensor:
            self.spot_encoder = SpotEncoder(n_filters=n_filters,drop=drop)
        if "S1" in sensor:
            self.s1_encoder = CNN2D_Encoder(n_filters = n_filters, drop = drop)
        if "S2" in sensor:
            self.s2_encoder = CNN1D_Encoder(n_filters = n_filters, drop = drop)

        if len(sensor)>1:
            print("fusion is on development")

        self.dense1 = nn.Linear(in_features = n_filters*2, out_features = num_units) 
        self.dense2 = nn.Linear(in_features = num_units, out_features = n_classes) 
        self.denseaux = nn.Linear(in_features = n_filters*2, out_features = num_units) 

    def forward(self,x_in):
        output = {}
        if len(self.sensor)==1:
            if "Spot" in self.sensor:
                x = self.spot_encoder(x_in["PAN"],x_in["MS"])
            elif "S1" in self.sensor:
                x = self.s1_encoder(x_in["S1"])
            elif "S2" in self.sensor:
                x = self.s2_encoder(x_in["S2"])
                
            x = self.dense1(x)
            x = self.dense2(x)
            return x

        elif len(self.sensor)==2:
            if "Spot" not in self.sensor:
                x1 = self.s1_encoder(x_in["S1"])
                x2 = self.s2_encoder(x_in["S2"])
                x = torch.add(x1,x2)
                output["S1"] = self.denseaux(x1)
                output["S2"] = self.denseaux(x2)

            if "S1" not in self.sensor:
                x1 = self.spot_encoder(x_in["PAN"],x_in["MS"])
                x2 = self.s2_encoder(x_in["S2"])
                x = torch.add(x1,x2)
                output["Spot"] = self.denseaux(x1)
                output["S2"] = self.denseaux(x2)

            if "S2" not in self.sensor:
                x1 = self.s1_encoder(x_in["S1"])
                x2 = self.spot_encoder(x_in["PAN"],x_in["MS"])
                x = torch.add(x1,x2)
                output["S1"] = self.denseaux(x1)
                output["Spot"] = self.denseaux(x2)

            x = self.dense1(x)
            x = self.dense2(x)
            output["fusion"] = x

            if self.aux==False: #maybe remove this
                return x
            else:
                # return x,self.denseaux(x1),self.denseaux(x2)
                return output

        elif len(self.sensor)==3:
            x1 = self.s1_encoder(x_in["S1"])
            x2 = self.s2_encoder(x_in["S2"])
            x3 = self.spot_encoder(x_in["PAN"],x_in["MS"])
            x = torch.add(torch.add(x1,x2),x3)
            x = self.dense1(x)
            x = self.dense2(x)

            if self.aux==False:
                return x
            else:
                output["fusion"] = x 
                output["S1"] = self.denseaux(x1) 
                output["S2"] = self.denseaux(x2) 
                output["Spot"] = self.denseaux(x3)
                # return x,self.denseaux(x1),self.denseaux(x2),self.denseaux(x3)
                return output
        else:
            print("problem of input more than four sources are in input!")
            breakpoint()
        

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)