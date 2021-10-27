
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SpotEncoder"]

class Conv2DBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=dropout),
            )

class Conv2DBlockPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, dropout, kernel_pooling, stride_pooling):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
            ),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_pooling,stride=stride_pooling),
            nn.Dropout(p=dropout),
            )

class SpotEncoder(nn.Module): #spot 
    def __init__(
            self,
            in_channels=1,
            n_filters=128,
            kernel_size=3,
            channel_concat = 4,
            drop=0.4,
            use_batchnorm=True, #todo pas utiliser encore 
    ):
        super().__init__()
        self.block1 = Conv2DBlockPooling(in_channels=in_channels,out_channels=n_filters,kernel_size=7,dropout=drop, kernel_pooling=3, stride_pooling=2)
        self.block2 = Conv2DBlock(in_channels=n_filters,out_channels=n_filters*2,kernel_size=5,dropout=drop)
        self.block3 = Conv2DBlockPooling(in_channels=n_filters*2 + channel_concat,out_channels=n_filters*2,kernel_size=3,dropout=drop, kernel_pooling=3, stride_pooling=2)
        self.block4 = Conv2DBlock(in_channels=n_filters*2,out_channels=n_filters*2,kernel_size=3,dropout=drop)
        self.block5 = Conv2DBlock(in_channels=n_filters*2,out_channels=n_filters*2,kernel_size=1,dropout=drop)
        self.avg = nn.AvgPool2d(1)
        
    def forward(self,input_pan,input_ms):
        x=self.block1(input_pan)
        x=self.block2(x)
        x = torch.cat((x, input_ms), 1) # 256 256 8  8 # todo c'est ok de pas le mettre dans self ? 
        x = F.pad(x, (0, 2, 2, 0)) # 256 256 10 10
        x=self.block3(x) # 256 256 3 3
        x=self.block4(x) # 256 256 1 1
        x=self.block5(x) # 256 256 1 1
        x=self.avg(x)
        return torch.squeeze(x)

class Model_SPOT(nn.Module):
    def __init__(self,n_classes,drop=0.4,n_filters=128,num_units=512):
        super().__init__()
        self.spot_encoder = SpotEncoder(n_filters=128,drop=drop)
        self.dense1 = nn.Linear(in_features = n_filters*2, out_features = num_units) 
        self.dense2 = nn.Linear(in_features = num_units, out_features = n_classes) 
        #self.softmax = torch.nn.Softmax(dim=1)

    def forward(self,x_pan,x_ms):
        x = self.spot_encoder(x_pan,x_ms) # rajouter is_training ici ?  #todo
        x = self.dense1(x)
        x = self.dense2(x)
        #x = self.softmax(x)
        return x
    
    def predict(self,x_pan,x_ms):
        """ Inference mode call forward with torch.no_grad() """
        if self.training:
            self.eval()
        with torch.no_grad():
            x = self.forward(x)
        return x