


class Conv1DBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, dropout, strides=1,padding_mode='valid'):
        super().__init__(
            nn.Conv1D(in_channels,out_channels,kernel_size=kernel_size, padding=padding_mode, strides=strides)
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=dropout),
            )



class CNN1D_Encoder(nn.Module): #spot 
    def __init__(self,n_filters=128,drop=0.4,use_batchnorm=True): #todo batchnorm pas utiliser encore   
        super().__init__()
        self.block1 = Conv1DBlock(in_channels=in_channels,out_channels=out_channels,kernel_size=5,dropout=drop)
        self.block2 = Conv1DBlock(in_channels=in_channels,out_channels=out_channels,kernel_size=3,dropout=drop,strides=2)
        self.block3 = Conv1DBlock(in_channels=in_channels,out_channels=out_channels,kernel_size=3,dropout=drop)
        self.block4 = Conv1DBlock(in_channels=in_channels,out_channels=out_channels,kernel_size=1,dropout=drop)
        self.avg = nn.AvgPool2d(1)
        
    def forward(self,inputs):
        x=self.block1(inputs)
        x=self.block2(x)
        x=self.block3(x) 
        x=self.block4(x) 
        return torch.squeeze(x) #squeeze ici ? todo

class Model_S1(nn.Module):
    def __init__(self,n_classes,drop=0.4,n_filters=128,num_units=512):
        super().__init__()
        self.s1_branch = CNN1D_Encoder(n_filters=n_filters,drop=drop)
        self.dense1 = nn.Linear(in_features = n_filters*2, out_features = num_units) 
        self.dense2 = nn.Linear(in_features = num_units, out_features = n_classes) 
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self,x_pan,x_ms):
        x = self.s1_branch(x_pan,x_ms) # rajouter is_training ici ?  #todo
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.softmax(x)
        return x

