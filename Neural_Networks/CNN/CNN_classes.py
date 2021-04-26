import torch
import torch.nn as nn


torch.manual_seed(42)
torch.cuda.manual_seed(42)

#############################################
######### Dataloarder #######################
#############################################


def CNN_data_load(file: str = 'data/'):
    return torch.load(file)[0]

def load_CNN_series(file: str = 'data/'):
    tesor_id, timestep = file.rsplit("_",1)
    return torch.load(tesor_id+".pt")[int(timestep)]


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, labels, data_dir='data/'):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.data_dir = data_dir

    def __len__(self):
        'Denotes the totaal number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generate one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get Label -> may dynamic load for labels also?
        X = load_CNN_series(self.data_dir + ID)
        y = self.labels[ID]

        return X, y


#############################################
######### Lossfunction ######################
#############################################

def weighted_mse(input,target):
    dif = (input - target)**2
    out = dif*target
    return out.mean()

##############################################
####### Build Model ##########################
##############################################


def conv_block(c_in, c_out, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels=c_in, out_channels=c_out, *args, **kwargs),
        nn.BatchNorm2d(c_out)
    )


class phiCNN(nn.Module):
    def __init__(self, in_c, hidden_c):
        super().__init__()

        self.encoder = nn.Sequential(
            conv_block(c_in=in_c, c_out=hidden_c, kernel_size=3, stride=1, padding=1, dilation=1),
            conv_block(c_in=hidden_c, c_out=hidden_c, kernel_size=3, stride=1, padding=1, dilation=1),
        conv_block(c_in=hidden_c, c_out=hidden_c, kernel_size=3, stride=1, padding=1, dilation=1),  # extradeep
        conv_block(c_in=hidden_c, c_out=hidden_c, kernel_size=3, stride=1, padding=1, dilation=1),  # extradeep
        conv_block(c_in=hidden_c, c_out=hidden_c, kernel_size=3, stride=1, padding=1, dilation=1)  # extradeep
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=hidden_c, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Hardtanh(min_val=0, max_val=1)
        )

    def forward(self, x):
        x = self.encoder(x)

        x = self.decoder(x)[:,0]

        return x

class phiDeepCNN(nn.Module):
    def __init__(self, in_c, hidden_c):
        super().__init__()

        self.encoder = nn.Sequential(
            conv_block(c_in=in_c, c_out=hidden_c, kernel_size=3, stride=1, padding=1, dilation=1),
            conv_block(c_in=hidden_c, c_out=hidden_c, kernel_size=3, stride=1, padding=1, dilation=1),
            conv_block(c_in=hidden_c, c_out=hidden_c, kernel_size=3, stride=1, padding=1, dilation=1)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=hidden_c, out_channels=hidden_c, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Conv2d(in_channels=hidden_c, out_channels=hidden_c, kernel_size=3, stride=1, padding=1, dilation=1),

            nn.Conv2d(in_channels=hidden_c, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Hardtanh(min_val=0, max_val=1)
        )

    def forward(self, x):
        x = self.encoder(x)

        x = self.decoder(x)[:,0]

        return x
