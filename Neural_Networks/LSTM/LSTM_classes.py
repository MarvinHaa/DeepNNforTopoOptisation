import torch
import torch.nn as nn
import torch.nn.functional as F

import math

torch.manual_seed(42)
torch.cuda.manual_seed(42)


#############################################
######### Dataloarder #######################
#############################################

def load_LSTM_series(file: str = 'data/'):
    tesor_id, timestep = file.rsplit("_", 1)
    return torch.load(tesor_id + ".pt")[int(timestep) * 5:(int(timestep) + 1) * 5, :, :, :]  # TO DO: variable sequence leght


def LSTM_load(file: str = 'data/'):
    return torch.load(file)[5:25, :, :, :]


class HadamardLayer(nn.Module):
    """ Custom Layer for Hardmandproduct with an lernable Wightmatrix W*C"""
    def __init__(self, H=101, W=201):
        super().__init__()
        self.Height = H
        self.Width = W
        weights = torch.Tensor(self.Height, self.Width)
        self.weights = nn.Parameter(weights,requires_grad=True)
        bias = torch.Tensor(self.Height, self.Width)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        out = x * self.weights # transpose ?
        return out #torch.add(out, self.bias)




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
        # print(self.data_dir + ID)
        # X = load_CNN_series(self.data_dir + ID)
        X = load_LSTM_series(self.data_dir + ID)
        y = self.labels[ID]

        return X, y


#############################################
######### Lossfunction ######################
#############################################

def weighted_mse(input, target):
    dif = (input - target) ** 2
    out = dif * target
    return out.mean()


##############################################
####### Build Model ##########################
##############################################


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channel, hidden_channel, kernel_size, bias):
        super().__init__()

        self.input_channel = input_channel
        self.hidden_channel = hidden_channel

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2  # or just 1
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_channel + self.hidden_channel,
                              out_channels=4 * self.hidden_channel,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        print("h_cur.size()", h_cur.size())
        print("c_cur.size()", c_cur.size())
        print("h_cur.device", h_cur.device)
        print("c_cur.device", c_cur.device)
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel (may curse error -> check if 1 is really channel dimension)
        print("combined.size()", combined.size())
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channel, dim=1)
        print("cc_i", cc_i.size())
        print("cc_f", cc_f.size())
        print("cc_o", cc_o.size())
        print("cc_g", cc_g.size())
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_channel, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_channel, height, width, device=self.conv.weight.device))


class MarvConvLSTMCell(nn.Module):
    def __init__(self, input_channel, hidden_channel, kernel_size, bias):
        super().__init__()

        self.input_channel = input_channel
        self.hidden_channel = hidden_channel

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2  # or just 1
        self.bias = bias
        print("hidden Channel:", hidden_channel)
        # dtype = torch.float
        # device = torch.device("cpu")
        # self.W_c = torch.randn((hidden_channel*3, 101, 201), device=device, dtype=dtype, requires_grad=True)
        # self.W_c = torch.FloatTensor((hidden_channel*3, 101, 201), requires_grad=True) # Height and With hardcoded!!!!!!!!!!! TO DO: Change
        #self.W_ci = torch.nn.Parameter(data=torch.rand(hidden_channel, 101, 201), requires_grad=True)
        #self.W_cf = torch.nn.Parameter(data=torch.rand(hidden_channel, 101, 201), requires_grad=True)
        #self.W_co = torch.nn.Parameter(data=torch.rand(hidden_channel, 101, 201), requires_grad=True)
        # print("initial W_ci", self.W_ci)
        self.W_ci = HadamardLayer(101, 201)
        self.W_cf = HadamardLayer(101, 201)
        self.W_co = HadamardLayer(101, 201)

        # print("Initial Weights", self.W_ci.weights)

        self.convX = nn.Conv2d(in_channels=self.input_channel,  # + self.hidden_channel,
                                out_channels=4 * self.hidden_channel,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)



        self.convH = nn.Conv2d(in_channels=self.hidden_channel,  # self.input_channel + self.hidden_channel,
                               out_channels=4 * self.hidden_channel,
                               kernel_size=self.kernel_size,
                               padding=self.padding,
                               bias=self.bias)

        self.BatchNorm2d = nn.BatchNorm2d(4 * self.hidden_channel)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        # print("h_cur.size()", h_cur.size())
        # print("c_cur.size()", c_cur.size())
        # print("h_cur.device", h_cur.device)
        # print("c_cur.device", c_cur.device)
        # combined = torch.cat([input_tensor, h_cur], dim=1) # concatenate along channel (may curse error -> check if 1 is really channel dimension)
        # print("combined.size()",combined.size())
        # print("input Tensor", input_tensor.size())
        # combined_conv = self.conv(combined)
        convX = self.BatchNorm2d(self.convX(input_tensor)) ## um kontrast zu steigern
        convH = self.convH(h_cur)
        X_cc_i, X_cc_f, X_cc_o, X_cc_C = torch.split(convX, self.hidden_channel, dim=1)
        H_cc_i, H_cc_f, H_cc_o, H_cc_C = torch.split(convH, self.hidden_channel, dim=1)

        W_ci = self.W_ci(c_cur)
        W_cf = self.W_co(c_cur)

        # W_c = self.W_c.unsqueeze(0)
        # W_ci, W_cf, W_co = W_c.chunk(3, 1)
        # print("W_ci",sum(sum(self.W_ci)))

        # print("cc_i",cc_i.size())
        # print("cc_f", cc_f.size())
        # print("cc_o", cc_o.size())
        # print("cc_g", cc_g.size())
        i = torch.sigmoid(X_cc_i + H_cc_i + W_ci)
        f = torch.sigmoid(X_cc_f + H_cc_f + W_cf)
        c_next = f * c_cur + i * torch.tanh(X_cc_C + H_cc_C)

        W_co = self.W_co(c_next)
        o = torch.sigmoid(X_cc_o + H_cc_o + W_co)
        h_next = o * torch.tanh(c_next)

        # c_next = f * c_cur + i * g
        # h_next = o * torch.tanh(c_next)
        # print(self.W_ci)
        # print("Weights:", self.W_co.weight)
        # print("Conv Weights", self.convX.weight.data.numpy())
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_channel, height, width, device=self.convX.weight.device),
                torch.zeros(batch_size, self.hidden_channel, height, width, device=self.convH.weight.device))


class EncoderDecoderPhiLSTM(nn.Module):
    def __init__(self, nf, in_chan):
        super().__init__()

        """ ARCHITECTURE 

            # Encoder (ConvLSTM)
            # Encoder Vector (final hidden state of encoder)
            # Decoder (ConvLSTM) - takes Encoder Vector as input
            # Decoder (3D CNN) - produces regression predictions for our model
        """

        self.encoder_1_convlstm = MarvConvLSTMCell(input_channel=in_chan,
                                                   hidden_channel=nf,
                                                   kernel_size=(3, 3),
                                                   bias=True)

        self.encoder_2_convlstm = MarvConvLSTMCell(input_channel=nf,
                                                   hidden_channel=nf,
                                                   kernel_size=(3, 3),
                                                   bias=True)

        self.decoder_1_convlstm = MarvConvLSTMCell(input_channel=nf,
                                                   hidden_channel=nf,
                                                   kernel_size=(3, 3),
                                                   bias=True)

        self.decoder_2_convlstm = MarvConvLSTMCell(input_channel=nf,
                                                   hidden_channel=nf,
                                                   kernel_size=(3, 3),
                                                   bias=True)

        # self.decoder_CNN = nn.Conv3d(in_channels=nf,
        #                              out_channels=1,
        #                              kernel_size=(1, 3, 3),
        #                              padding=(0, 1, 1))

        self.decoder_CNN = nn.Sequential(
            #nn.BatchNorm3d(nf),
            nn.Conv3d(in_channels=nf,
                      out_channels=1,
                      kernel_size=(1, 3, 3),
                      padding=(0, 1, 1)),
            nn.Hardtanh(min_val=0, max_val=1)

        )

        # # Outputlayer for N(I_t) = O_T for all t<T
        # self.decoder_CNN = nn.Sequential(
        #     # nn.BatchNorm3d(nf),
        #     nn.Conv3d(in_channels=nf,
        #               out_channels=1,
        #               kernel_size=(10, 3, 3),
        #               padding=(0, 1, 1)),
        #     nn.Hardtanh(min_val=0, max_val=1)
        #
        # )

    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        outputs = []

        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here

            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here

        # encoder_vector
        encoder_vector = h_t2

        h_t3, c_t3 = h_t, c_t
        h_t4, c_t4 = h_t2, c_t2

        # decoder
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=encoder_vector,
                                                 cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here

            h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
                                                 cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here

            encoder_vector = h_t4
            outputs += [h_t4]  # predictions
        #print(outputs)
        outputs = torch.stack(outputs, 1)
        #print(outputs.size())
        outputs = outputs.permute(0, 2, 1, 3, 4)
        #print('Dimensionen N_LSTM Output:', outputs.size())
        outputs = self.decoder_CNN(outputs)
        #print('Dimensionen N_LSTM Output:',outputs.size())
        #outputs = torch.nn.Hardtanh(min_val=0, max_val=1)(outputs)
        outputs = outputs.squeeze()
        #print('Dimensionen N_LSTM Output:', outputs.size())
        return outputs

    def forward(self, x, future_seq=0, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """
        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)

        return outputs