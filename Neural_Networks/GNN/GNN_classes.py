import torch
import torch.nn as nn
from Neural_Networks.GNN.GraphDataset import GraphDataset
from torch.utils.data import DataLoader
import dgl
import dgl.nn as dglnn

import time




def tic():
    global _start_time
    _start_time = time.time()

def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec))

class GraphConvPhi(nn.Module):
    def __init__(self, in_feats, hid_feats):
        super().__init__()


        # 15hid-3Relu
        self.conv1 = dglnn.GraphConv(in_feats=in_feats, out_feats=hid_feats//2, norm='right', weight=True, bias=True, activation=None, allow_zero_in_degree=False)
        self.conv2 = dglnn.GraphConv(in_feats=hid_feats//2, out_feats=hid_feats, norm='right', weight=True, bias=True, activation=None, allow_zero_in_degree=False)
        self.conv3 = dglnn.GraphConv(in_feats=hid_feats, out_feats=hid_feats, norm='right', weight=True, bias=True, activation=None, allow_zero_in_degree=False)
        self.conv4 = dglnn.GraphConv(in_feats=hid_feats, out_feats=hid_feats, norm='right', weight=True, bias=True, activation=None, allow_zero_in_degree=False)
        self.conv5 = dglnn.GraphConv(in_feats=hid_feats, out_feats=hid_feats, norm='right', weight=True, bias=True, activation=None, allow_zero_in_degree=False)
        self.conv6 = dglnn.GraphConv(in_feats=hid_feats, out_feats=hid_feats, norm='right', weight=True, bias=True, activation=None, allow_zero_in_degree=False)
        self.conv7 = dglnn.GraphConv(in_feats=hid_feats, out_feats=hid_feats, norm='right', weight=True, bias=True, activation=None, allow_zero_in_degree=False)
        self.conv8 = dglnn.GraphConv(in_feats=hid_feats, out_feats=hid_feats, norm='right', weight=True, bias=True, activation=None, allow_zero_in_degree=False)
        self.conv9 = dglnn.GraphConv(in_feats=hid_feats, out_feats=hid_feats, norm='right', weight=True, bias=True, activation=None, allow_zero_in_degree=False)
        self.conv10 = dglnn.GraphConv(in_feats=hid_feats, out_feats=1, norm='right', weight=True, bias=True, activation=None, allow_zero_in_degree=False)



        self.Batchnorm = torch.nn.BatchNorm1d(hid_feats, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.htan = nn.Hardtanh(min_val=0, max_val=1)
        self.sig = nn.Sigmoid()
        self.Lrelu = nn.LeakyReLU()

    def forward(self, graph):

        input = torch.cat((graph.ndata['value'], graph.ndata['u']), 1) # TO DO: Include positions #, graph.ndata['position']


        # encoder
        h = self.conv1(graph, input)
        h = self.conv2(graph, h)
        h = self.Lrelu(h)

        h = self.conv3(graph, h)
        h = self.Lrelu(h)

        h = self.conv4(graph, h)
        h = self.Lrelu(h)

        h = self.conv5(graph, h)
        h = self.conv6(graph, h)
        h = self.conv7(graph, h)


        # decoder
        h = self.conv8(graph, h)
        h = self.conv9(graph, h)
        h = self.sig(h)
        h = self.conv10(graph, h)

        return h



def _collate_fn(batch):
    g = dgl.batch(batch, edata=None)
    label = g.ndata['labels']
    return g, label

def _collate_fn_u(batch):
    g = dgl.batch(batch, edata=None)
    label = g.ndata['labels']
    u_label = g.ndata['labels_u']
    return g, label, u_label


if __name__=='__main__':

    maxepochs = 40
    batchsize = 7
    input_channel = 3
    hidden_channel = 9

    tic()
    dataset = GraphDataset()


    model = GraphConvPhi(in_feats=input_channel, hid_feats=hidden_channel)

    opt = torch.optim.Adam(model.parameters())
    #opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    Loss_MSE = nn.MSELoss()
    #LOSS_BCE = nn.BCELoss()

    # create dataloaders
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, collate_fn=_collate_fn)


    #training
    train_loss_values = []
    for epoch in range(maxepochs):
        model.train()


        running_loss = 0.0
        i = 0

        for g, labels in dataloader:

            opt.zero_grad()

            logits = model(g).squeeze()

            MSE = Loss_MSE(logits, labels)
            #BCE = LOSS_BCE(logits, labels)


            #(MSE + BCE).backward()
            MSE.backward()
            opt.step()

            # print statistics
            running_loss += MSE.item() #+ BCE.item()
            print('TRAINING: Epoch %d Batch %d loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / (i + 1)))
            i +=1
        train_loss_values.append(running_loss / len(dataloader))

    tac()

    torch.save(model.state_dict(), 'model/' + 'Convelutional_Graph_NN_'+str(maxepochs)+'Epochs_'+str(batchsize)+'batchsize_'+str(input_channel)+'inputchannel_'+str(hidden_channel)+'.pth')



