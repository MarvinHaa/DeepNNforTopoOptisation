import torch
import torch.nn as nn
import torch.optim as optim

import os
import time as tm

from sklearn.model_selection import train_test_split

from Neural_Networks.LSTM.LSTM_classes import Dataset, EncoderDecoderPhiLSTM


DATA_PATH = 'data/'


def tic():
    global _start_time
    _start_time = tm.time()

def tac():
    t_sec = round(tm.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec))



# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#device = "cpu"
torch.backends.cudnn.benchmark = True

torch.manual_seed(42)
torch.cuda.manual_seed(42)

print("Device:", device)
# Parameters
batch_size = 77
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 1}
max_epochs = 3

in_seq = 5
out_seq = 10

# Datasets
datalist = os.listdir(DATA_PATH)

train_data_list, validation_data_list = train_test_split(datalist, test_size=0.33, random_state=42)

extended_train_data_list = []

train_labels = {}
for seq_id in train_data_list:
    seq_tensor = torch.load(DATA_PATH + seq_id)
    for i in range((101 // in_seq) - 2):
        label, _ = seq_id.rsplit('.', 1)
        new_label_id = label + "_" + str(i)
        train_labels[new_label_id] = torch.load(DATA_PATH +seq_id)[(i+1)*in_seq:(i+1)*in_seq + out_seq, 0, :, :]
        extended_train_data_list.append(new_label_id)

extended_validation_data_list = []

validation_labels = {}
for seq_id in validation_data_list:
    seq_tensor = torch.load(DATA_PATH + seq_id)
    for i in range((101 // in_seq) - 2):
        label, _ = seq_id.rsplit('.', 1)
        new_label_id = label + "_" + str(i)

        validation_labels[new_label_id] = torch.load(DATA_PATH +seq_id)[(i+1)*in_seq:(i+1)*in_seq + out_seq, 0, :, :]
        extended_validation_data_list.append(new_label_id)


NSamples = len(extended_train_data_list)

partition = {'train': extended_train_data_list, 'validation': extended_validation_data_list}

# Generators
training_set = Dataset(partition['train'], train_labels, DATA_PATH)
training_generator = torch.utils.data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'], validation_labels, DATA_PATH)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

# Define Loss function
model = EncoderDecoderPhiLSTM(nf=9, in_chan=3)
model.to(torch.device(device))
criterion = nn.MSELoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters())

train_loss = []
valiation_loss = []

tic()
# Loop over epochs
for epoch in range(max_epochs):
    running_loss = 0.0
    epoch_train_loss = 0.0
    epoch_validation_loss = 0.0
    # Training
    for i, data in enumerate(training_generator,0):
        local_batch, local_labels = data
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(local_batch.float(), out_seq)

        loss = criterion(output, local_labels.float())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        epoch_train_loss += loss.item()

        print('TRAINING: Epoch %d Batch %d loss: %.3f' %
              (epoch + 1, i + 1, running_loss))


        running_loss = 0.0

    train_loss.append(epoch_train_loss)


    # Validation
    running_val_loss = 0.0
    with torch.set_grad_enabled(False):
        for i, data in enumerate(validation_generator):
            local_batch, local_labels = data

            # Transfer to GPU
            #local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            output = model(local_batch.float(), out_seq)

            loss = criterion(output, local_labels.float())
            running_val_loss += loss.item()
            epoch_validation_loss += loss.item()
            print('VALUATION: Epoch %d Batch %d loss: %.3f' %
                  (epoch + 1, i + 1, running_val_loss))


            running_val_loss = 0.0

    valiation_loss.append(epoch_validation_loss)




print('Finished Training')
tac()


torch.save(model.state_dict(), 'Neural_Networks/LSTM/model/ConvLSTM-model_3_9_9_9_9_1_sequence_5in_10out_3618Samples_77batch_100epochs.pth')
