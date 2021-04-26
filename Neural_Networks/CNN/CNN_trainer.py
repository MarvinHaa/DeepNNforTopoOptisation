import torch
import torch.nn as nn
import torch.optim as optim
import os

from sklearn.model_selection import train_test_split

from Neural_Networks.CNN.CNN_classes import Dataset, phiDeepCNN

import time


def tic():
    global _start_time
    _start_time = time.time()

def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec))



DATA_PATH = 'data/'

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


# Parameters
params = {'batch_size': 14,
          'shuffle': True,
          'num_workers': 40}
max_epochs = 3

# Datasets
datalist = os.listdir(DATA_PATH)


extendet_datalist = []

# Label Data -> Input  phi_t+1 is lable for input phi_t
labels = {}
for ID in datalist:
    label_id = ID.rsplit('.', 1)[0]
    tensor = torch.load(DATA_PATH + '/' + ID)
    for i in range(99):
        extendet_label_id = label_id+'_'+str(i)
        labels[extendet_label_id] = tensor[i+1, 0]
        extendet_datalist.append(extendet_label_id)

# start timetracking
tic()

train_data_list, validation_data_list = train_test_split(extendet_datalist, test_size=0.33, random_state=42)

partition = {'train': train_data_list, 'validation': validation_data_list}

# Generators
training_set = Dataset(partition['train'], labels, DATA_PATH)
training_generator = torch.utils.data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'], labels, DATA_PATH)
validation_generator = torch.utils.data.DataLoader(validation_set, **params)

# Define Loss function
model = phiDeepCNN(3,15).to(device)
criterion = nn.MSELoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters())

train_loss_values = []
valuation_loss_values = []
# Loop over epochs
for epoch in range(max_epochs):
    running_loss = 0.0

    # Training
    for i, data in enumerate(training_generator,0):
        local_batch, local_labels = data

        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimizeth
        outputs = model(local_batch.float())

        loss = criterion(outputs, local_labels.float())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print('TRAINING: Epoch %d Batch %d loss: %.3f' %
              (epoch + 1, i + 1, running_loss/(i+1)))

    train_loss_values.append(running_loss / len(training_generator))

    # Validation
    running_val_loss = 0.0
    with torch.set_grad_enabled(False):
        for i, data in enumerate(validation_generator):
            local_batch, local_labels = data
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            outputs = model(local_batch.float())

            loss = criterion(outputs, local_labels.float())
            running_val_loss += loss.item()

            print('VALUATION: Epoch %d Batch %d loss: %.3f' %
                  (epoch + 1, i + 1, running_val_loss/(i+1)))
    valuation_loss_values.append(running_loss / len(validation_generator))

print('Finished Training')
tac()

#torch.save(model.state_dict(), 'model/CNN-model_3_15_15_15_15_15_15_15_15_1_5iterstep_100_sequence_14batch_50epoch.pth') #model/CNN-model_3_15_15_15_1_one_step_100_sequence_77batch_100epoch.pth

