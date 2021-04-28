import torch
from Neural_Networks.CNN.CNN_classes import phiDeepCNN
import matplotlib.pyplot as plt


model = phiDeepCNN(3,15)


test_tensor = torch.load('../LSTM/data/(0,-5000)_truncnorm_bridge2D_tensor_100_sequence.pt')

model.load_state_dict(torch.load('model/CNN-model_3_15_15_15_15_15_1_one_step_77_Batch_100_epoch.pth'))



for k in range(10):
    i = k*10
    input = test_tensor[i, :, :, :].unsqueeze(0).float()

    out = model(input)


    fig = plt.figure()
    plt.imshow(test_tensor[i, 0, :, :].detach().numpy())
    fig.suptitle('φ_'+ str(i*5))
    plt.show()
    #plt.savefig('plots/g5000_'+str(k) + '.jpg')


    fig = plt.figure()
    plt.imshow(out[0].detach().numpy())
    fig.suptitle("N("+'φ_'+ str(i*5)+')')
    #plt.savefig('plots/N100_pred_' + str(k) +'.jpg')
    plt.show()

    fig = plt.figure()
    plt.imshow(test_tensor[i+20, 0, :, :].detach().numpy())
    fig.suptitle('φ_'+ str(i*5+100))
    #plt.savefig('plots/N100_true_' + str(k) +'.jpg')
    plt.show()