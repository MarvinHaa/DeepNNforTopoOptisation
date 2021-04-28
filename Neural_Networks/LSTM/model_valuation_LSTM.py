import torch
from Neural_Networks.LSTM.LSTM_classes import EncoderDecoderPhiLSTM


model = EncoderDecoderPhiLSTM(nf=9, in_chan=3)


model.load_state_dict(torch.load('model/ConvLSTM-model_3_9_9_9_9_1_sequence_5in_10out_3618Samples_77batch_100epochs.pth'))

test_tensor = torch.load('data/(0,-5000)_truncnorm_bridge2D_tensor_100_sequence.pt')


import matplotlib.pyplot as plt


for i in [2, 4, 6, 8]:
    input = test_tensor[i*5:(i+1)*5, :, :, :].unsqueeze(0).float()
    out = model(input,20)


    fig = plt.figure()
    plt.imshow(test_tensor[(i + 1) * 5, 0, :, :].detach().numpy())
    fig.suptitle('φ_'+ str((i + 1) * 5))
    plt.show()
    #plt.savefig('plots/LSTM_plus20_input_'+str((i + 1) * 5) + '.jpg')


    fig = plt.figure()
    plt.imshow(out[19].detach().numpy())
    fig.suptitle("N(" + 'φ_' + str((i + 1) * 5) + ')')
    #plt.savefig('plots/LSTM_plus20_pred_' + str((i + 1) * 5) +'.jpg')
    plt.show()

    fig = plt.figure()
    plt.imshow(test_tensor[(i+1)*5+19, 0, :, :].detach().numpy())
    fig.suptitle('φ_' + str((i + 1) * 5 + 20))
    #plt.savefig('plots/LSTM_plus20_true_' + str((i + 1) * 5 ) +'.jpg')
    plt.show()