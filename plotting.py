import matplotlib.pyplot as plt
import torch

def plotting_function(data, target, data_test,
                      target_test,  data_train,
                      target_train, model):
    fig, axs = plt.subplots(3,1)
    data_ = data.cpu().detach().numpy()
    target_ = target.cpu().detach().numpy()
    data_test_ = data_test.cpu().detach().numpy()
    target_test_ = target_test.cpu().detach().numpy()
    data_train_ = data_train.cpu().detach().numpy()
    target_train_ = target_train.cpu().detach().numpy()
    # all data
    preds = torch.argmax(model(data), axis=1)
    correct = ((preds - target).abs() < 1e-2)
    colours = ['green' if item else 'red' for item in preds]
    axs[0].scatter(data[:,0].cpu().detach().numpy(), data[:,1].cpu().detach().numpy(), color=colours)
    axs[0].set_title('all data')

    # test data
    preds = torch.argmax(model(data_test), axis=1)
    correct = ((preds - target_test).abs() < 1e-2)
    colours = ['green' if item else 'red' for item in preds]
    axs[1].scatter(data_test[:,0].cpu().detach().numpy(),
                   data_test[:,1].cpu().detach().numpy(), color=colours)
    axs[1].set_title('all test')
    # train data
    preds = torch.argmax(model(data_train), axis=1)
    correct = ((preds - target_train).abs() < 1e-2)
    colours = ['green' if item else 'red' for item in preds]
    axs[2].scatter(data_train[:,0].cpu().detach().numpy(),
                   data_train[:,1].cpu().detach().numpy(), color=colours)
    axs[2].set_title('all train')
    for ax in axs.flat:
        ax.set(xlabel='non causal', ylabel='causal')
        ax.axhline(y = 0, color = 'b', linestyle = '-') 
    for ax in axs.flat:
        ax.label_outer()

    plt.show()

