from utils import show_uncertainties
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



def plotting_function(data, target, data_test,
                      target_test,  data_train,
                      target_train, model):
    fig, axs = plt.subplots(3,1)
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

    #plt.show()
    return fig

def plotting_uncertainties(data, data_train, data_test, model, scorer):
    scores, preds = show_uncertainties(data, model, scorer)
    min_score = np.min(scores)
    scores = scores - min_score
    scores = scores / np.max(scores)
    fig, axs = plt.subplots(3,1)
    # all data
    data_ones = [dat.cpu().detach().numpy() for dat, pred in zip(data, preds) if pred]
    scores_ones = [score for score, pred in zip(scores, preds) if pred]

    data_zeros = [dat.cpu().detach().numpy() for dat, pred in zip(data, preds) if not pred]
    scores_zeros = [score for score, pred in zip(scores, preds) if not pred]
    data_ones = np.stack(data_ones)
    data_zeros = np.stack(data_zeros)

    axs[0].scatter(data_ones[:,0], data_ones[:,1], c=scores_ones, cmap='winter')
    axs[0].scatter(data_zeros[:,0], data_zeros[:,1], c=scores_zeros, cmap='autumn')
    axs[0].set_title('all data (pool)')
    
    
    scores, preds = show_uncertainties(data_test, model, scorer)
    min_score = np.min(scores)
    scores = scores - min_score
    scores = scores / np.max(scores)

    data_ones = [dat.cpu().detach().numpy() for dat, pred in zip(data_test, preds) if pred]
    scores_ones = [score for score, pred in zip(scores, preds) if pred]

    data_zeros = [dat.cpu().detach().numpy() for dat, pred in zip(data_test, preds) if not pred]
    scores_zeros = [score for score, pred in zip(scores, preds) if not pred]

    data_ones = np.stack(data_ones)
    data_zeros = np.stack(data_zeros)

    axs[1].scatter(data_ones[:,0], data_ones[:,1], c=scores_ones, cmap='winter')
    axs[1].scatter(data_zeros[:,0], data_zeros[:,1], c=scores_zeros, cmap='autumn')
    
    axs[1].set_title('all test')
    # train data
    scores, preds = show_uncertainties(data_train, model, scorer)
    min_score = np.min(scores)
    scores = scores - min_score
    scores = scores / np.max(scores)

    preds = torch.argmax(model(data_train), axis=1)
    data_ones = [dat.cpu().detach().numpy() for dat, pred in zip(data_train, preds) if pred]
    scores_ones = [score for score, pred in zip(scores, preds) if pred]

    data_zeros = [dat.cpu().detach().numpy() for dat, pred in zip(data_train, preds) if not pred]
    scores_zeros = [score for score, pred in zip(scores, preds) if not pred]

    data_ones = np.stack(data_ones)
    data_zeros = np.stack(data_zeros)
    axs[2].scatter(data_ones[:,0], data_ones[:,1], c=scores_ones, cmap='winter')
    axs[2].scatter(data_zeros[:,0], data_zeros[:,1], c=scores_zeros, cmap='autumn')

    axs[2].set_title('train (selected)')
    for ax in axs.flat:
        ax.set(xlabel='non causal', ylabel='causal')
        ax.axhline(y = 0, color = 'b', linestyle = '-') 
    for ax in axs.flat:
        ax.label_outer()
    plt.show()
    return fig
