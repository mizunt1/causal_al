from utils import show_uncertainties
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plotting_function(data, target, data_test,
                      target_test,  data_train,
                      target_train, model, predicted=False, logits=False):
    fig, axs = plt.subplots(3,1)
    # all data
    if logits:
        preds = (model(data) > 0)*1
    else:
        preds = torch.argmax(model(data), axis=1)
    if predicted: 
        to_plot = preds
    else:
        to_plot = target
    correct = ((preds - target).abs() < 1e-2)
    colours = ['green' if item else 'red' for item in to_plot]
    axs[0].scatter(data[:,0].cpu().detach().numpy(), data[:,1].cpu().detach().numpy(), color=colours)
    axs[0].set_title('all data')

    # test data
    if logits:
        preds = (model(data_test) > 0)*1
    else:
        preds = torch.argmax(model(data_test), axis=1)
        
    if predicted: 
        to_plot = preds
    else:
        to_plot = target_test

    correct = ((preds - target_test).abs() < 1e-2)
    colours = ['green' if item else 'red' for item in to_plot]
    axs[1].scatter(data_test[:,0].cpu().detach().numpy(),
                   data_test[:,1].cpu().detach().numpy(), color=colours)
    axs[1].set_title('all test')
    # train data
    if logits:
        preds = (model(data_train) > 0)*1
    else:
        preds = torch.argmax(model(data_train), axis=1)        
    if predicted: 
        to_plot = preds
    else:
        to_plot = target_train

    correct = ((preds - target_train).abs() < 1e-2)
    colours = ['green' if item else 'red' for item in to_plot]
    axs[2].scatter(data_train[:,0].cpu().detach().numpy(),
                   data_train[:,1].cpu().detach().numpy(), color=colours)
    axs[2].set_title('all train')
    for ax in axs.flat:
        ax.set(xlabel='non causal', ylabel='causal')
        ax.axhline(y = 0, color = 'b', linestyle = '-') 
    for ax in axs.flat:
        ax.label_outer()

    # plt.show()
    return fig

def plotting_uncertainties(data, data_train, data_test, target_test, model, scorer, train_indices, model_reg,
                           prop, show=False):
    train_indices = np.where(np.array(train_indices)>0)[0]
    scores, preds = show_uncertainties(data, model, scorer, model_reg)
    min_score = np.min(scores)
    scores = scores - min_score
    scores = scores / np.max(scores)
    fig, axs = plt.subplots(3,1)
    # all data
    majority_idx = int(prop*len(data))

    data_ones = [dat.cpu().detach().numpy() for dat, pred in zip(data, preds) if pred]
    scores_ones = [score for score, pred in zip(scores, preds) if pred]

    data_zeros = [dat.cpu().detach().numpy() for dat, pred in zip(data, preds) if not pred]
    scores_zeros = [score for score, pred in zip(scores, preds) if not pred]
    
    data_ones = np.stack(data_ones)
    axs[0].scatter(data_ones[:,0], data_ones[:,1], c=scores_ones, cmap='winter')
    data_zeros = np.stack(data_zeros)
    axs[0].scatter(data_zeros[:,0], data_zeros[:,1], c=scores_zeros, cmap='autumn')
    data_cpy = data.detach().cpu().numpy()
    axs[0].set_title('all data (pool)')
    #axs[0].scatter(data_cpy[:majority_idx,0], data_cpy[:majority_idx,1], marker='2')
    axs[0].scatter(data_cpy[majority_idx:, 0],data_cpy[majority_idx:, 1], marker='x', color='black')
    
    scores, preds = show_uncertainties(data_test, model, scorer, model_reg)
    min_score = np.min(scores)
    scores = scores - min_score
    scores = scores / np.max(scores)

    data_ones = [dat.cpu().detach().numpy() for dat, pred in zip(data_test, preds) if pred]
    scores_ones = [score for score, pred in zip(scores, preds) if pred]

    data_zeros = [dat.cpu().detach().numpy() for dat, pred in zip(data_test, preds) if not pred]
    scores_zeros = [score for score, pred in zip(scores, preds) if not pred]
    data_ones = np.stack(data_ones)
    axs[1].scatter(data_ones[:,0], data_ones[:,1], c=scores_ones, cmap='winter')
    data_zeros = np.stack(data_zeros)
    axs[1].scatter(data_zeros[:,0], data_zeros[:,1], c=scores_zeros, cmap='autumn')
    axs[1].set_title('all test')
    data_test =  data_test.detach().cpu().numpy()
    axs[1].scatter(data_test[:len(data_test) - majority_idx, 0], data_test[:len(data_test) - majority_idx, 1], marker='2', color='black')
    
    num_correct = ((preds - target_test.cpu().detach().numpy())==0).sum()
    accuracy = num_correct / (len(target_test))

    print(accuracy)

    
    # train data
    scores, preds = show_uncertainties(data_train, model, scorer, model_reg)
    min_score = np.min(scores)
    scores = scores - min_score
    scores = scores / np.max(scores)

    preds = torch.argmax(model(data_train), axis=1)
    data_ones = [dat.cpu().detach().numpy() for dat, pred in zip(data_train, preds) if pred]
    scores_ones = [score for score, pred in zip(scores, preds) if pred]

    data_zeros = [dat.cpu().detach().numpy() for dat, pred in zip(data_train, preds) if not pred]
    scores_zeros = [score for score, pred in zip(scores, preds) if not pred]
    data_ones = np.stack(data_ones)
    axs[2].scatter(data_ones[:,0], data_ones[:,1], c=scores_ones, cmap='winter')

    data_zeros = np.stack(data_zeros)

    

    axs[2].scatter(data_zeros[:,0], data_zeros[:,1], c=scores_zeros, cmap='autumn')
    axs[2].set_title('train (selected)')
    try:
        train_maj_data = np.stack([data.detach().cpu().numpy() for data, index in zip(data_train, train_indices) if index < majority_idx])
        # axs[2].scatter(train_maj_data[:,0], train_maj_data[:,1], marker="2", color='black')
    except:
        print('no maj data selected from pool set')
    try:
        train_min_data = np.stack([data.detach().cpu().numpy() for data, index in zip(data_train, train_indices) if index > majority_idx])
        axs[2].scatter(train_min_data[:,0], train_min_data[:,1], marker="x", color='black')
    except:
        print('no min data selected from pool set')

        
    for ax in axs.flat:
        ax.set(ylabel='causal')
        ax.axhline(y = 0, color = 'b', linestyle = '-') 
    if show:
        plt.show()
    return fig
