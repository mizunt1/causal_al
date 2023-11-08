import numpy as np
import torch
def reg_score(data, model_reg, n_largest, train_indices, prop, return_all_scores=False):
    out = model_reg(data[:,0].unsqueeze(1))
    score = torch.abs(out - data[:,1].unsqueeze(1))
    majority_data = int(np.floor(data[0].shape[0]*prop))
    minority_data = data[0].shape[0] - majority_data
    
    mean_score_maj = score[0:majority_data].mean()
    mean_score_min = score[majority_data:].mean()
    
    print('mean score majority data: {}'.format(score[0:majority_data].mean()))
    print('mean score minority data: {}'.format(score[majority_data:].mean()))
    #means = preds.mean(axis=0).detach().numpy().squeeze(1)
    #sd = preds.std(axis=0).detach().numpy().squeeze(1)
    score_masked = [score_val[0] if idx < 1 else -10000 for score_val, idx in zip(score.cpu().detach().numpy(), train_indices)]

    score_labeled = np.vstack([score_masked, [i for i in range(len(score))]]) 
    score_sorted = np.argsort(score_masked)
    score_selected = score_labeled[:,score_sorted]
    score_largest = score_selected[1,n_largest*-1:]
    if return_all_scores:
        return score
    else:
        return score_largest, mean_score_maj, mean_score_min

def logistic_entropy(p):
    term1 = -p*np.where(p< 1e-23, p, np.log(p))
    q = 1 -p
    term2 = -q*np.where(q< 1e-23, q, np.log(q))
    return term1 + term2

def mi_score(preds, n_largest, train_indices, prop, return_all_scores=False):
    # finds the max entropy points from predictions on
    # the whole dataset, and removes items which are no longer
    # in the poolset. 
    # returns index of data with respect to whole dataset

    # majority data comes first.
    majority_data = int(np.floor(preds[0].shape[0]*prop))
    minority_data = preds[0].shape[0] - majority_data
    preds = np.stack(preds)
    log_preds = logistic_entropy(preds)
    variance_within = np.mean(np.sum(log_preds, axis=2), axis=0)
    # variance within a model
    variance_step = np.argmax(preds, axis=2)
    variance_step2 = np.sum(variance_step, axis=0)/preds.shape[0]
    variance_between = logistic_entropy(variance_step2)
    #variance between 10 models for each point
    score = variance_between - variance_within

    mean_score_maj = score[0:majority_data].mean()
    mean_score_min = score[majority_data:].mean()
    print('mean score majority data: {}'.format(score[0:majority_data].mean()))
    print('mean score minority data: {}'.format(score[majority_data:].mean()))
    #means = preds.mean(axis=0).detach().numpy().squeeze(1)
    #sd = preds.std(axis=0).detach().numpy().squeeze(1)
    score_masked = [score_val if idx < 1 else -10000. for score_val, idx in zip(score, train_indices)]
    score_labeled = np.vstack([score_masked, [i for i in range(len(score))]]) 
    score_sorted = np.argsort(score_masked)
    score_selected = score_labeled[:,score_sorted]
    score_largest = score_selected[1,n_largest*-1:]
    if return_all_scores:
        return score
    else:
        return score_largest, mean_score_maj, mean_score_min

def ent_score(preds, n_largest, train_indices, prop, return_all_scores=False):
    majority_data = int(np.floor(preds[0].shape[0]*prop))
    minority_data = preds[0].shape[0] - majority_data
    preds = np.stack(preds)
    variance_step = np.argmax(preds, axis=2)
    variance_step2 = np.sum(variance_step, axis=0)/preds.shape[0]
    score = logistic_entropy(variance_step2)
    mean_score_maj = score[0:majority_data].mean()
    mean_score_min = score[majority_data:].mean()
    print('mean score majority data: {}'.format(score[0:majority_data].mean()))
    print('mean score minority data: {}'.format(score[majority_data:].mean()))
    #means = preds.mean(axis=0).detach().numpy().squeeze(1)
    #sd = preds.std(axis=0).detach().numpy().squeeze(1)
    score_masked = [score_val if idx < 1 else -10000. for score_val, idx in zip(score, train_indices)]
    score_labeled = np.vstack([score_masked, [i for i in range(len(score))]]) 
    score_sorted = np.argsort(score_masked)
    score_selected = score_labeled[:,score_sorted]
    score_largest = score_selected[1,n_largest*-1:]
    if return_all_scores:
        return score
    else:
        return score_largest, mean_score_maj, mean_score_min


def calc_score_debug(preds):
    # finds the max entropy points from predictions on
    # the whole dataset, and removes items which are no longer
    # in the poolset. 
    # returns index of data with respect to whole dataset
    preds = np.stack(preds)
    preds_step = preds + (1e-23)*(preds==0)
    log_preds = -preds*np.log(preds_step)
    variance_within = np.mean(np.sum(log_preds, axis=2), axis=0)
    # variance within a model
    variance_step = np.sum(np.argmax(preds, axis=2), axis=0)/preds.shape[1]
    variance_step2 = variance_step + 1e-23*(variance_step==0)
    alternate_term = 1 - variance_step
    alternate_term = alternate_term + 1e-23*(alternate_term==0)
    variance_between = -1*(variance_step* np.log(variance_step2) + alternate_term*np.log(alternate_term))
    #variance between 10 models for each point
    score = variance_between - variance_within
    return variance_between.mean(), variance_within.mean(),  score.mean()
