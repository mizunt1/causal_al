import numpy as np
from scores import mi_score, ent_score, reg_score
import torch.nn.functional as F
import torch

def show_uncertainties(data, model, score_name, model_reg):
    if score_name == 'mi':
        scorer = mi_score
    elif score_name == 'ent':
        scorer = ent_score
    elif score_name == 'reg':
        scorer = reg_score
    else:
        print('scorer not valid')
    if score_name == 'reg':
        preds = torch.argmax(model(data), dim=1).cpu().clone().detach().numpy()
        scores = reg_score(data, model_reg, 4, [1 for i in range(len(data))], 0.8, return_all_scores=True)
        return scores.cpu().detach().numpy(), preds
    else:
        model.train()
        preds = [F.softmax(model(data), dim=1).cpu().clone().detach().numpy() for _ in range(10)]
        scores = scorer(
            preds, 2, [1 for i in range(len(data))], 0.7, return_all_scores=True)
        preds = np.stack(preds)
        preds = np.sum(preds, axis=0)
        preds = np.argmax(preds, axis=1)
        return scores, preds
