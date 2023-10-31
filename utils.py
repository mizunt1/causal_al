import numpy as np
from scores import mi_score, ent_score
import torch.nn.functional as F

def show_uncertainties(data, model, score_name):
    if score_name == 'mi':
        scorer = mi_score
    elif score_name == 'ent':
        scorer = ent_score
    else:
        print('scorer not valid')
        
    model.train()
    preds = [F.softmax(model(data), dim=1).cpu().clone().detach().numpy() for _ in range(10)]
    scores = scorer(
        preds, 2, [1 for i in range(len(data))], 0.7, return_all_scores=True)
    preds = np.stack(preds)
    preds = np.sum(preds, axis=0)
    preds = np.argmax(preds, axis=1)
    return scores, preds
