import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from erm_failures import main
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import numpy as np
def pretty_print(*values):
    col_width = 13
    def format_val(v):
        if isinstance(v, float):
            v = "{:5.3f}".format(v)
        elif not isinstance(v, str):
            # if not string convert from array to string
            v = str(v)
        return v.ljust(col_width)

    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))

name = 'irm_2_partial'
#data_name = 'anti_corr'
#data_name = 'scm_i_two_env'
data_name = 'spu_corr'
irm = False
ideal = False
#proportions_of_env1 = [1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 8, 6, 4, 2, 0]
#proportions_of_env1 = [1500,1000,900,800, 600, 500, 300, 200, 100, 50, 20, 10, 5, 0]
proportions_of_env1 =  [1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0]

if ideal:
    proportions_of_env1 = [0.9]
parser = ArgumentParser()

args = parser.parse_args()
column_names = ["proportion", "train ac w", "test acc", "worst acc", "coef1", "coef2"]
df = pd.DataFrame(columns=column_names)
pretty_print("proportion", "train ac w", "test acc", "worst acc", "coef1", "coef2", "env_amounts")
for proportion in proportions_of_env1:
    args_dict = {'data': data_name, 'seed': 0, 'data_size_train': 1000, 'data_size_pool': 1000, 'data_size_test': 1000, 'num_models': 10, 'train_proportion': proportion, 'test_proportion': 0.5, 'num_env2': None, 'num_epochs': 3010, 'lr': 0.01, 'project_name': 'causal_al', 'non_lin_entangle': False, 'plot': False, 'score': 'ent', 'irm': irm, 'log_reg': True, 'l2_regularizer_weight': 0.001, 'lr_irm': 0.1, 'n_restarts': 1, 'penalty_anneal_iters': 100, 'penalty_weight': 10000.0, 'steps': 500, 'ideal':ideal}

    train_acc_w, test_acc, worst, coeff1, coeff2, env_amounts = main(**args_dict)
    df1 = pd.DataFrame({"proportion": proportion, "train ac w":train_acc_w.detach().cpu().item(), "test acc": test_acc.detach().cpu().item(), "worst acc": worst, "coef1": coeff1, "coef2":coeff2}, index=[0])
    df = pd.concat([df, df1])
    pretty_print(
        proportion,
        train_acc_w.detach().cpu().item(),
        test_acc.detach().cpu().item(),
        worst,
        coeff1,
        coeff2,
        env_amounts
    )

if not ideal:
    df.to_csv(name +'.csv')
