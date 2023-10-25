import numpy as np
import torch.nn.functional as F
import torch 
import random
import wandb

from scm import scm_rand_corr, scm_anti_corr, scm_same
from scores import calc_score
from models.drop_out_model import Model, train, test
from plotting import plotting_function

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data', choices=['rand_corr', 'anti_corr','same'], default='anti_corr')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_size_train', type=int, default=8)
    parser.add_argument('--data_size_test', type=int, default=1000)
    parser.add_argument('--num_models', type=int, default=10) 
    parser.add_argument('--proportion', type=float, default=0.50)
    parser.add_argument('--num_epochs', type=int, default=3010)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--project_name', type=str, default='causal_al')
    parser.add_argument('--non_lin_entangle', action='store_true')
    
    
    args = parser.parse_args()
    run = wandb.init(
        project=args.project_name,
        settings=wandb.Settings(start_method='fork')
    )
    wandb.config.update(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.data == 'anti_corr':
        data, target = scm_anti_corr(
            args.seed, prop_1=args.proportion, entangle=args.non_lin_entangle, flip_y=0, num_samples=args.data_size_train, device=device)
        data_test, target_test = scm_anti_corr(
            args.seed+1, prop_1=(0.5), entangle=args.non_lin_entangle, flip_y=0, num_samples=args.data_size_test, device=device)
    if args.data == 'rand_corr':
        data, target = scm_rand_corr(
            args.seed, prop_1=args.proportion, entangle=args.non_lin_entangle, flip_y=0, num_samples=args.data_size_train)
        data_test, target_test = scm_rand_corr(
            args.seed+1, prop_1=0.5, entangle=args.non_lin_entangle, flip_y=0, num_samples=args.data_size_test)
    if args.data == 'same':
        data, target = scm_same(
            args.seed, prop_1=args.proportion, entangle=args.non_lin_entangle, flip_y=0, num_samples=args.data_size_train)
        data_test, target_test = scm_same(
            args.seed+1, prop_1=(1-args.proportion), entangle=args.non_lin_entangle, flip_y=0, num_samples=args.data_size_test)
    input_size = 2
    #data, target = scm.sample(split='train')
    #data_test, target_test = scm.sample(split='test')
    majority_data = int(np.floor(args.data_size_train*args.proportion))
    minority_data = args.data_size_train - majority_data
    models = Model(input_size, args.num_models)
    models.to(device)
    
    train_acc = train(args.num_epochs, models, data, target, args.lr, device, ensemble=False)
    test_acc = test(models, data_test, target_test, device, ensemble=False)
    
    print('test accuracy: {}'.format(test_acc))
    plotting_function(data, target, data_test,
                      target_test,  data,
                      target,  models)
