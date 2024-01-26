import numpy as np
import torch.nn.functional as F
import torch 
import random
import wandb

from scm import scm_rand_corr, scm_anti_corr, scm_same, scm_f, scm_i, scm_i_sep, combine_envs
from scores import mi_score, ent_score, reg_score
from models.drop_out_model import Model, train, test
from models.model_reg import ModelReg, train_reg, test_reg
from plotting import plotting_function, plotting_uncertainties

def al_loop_reg(models, model_reg, data, target, data_test, target_test,
            n_largest, al_iters, lr, num_epochs, device, prop, wandb, score,
            log_int = 1000, random_ac=False, ensemble=False):
    majority_data = int(np.floor(data.shape[0]*prop))
    minority_data = data.shape[0] - majority_data
    mean_score_min = 0
    mean_score_maj = 0
    assert(al_iters*n_largest < len(data))
    pool_indices = [1 for i in range(len(data))]
    train_indices = [0 for i in range(len(data))]
    seed_set = False
    if seed_set:
        seed_amount = 10
        rand_seed = random.sample([i for i in range(len(data))], seed_amount)
        for idx in rand_seed:
            train_indices[int(idx)] = 1
            pool_indices[int(idx)] = 0
    for iter in range(al_iters):
        if random_ac:
            pool_idx = [idx for (idx, in_pool) in enumerate(pool_indices) if in_pool > 0 ]
            models.train()
            preds = F.softmax(models(data), dim=1).cpu().clone().detach().numpy()
            if score == 'mi':
                n_largest_idx, mean_score_maj, mean_score_min = mi_score(
                    preds, n_largest, train_indices, prop)
            else:  
                n_largest_idx, mean_score_maj, mean_score_min = ent_score(
                    preds, n_largest, train_indices, prop)

            n_largest_idx = random.sample(pool_idx, n_largest)
        else:
            models.train()
            preds = F.softmax(models(data), dim=1).cpu().clone().detach().numpy()
            n_largest_idx, mean_score_maj, mean_score_min = reg_score(
                data, model_reg, n_largest, train_indices, prop)
            # print(n_largest_idx)
        maj_sub_min = mean_score_maj - mean_score_min
        wandb.log({'step': iter, 'mean score maj': mean_score_maj,
                   'mean_score_min': mean_score_min, 'maj sub min': maj_sub_min})
        for idx in n_largest_idx:
            train_indices[int(idx)] = 1
            pool_indices[int(idx)] = 0
        data_pool = [data[idx] for idx, in_pool in enumerate(pool_indices) if in_pool > 0]
        data_train = [data[idx] for idx, in_train in enumerate(train_indices) if in_train > 0]
        target_pool = [target[idx] for idx, in_pool in enumerate(pool_indices) if in_pool > 0]
        target_train = [target[idx] for idx, in_train in enumerate(train_indices) if in_train > 0]
        data_pool, target_pool = torch.stack(data_pool).to(device), torch.stack(target_pool).to(device)
        data_train, target_train = torch.stack(data_train).to(device), torch.stack(target_train).to(device)
        train_acc = train(
            num_epochs, models, data_train, target_train,
            lr, device, log_interval=log_int, ensemble=ensemble)
        test_acc = test(models, data_test, target_test, device, ensemble=ensemble)
        print('Al iter: {} train acc: {}'.format(iter, train_acc))
        print('Al iter: {} test accuracy: {}'.format(iter, test_acc))
        print('Al iter: {} points in train set: {}'.format(iter, data_train.shape[0]))
        print('Al iter: {} points in pool set: {}'.format(iter, data_pool.shape[0]))
        train_acc_reg = train_reg(
            num_epochs, model_reg, data_train[:,0].unsqueeze(1), data_train[:,1].unsqueeze(1),
            lr, device, log_interval=log_int)
        test_acc_reg = test_reg(model_reg, data_test[:,0].unsqueeze(1), data_test[:,1].unsqueeze(1), device)
        print('reg model: Al iter: {} train mse: {}'.format(iter, train_acc_reg))
        print('reg model: Al iter: {} points in train set: {}'.format(iter, data_train.shape[0]))
        print('reg model: Al iter: {} points in pool set: {}'.format(iter, data_pool.shape[0]))
        
    print('final train size {}'.format(data_train.shape[0]))
    prop_maj = np.sum(
        np .where(np.asarray(train_indices) == 1)[0]<majority_data)/len(
            np.where(np.asarray(train_indices) ==1)[0])
    prop_minority = np.sum(
            np.where(np.asarray(train_indices) == 1)[0]>majority_data)/len(
            np.where(np.asarray(train_indices) ==1)[0])
    print('prop majority selected for train: {}'.format(prop_maj))
    print('prop minority selected for train: {}'.format(prop_minority))
    return train_acc, test_acc, prop_maj, prop_minority, data_train, target_train,  data_pool, target_pool, mean_score_maj, mean_score_min, train_indices

def al_loop(models, data, target, data_test, target_test,
            n_largest, al_iters, lr, num_epochs, device, prop, wandb, score,
            log_int = 1000, random_ac=False, ensemble=False, plot_iter=False):
    majority_data = int(np.floor(data.shape[0]*prop))
    minority_data = data.shape[0] - majority_data
    mean_score_min = 0
    mean_score_maj = 0
    assert(al_iters*n_largest < len(data))
    pool_indices = [1 for i in range(len(data))]
    train_indices = [0 for i in range(len(data))]
    seed_set = False
    if seed_set:
        seed_amount = 10
        rand_seed = random.sample([i for i in range(len(data))], seed_amount)
        for idx in rand_seed:
            train_indices[int(idx)] = 1
            pool_indices[int(idx)] = 0
    for iter in range(al_iters):
        if random_ac:
            pool_idx = [idx for (idx, in_pool) in enumerate(pool_indices) if in_pool > 0 ]
            models.train()
            preds = [F.softmax(models(data), dim=1).cpu().clone().detach().numpy() for _ in range(10)]
            if score == 'mi':
                n_largest_idx, mean_score_maj, mean_score_min = mi_score(
                    preds, n_largest, train_indices, prop)
            else: 
                n_largest_idx, mean_score_maj, mean_score_min = ent_score(
                    preds, n_largest, train_indices, prop)

            n_largest_idx = random.sample(pool_idx, n_largest)
        else:
            models.train()
            preds = [F.softmax(models(data), dim=1).cpu().clone().detach().numpy() for _ in range(10)]
            if score == 'mi':
                n_largest_idx, mean_score_maj, mean_score_min = mi_score(
                    preds, n_largest, train_indices, prop)
            else: 
                n_largest_idx, mean_score_maj, mean_score_min = ent_score(
                    preds, n_largest, train_indices, prop)
            # print(n_largest_idx)
        maj_sub_min = mean_score_maj - mean_score_min


        wandb.log({'step': iter, 'mean score maj': mean_score_maj,
                   'mean_score_min': mean_score_min, 'maj sub min': maj_sub_min})
        for idx in n_largest_idx:
            train_indices[int(idx)] = 1
            pool_indices[int(idx)] = 0
        data_pool = [data[idx] for idx, in_pool in enumerate(pool_indices) if in_pool > 0]
        data_train = [data[idx] for idx, in_train in enumerate(train_indices) if in_train > 0]
        target_pool = [target[idx] for idx, in_pool in enumerate(pool_indices) if in_pool > 0]
        target_train = [target[idx] for idx, in_train in enumerate(train_indices) if in_train > 0]
        data_pool, target_pool = torch.stack(data_pool).to(device), torch.stack(target_pool).to(device)
        data_train, target_train = torch.stack(data_train).to(device), torch.stack(target_train).to(device)

        train_acc = train(
            num_epochs, models, data_train, target_train,
            lr, device, log_interval=log_int, ensemble=ensemble)
        test_acc = test(models, data_test, target_test, device, ensemble=ensemble)
        print('Al iter: {} test accuracy: {}'.format(iter, test_acc))
        print('Al iter: {} points in train set: {}'.format(iter, data_train.shape[0]))
        print('Al iter: {} points in pool set: {}'.format(iter, data_pool.shape[0]))
        if plot_iter: 
            plt_unc = plotting_uncertainties(data_pool, data_train, data_test, target_test,
                                             models, args.score, train_indices, None, prop,
                                             show=False)
            image = wandb.Image(plt_unc)
            wandb.log({"fig unc": image})
    print('final train size {}'.format(data_train.shape[0]))
    prop_maj = np.sum(
        np.where(np.asarray(train_indices) == 1)[0]<majority_data)/len(
            np.where(np.asarray(train_indices) ==1)[0])
    prop_minority = np.sum(
            np.where(np.asarray(train_indices) == 1)[0]>majority_data)/len(
            np.where(np.asarray(train_indices) ==1)[0])
    print('prop majority selected for train: {}'.format(prop_maj))
    print('prop minority selected for train: {}'.format(prop_minority))
    return train_acc, test_acc, prop_maj, prop_minority, data_train, target_train,  data_pool, target_pool, mean_score_maj, mean_score_min, train_indices
    
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data', choices=['rand_corr', 'anti_corr', 'scm_f', 'same', 'scm_i'], default='anti_corr')
    parser.add_argument('--score', choices=['mi', 'ent', 'reg'], default='ent')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pool_size', type=int, default=2000)
    parser.add_argument('--test_size', type=int, default=2000)
    parser.add_argument('--n_largest', type=int, default=2)
    parser.add_argument('--al_iters', type=int, default=4) 
    parser.add_argument('--num_models', type=int, default=10) 
    parser.add_argument('--proportion', type=float, default=0.95)
    parser.add_argument('--num_epochs', type=int, default=3010)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--project_name', type=str, default='causal_al')
    parser.add_argument('--non_lin_entangle', action='store_true')
    parser.add_argument('--rand_ac', action='store_true')
    parser.add_argument('--standard_train', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--plot_iter', action='store_true')
    
    
    args = parser.parse_args()
    run = wandb.init(
        project=args.project_name,
        settings=wandb.Settings(start_method='fork')
    )
    wandb.config.update(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.data == 'anti_corr':
        data, target = scm_anti_corr(
            args.seed, prop_1=args.proportion, entangle=args.non_lin_entangle, flip_y=0, num_samples=args.pool_size, device=device)
        data_test, target_test = scm_anti_corr(
            args.seed+1, prop_1=(1-args.proportion), entangle=args.non_lin_entangle, flip_y=0, num_samples=args.test_size, device=device)
    if args.data == 'rand_corr':
        data, target = scm_rand_corr(
            args.seed, prop_1=args.proportion, entangle=args.non_lin_entangle, flip_y=0, num_samples=args.pool_size)
        data_test, target_test = scm_rand_corr(
            args.seed+1, prop_1=(1-args.proportion), entangle=args.non_lin_entangle, flip_y=0, num_samples=args.test_size)
    if args.data == 'same':
        data, target = scm_same(
            args.seed, prop_1=args.proportion, entangle=args.non_lin_entangle, flip_y=0, num_samples=args.pool_size)
        data_test, target_test = scm_same(
            args.seed+1, prop_1=(1-args.proportion), entangle=args.non_lin_entangle, flip_y=0, num_samples=args.test_size)
    if args.data == 'scm_f':
        data, target = scm_f(args.seed, args.proportion, args.pool_size, device)
        data_test, target_test = scm_f(args.seed +1, args.proportion, args.test_size, device)

    if args.data == 'scm_i':
        if args.proportion < 1:
            num_samples_e1 = args.proportion*args.num_samples
            test_num_samples_e1 = 1-args.proportion
        else:
            num_samples_e1 = args.proportion
            test_num_samples_e1 = args.proportion
        num_samples_e2 = args.num_samples - num_samples_e1
        test_num_samples_e2 = args.test_size - test_num_samples_e1

        data_e1, target_e1 = scm_i_sep(args.seed, num_samples_e1, device, environment=1)
        data_e2, target_e2 = scm_i_sep(args.seed, num_samples_e2, device, environment=2)
        data, target = combine_envs(data_e1, data_e2, target_e1, target_e2)
        data_e1_test, target_e1_test = scm_i_sep(args.seed, test_num_samples_e1, args.pool_size, device, environment=1)
        data_e2_test, target_e2_test = scm_i_sep(args.seed, test_num_samples_e2, args.pool_size, device, environment=2)
        data_test, target_test = combine_envs(data_e1_test, data_e2_test, target_e1_test, target_e2_test)
    input_size = 2
    
    #data, target = scm.sample(split='train')
    #data_test, target_test = scm.sample(split='test')
    majority_data = int(np.floor(args.pool_size*args.proportion))
    minority_data = args.pool_size - majority_data
    models = Model(input_size, args.num_models)
    models.to(device)
    if args.standard_train:
        print("standard train trains with all of the pool data labelled")
        train_acc = train(args.num_epochs, models, data, target, args.lr, device, ensemble=False)
        test_acc = test(models, data_test, target_test, device, ensemble=False)
        print('test accuracy: {}'.format(test_acc))
        mean_score_min = 0
        mean_score_maj = 0

    else:
        if args.score == 'reg':
            input_size_reg = 1
            model_reg = ModelReg(input_size_reg)
            model_reg.to(device)
            train_acc, test_acc, prop_maj, prop_minority,\
                data_train, target_train,  data_pool, target_pool,\
                mean_score_maj, mean_score_min = al_loop_reg(
                models, model_reg, data, target, data_test, target_test,
                args.n_largest, args.al_iters, args.lr, args.num_epochs, device,
                args.proportion, run, args.score, random_ac=args.rand_ac)

        else:
            train_acc, test_acc, prop_maj, prop_minority, \
                data_train, target_train,  data_pool, target_pool,\
                mean_score_maj, mean_score_min, train_indices = al_loop(
                models, data, target, data_test, target_test,
                args.n_largest, args.al_iters, args.lr, args.num_epochs, device,
                args.proportion, run, args.score, random_ac=args.rand_ac, plot_iter=args.plot_iter)
    if args.plot:
        if args.standard_train:
            data_train = data
            target_train = target
        plt = plotting_function(data, target, data_test,
                                target_test,  data_train,
                                target_train,  models)
        image = wandb.Image(plt)
        wandb.log({"fig": image})
        if args.score == 'reg':
            pass
        else:
            model_reg = None
        plt_unc = plotting_uncertainties(data, data_train, data_test, target_test, models, args.score, train_indices,
                                         model_reg, args.proportion,
                                         show=True)
        image = wandb.Image(plt_unc)
        wandb.log({"fig unc": image})
        print("random: ", args.rand_ac)
    if not args.standard_train:
        wandb.run.summary.update({"test acc": test_acc,
                                  "train_acc": train_acc,
                                  "mean score majority env": mean_score_maj,
                                  "mean score minority env": mean_score_min,
                                  "points in train set": data_train.shape[0],
                                  "points in pool set": data_pool.shape[0],
                                  "prop minority selected for train": prop_minority,
                                  "prop majority selected for train": prop_maj,
                                  "minority data size":minority_data})
    else:
        wandb.run.summary.update({"test acc": test_acc,
                                  "train_acc": train_acc})
                                          
