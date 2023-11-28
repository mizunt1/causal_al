import numpy as np
import torch 
import wandb

from scm import scm_rand_corr, scm_anti_corr, scm_same, scm_f, scm_i, scm_i_sep, combine_envs
from models.drop_out_model import Model, train, test
from plotting import plotting_function, plotting_uncertainties
from models.model_reg import ModelReg, train_reg, test_reg
from irm.colored_mnist.main import MLP_IRM_SIMULATED, train
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data', choices=['rand_corr', 'anti_corr','scm_f','same', 'scm_i'], default='scm_i')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_size_train', type=int, default=1000)
    parser.add_argument('--data_size_pool', type=int, default=1000)
    parser.add_argument('--data_size_test', type=int, default=1000)
    parser.add_argument('--num_models', type=int, default=10) 
    parser.add_argument('--proportion', type=float, default=1.0)
    parser.add_argument('--num_env2', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=3010)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--project_name', type=str, default='causal_al')
    parser.add_argument('--non_lin_entangle', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--score', choices=['ent', 'reg'], default='ent')
    parser.add_argument('--erm', action='store_true')
    parser.add_argument('--irm', action='store_true')

    parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
    parser.add_argument('--lr_irm', type=float, default=0.01)
    parser.add_argument('--n_restarts', type=int, default=1)
    parser.add_argument('--penalty_anneal_iters', type=int, default=100)
    parser.add_argument('--penalty_weight', type=float, default=10000.0)
    parser.add_argument('--steps', type=int, default=500)
    
    args = parser.parse_args()
    run = wandb.init(
        project=args.project_name,
        settings=wandb.Settings(start_method='fork')
    )
    wandb.config.update(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.data == 'anti_corr':
        data_test, target_test = scm_anti_corr(
            args.seed+1, prop_1=(1-args.proportion),
            entangle=args.non_lin_entangle, flip_y=0, num_samples=args.data_size_test, device=device)
        if args.erm:
            # train data for erm is merged across environments
            data_train, target_train = scm_anti_corr(
                args.seed, prop_1=args.proportion, entangle=args.non_lin_entangle,
                flip_y=0, num_samples=args.data_size_train, device=device)
        else:
            # data from envs are separate for IRM
            data_train_e1, target_train_e1 = scm_anti_corr(
                args.seed, prop_1=1, entangle=args.non_lin_entangle,
                flip_y=0, num_samples=int(args.data_size_train*args.proportion), device=device)
            env1 = {'images': data_train_e1, 'labels': target_train_e1.float() }
            data_train_e2, target_train_e2 = scm_anti_corr(
                args.seed, prop_1=0, entangle=args.non_lin_entangle,
                flip_y=0, num_samples=int(args.data_size_train*(1-args.proportion)), device=device)
            env2 = {'images': data_train_e2, 'labels': target_train_e2.float()}
            env3 = {'images': data_test, 'labels': target_test.float()}
            envs = [env1, env2, env3]
            data_train = torch.cat((data_train_e1, data_train_e2))
            target_train = torch.cat((target_train_e1, target_train_e2))
        data_pool, target_pool = data_train, target_train

    if args.data == 'rand_corr':
        data_train, target_train = scm_rand_corr(
            args.seed, prop_1=args.proportion, entangle=args.non_lin_entangle,
            flip_y=0, num_samples=args.data_size_train)
        data_test, target_test = scm_rand_corr(
            args.seed+1, prop_1=0.5, entangle=args.non_lin_entangle, flip_y=0, num_samples=args.data_size_test)
        data_pool, target_pool = scm_rand_corr(
            args.seed, prop_1=args.proportion,
            entangle=args.non_lin_entangle, flip_y=0, num_samples=args.data_size_pool)

    if args.data == 'same':
        data_train, target_train = scm_same(
            args.seed, prop_1=args.proportion, entangle=args.non_lin_entangle, flip_y=0, num_samples=args.data_size_train)
        data_test, target_test = scm_same(
            args.seed+1, prop_1=(1-args.proportion), entangle=args.non_lin_entangle, flip_y=0, num_samples=args.data_size_test)
        data_pool, target_pool = scm_same(
            args.seed, prop_1=args.proportion, entangle=args.non_lin_entangle, flip_y=0, num_samples=args.data_size_pool)

    if args.data == 'scm_f':
        data_train, target_train = scm_f(args.seed, args.proportion, args.data_size_train, device)
        data_test, target_test = scm_f(args.seed+1, args.proportion, args.data_size_test, device)


    if args.data == 'scm_i':
        if args.proportion < 1:
            num_samples_e1 = round(args.proportion*args.data_size_train)
            test_num_samples_e1 = round(1-args.proportion)
        else:
            num_samples_e1 = round(args.proportion)
            test_num_samples_e1 = round(args.data_size_test - args.proportion)
        num_samples_e2 = round(args.data_size_train - num_samples_e1)
        test_num_samples_e2 = round(args.data_size_test - test_num_samples_e1)
        data_e1, target_e1 = scm_i_sep(args.seed, num_samples_e1, device, environment=1)
        data_e2, target_e2 = scm_i_sep(args.seed, num_samples_e2, device, environment=2)
        data, target = combine_envs(data_e1, data_e2, target_e1, target_e2)
        data_e1_test, target_e1_test = scm_i_sep(args.seed, test_num_samples_e1, device, environment=1)
        data_e2_test, target_e2_test = scm_i_sep(args.seed, test_num_samples_e2, device, environment=2)
        data_test, target_test = combine_envs(data_e1_test, data_e2_test, target_e1_test, target_e2_test)

        print("number of points in env 1: {}".format(len(target_e1)))
        env1 = {'images': data_e1, 'labels': target_e1.float()}
        print("number of points in env 2: {}".format(len(target_e2)))
        env2 = {'images': data_e2, 'labels': target_e2.float()}
        env3 = {'images': data_test, 'labels': target_test.float()}
        env_train = [env1, env2]
        env_test = env3
        data_train = torch.cat((data_e1, data_e2))
        target_train = torch.cat((target_e1, target_e2))
    data_pool, target_pool = data_train, target_train

    input_size = 2
    #data, target = scm.sample(split='train')
    #data_test, target_test = scm.sample(split='test')
    majority_data = int(np.floor(args.data_size_train*args.proportion))
    minority_data = args.data_size_train - majority_data
    models = MLP_IRM_SIMULATED()
    models.to(device)
    train_acc, test_acc = train(models, args, env_train, env_test, args.irm)

    if args.plot:
        logits = True
        plt = plotting_function(data_pool, target_pool, data_test,
                          target_test,  data_train,
                          target_train,  models, predicted=False, logits=logits)
        plt = plotting_function(data_pool, target_pool, data_test,
                          target_test,  data_train,
                          target_train,  models, predicted=True, logits=logits)

        image = wandb.Image(plt)
        wandb.log({"fig unc": image})
        
        train_indices = [0 for i in range(len(data_train))]
        if args.score == 'reg':
            #plt_unc = plotting_uncertainties(data_pool, data_train, data_test,
            #                                 target_test, models, 'reg', train_indices, model_reg, args.proportion, show=True)
            pass
        else:
            pass
            #plt_unc = plotting_uncertainties(data_pool, data_train, data_test,
            #                                target_test, models, 'ent', train_indices, None, args.proportion, show=True)
        
    wandb.run.summary.update({"test acc": test_acc,
                              "train_acc": train_acc})
