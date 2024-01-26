import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch 
import wandb
import pandas as pd
import json
from scm import scm_rand_corr, scm_anti_corr, scm_same, scm_f, scm_i, scm_i_sep, combine_envs, scm_i_one_env, combine_many_envs, standardise_envs, scm_spurr_noise, scm_i_one_env_mechs_test1, scm_i_one_env_mechs_test2, scm_i_one_env_mechs_test3
from plotting import plotting_function, plotting_uncertainties
from models.model_reg import ModelReg, train_reg, test_reg
from irm.colored_mnist.main import MLP_IRM_SIMULATED, train, Log_reg
from sklearn.linear_model import LogisticRegression

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


def accuracy(predicted, target):
    return np.sum((predicted == target))/ len(target)

def group_accuracies(predicted, target, groups):
    # groups correspond to list of indices, where each item corresponds
    # to a starting point for each group.
    accuracies = []
    current_idx = 0
    for idx in range(len(groups)):
        predicted_section = predicted[current_idx: groups[idx]]

        target_section = target[current_idx: groups[idx]]
        current_idx = groups[idx]
        acc = accuracy(predicted_section, target_section)
        accuracies.append(acc)
    predicted_section = predicted[current_idx:]
    target_section = target[current_idx:]
    acc = accuracy(predicted_section, target_section)
    accuracies.append(acc)
    return np.array(accuracies)

def main(data, seed,
         data_size_train,
         data_size_pool, data_size_test,
         num_models, train_proportion,
         test_proportion, num_env2,
         num_epochs, lr, project_name,
         non_lin_entangle, plot, score,
         irm, log_reg,
         l2_regularizer_weight,
         lr_irm, n_restarts,
         penalty_anneal_iters, penalty_weight, steps,ideal=False, print_=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = wandb.init(
        project=project_name,
        settings=wandb.Settings(start_method='fork')
    )
    if data == 'test1_mech_complex':
        data_train, target_train = scm_i_one_env_mechs_test1(
            seed, data_size_train, device, False)
        data_test, target_test = scm_i_one_env_mechs_test1(
            seed+1, data_size_test, device, True)
        data_train = standardise_envs([data_train])
        data_test = standardise_envs([data_test])
        env_train = [{'images': data_train, 'labels': target_train.float()}]
        env_test = {'images': data_test, 'labels': target_test.float()}
        env_amounts = [data_size_train]
        groups_test = [data_size_test]
    if data == 'test2_mech_complex':
        data_train, target_train = scm_i_one_env_mechs_test2(
            seed, data_size_train, device, False)
        data_test, target_test = scm_i_one_env_mechs_test2(
            seed+1, data_size_test, device, True)
        data_train = standardise_envs([data_train])
        data_test = standardise_envs([data_test])
        env_train = [{'images': data_train, 'labels': target_train.float()}]
        env_test = {'images': data_test, 'labels': target_test.float()}
        env_amounts = [data_size_train]
        groups_test = [data_size_test]
    if data == 'test3_mech_complex':
        data_train, target_train = scm_i_one_env_mechs_test3(
            seed, data_size_train, device, False)
        data_test, target_test = scm_i_one_env_mechs_test3(
            seed+1, data_size_test, device, True)
        data_train = standardise_envs([data_train])
        data_test = standardise_envs([data_test])
        env_train = [{'images': data_train, 'labels': target_train.float()}]
        env_test = {'images': data_test, 'labels': target_test.float()}
        env_amounts = [data_size_train]
        groups_test = [data_size_test]


    if data == 'anti_corr':
        if train_proportion <= 1:
            num_samples_e1 = round(train_proportion*data_size_train)
        else:
            num_samples_e1 = round(train_proportion)
        if test_proportion <= 1:
            test_num_samples_e1 = round((test_proportion)*data_size_test)
        else:
            test_num_samples_e1 = round(test_proportion)
        num_samples_e2 = round(data_size_train - num_samples_e1)
        test_num_samples_e2 = int(data_size_test - test_num_samples_e1)
        # train
        data_train_e1, target_train_e1 = scm_anti_corr(
            seed, num_samples_e1, environment=1, entangle=non_lin_entangle,
            flip_y=0, device=device)
        data_train_e2, target_train_e2 = scm_anti_corr(
            seed, num_samples_e2, environment=2, entangle=non_lin_entangle,
            flip_y=0, device=device)
        data_train_e1, data_train_e2 = standardise_envs([data_train_e1, data_train_e2])
        env1 = {'images': data_train_e1, 'labels': target_train_e1.float() }
        env2 = {'images': data_train_e2, 'labels': target_train_e2.float()}
        env_amounts = [len(target_train_e1), len(target_train_e2)]
        # train
        data_test_e1, target_test_e1 = scm_anti_corr(
            seed, test_num_samples_e1, environment=1, entangle=non_lin_entangle,
            flip_y=0, device=device)
        data_test_e2, target_test_e2 = scm_anti_corr(
            seed, test_num_samples_e1, environment=2, entangle=non_lin_entangle,
            flip_y=0, device=device)

        data_test, target_test = combine_envs(data_test_e1, data_test_e2, target_test_e1, target_test_e2)

        data_test = standardise_envs([data_test])
        env_test = {'images': data_test, 'labels': target_test.float()}
        env_train = [env1, env2]
        groups_test = [test_num_samples_e2]

    if data == 'spu_corr':
        if train_proportion <= 1:
            num_samples_e1 = round(train_proportion*data_size_train)
        else:
            num_samples_e1 = round(train_proportion)
        if test_proportion <= 1:
            test_num_samples_e1 = round((test_proportion)*data_size_test)
        else:
            test_num_samples_e1 = round(test_proportion)
        num_samples_e2 = round(data_size_train - num_samples_e1)
        test_num_samples_e2 = int(data_size_test - test_num_samples_e1)
        # train
        data_train_e1, target_train_e1 = scm_spurr_noise(
            seed, num_samples_e1, noise=1.0, entangle=non_lin_entangle,
            flip_y=0, device=device)
        data_train_e2, target_train_e2 = scm_spurr_noise(
            seed, num_samples_e2, noise=0.1, entangle=non_lin_entangle,
            flip_y=0, device=device)
        data_train_e1, data_train_e2 = standardise_envs([data_train_e1, data_train_e2])
        env1 = {'images': data_train_e1, 'labels': target_train_e1.float() }
        env2 = {'images': data_train_e2, 'labels': target_train_e2.float()}
        env_amounts = [len(target_train_e1), len(target_train_e2)]
        # train
        data_test_e1, target_test_e1 = scm_spurr_noise(
            seed, num_samples_e1, noise=1.0, entangle=non_lin_entangle,
            flip_y=0, device=device)
        data_test_e2, target_test_e2 = scm_spurr_noise(
            seed, num_samples_e2, noise=0.1, entangle=non_lin_entangle,
            flip_y=0, device=device)

        data_test, target_test = combine_envs(data_test_e1, data_test_e2, target_test_e1, target_test_e2)

        data_test = standardise_envs([data_test])
        env_test = {'images': data_test, 'labels': target_test.float()}
        env_train = [env1, env2]
        groups_test = [test_num_samples_e2]

    if data == 'rand_corr':
        data_train, target_train = scm_rand_corr(
            seed, prop_1=train_proportion, entangle=non_lin_entangle,
            flip_y=0, num_samples=data_size_train)
        data_test, target_test = scm_rand_corr(
            seed+1, prop_1=0.5, entangle=non_lin_entangle, flip_y=0, num_samples=data_size_test)
        data_pool, target_pool = scm_rand_corr(
            seed, prop_1=train_proportion,
            entangle=non_lin_entangle, flip_y=0, num_samples=data_size_pool)

    if data == 'same':
        data_train, target_train = scm_same(
            seed, prop_1=train_proportion, entangle=non_lin_entangle, flip_y=0, num_samples=data_size_train)
        data_test, target_test = scm_same(
            seed+1, prop_1=(1-test_proportion), entangle=non_lin_entangle, flip_y=0, num_samples=data_size_test)
        data_pool, target_pool = scm_same(
            seed, prop_1=train_proportion, entangle=non_lin_entangle, flip_y=0, num_samples=data_size_pool)

    if data == 'scm_f':
        data_train, target_train = scm_f(seed, train_proportion, data_size_train, device)
        data_test, target_test = scm_f(seed+1, test_proportion, data_size_test, device)


    if data == 'scm_i_two_env':
        if train_proportion <= 1:
            num_samples_e1 = round(train_proportion*data_size_train)
        else:
            num_samples_e1 = round(train_proportion)
        if test_proportion <= 1:
            test_num_samples_e1 = round((test_proportion)*data_size_test)
        else:
            test_num_samples_e1 = round(test_proportion)
        num_samples_e2 = round(data_size_train - num_samples_e1)
        test_num_samples_e2 = int(data_size_test - test_num_samples_e1)
        # generate train data
        data_e1, target_e1 = scm_i_sep(
            seed, num_samples_e1, device, environment=1, env1_noise=(1.0, 0.0), env2_noise=(None, None))
        data_e2, target_e2 = scm_i_sep(seed, num_samples_e2, device, environment=2, env1_noise=(None, None), env2_noise=(0.1, 0.5))
        data, target = combine_envs(data_e1, data_e2, target_e1, target_e2)
        # generate test data
        env_amounts = [len(target_e1), len(target_e2)]
        data_e1_test, target_e1_test = scm_i_sep(
            seed+1, test_num_samples_e1, device, environment=1, env1_noise=(0.3, 0.5), env2_noise=(0.5, 0.1))
        data_e2_test, target_e2_test = scm_i_sep(
            seed+1, test_num_samples_e2, device, environment=2, env1_noise=(0.4, 0.5), env2_noise=(0.5, 0.1))
        if print_:
            print("num points in e1 test {}".format(len(target_e1_test)))
            print("num points in e2 test {}".format(len(target_e2_test)))
        mask = True
        if mask:
            data_e1[:, 0] = torch.rand(len(data_e1)) 
            data_e2[:, 0] = torch.rand(len(data_e2))
            

        data_test, target_test = combine_envs(data_e1_test, data_e2_test, target_e1_test, target_e2_test)
        data_train = torch.cat((data_e1, data_e2))
        target_train = torch.cat((target_e1, target_e2))
        means_train = data_train.mean(dim=0, keepdim=True)
        std_train = data_train.std(dim=0, keepdim=True)
        data_train = (data_train - means_train)/std_train
        means_test = data_test.mean(dim=0, keepdim=True)
        std_test = data_test.std(dim=0, keepdim=True)
        data_test = (data_test - means_test)/std_test
        env1 = {'images': (data_e1 - means_train)/std_train, 'labels': target_e1.float()}
        if print_:
            print("number of points in env 1: {}".format(len(target_e1)))
            print("number of points in env 2: {}".format(len(target_e2)))
        env2 = {'images': (data_e2 - means_train)/std_train, 'labels': target_e2.float()}
        env3 = {'images': data_test, 'labels': target_test.float()}
        env_train = [env1, env2]
        env_test = env3
        groups_test = [test_num_samples_e1]

    if data == 'scm_i_three_env':
        # create train env
        prop_env2 = ((1-train_proportion)/2)
        total_samples = 1000
        num_samples_env1 = round(total_samples*train_proportion)
        num_samples_env2 = round(total_samples*prop_env2)
        num_samples_env3 = int(total_samples - num_samples_env1 - num_samples_env2)
        data_env1, target_env1 = scm_i_one_env(
            seed, num_samples_env1, device, noise_x1=0.5, noise_x2=0.1)
        data_env2, target_env2 = scm_i_one_env(
            seed+1, num_samples_env2, device, noise_x1=0.1, noise_x2=0.4)
        data_env3, target_env3 = scm_i_one_env(
            seed+2, num_samples_env3, device, noise_x1=0.1, noise_x2=0.3)
        datas = standardise_envs((data_env1, data_env2, data_env3))
        env1 = {'images': datas[0], 'labels': target_env1.float()}
        env2 = {'images': datas[1], 'labels': target_env2.float()}
        env3 = {'images': datas[2], 'labels': target_env3.float()}
        groups_train = [num_samples_env1, int(num_samples_env1+num_samples_env2)]
        env_train = [env1, env2, env3]
        # create test envs
        num_samples_env1_test = 500
        num_samples_env2_test = 500
        num_samples_env3_test = 500

        data_env1, target_env1 = scm_i_one_env(
            seed+3, num_samples_env1_test, device, noise_x1=0.1, noise_x2=0.3)
        data_env2, target_env2 = scm_i_one_env(
            seed+4, num_samples_env2_test, device, noise_x1=0.1, noise_x2=0.4)
        data_env3, target_env3 = scm_i_one_env(
            seed+5, num_samples_env3_test, device, noise_x1=0.5, noise_x2=0.1)
        mask = True
        if mask:
            data_env1[:, 0] = torch.rand(len(data_train)) 
            data_env2[:, 0] = torch.rand(len(data_test))
            data_env3[:, 0] = torch.rand(len(data_test))

        data_test, target_test = combine_many_envs(
            (data_env1, data_env2, data_env3), (target_env1, target_env2, target_env3))
        groups_test = [num_samples_env1_test, int(num_samples_env1_test+num_samples_env2_test)]
        env_test = {'images': data_test, 'labels': target_test.float()}
    mask = True
    if mask:
        data_train[:,0] = torch.rand(len(data_train)) 
        data_test[:, 0] = torch.rand(len(data_test))
    input_size = 2
    #data, target = scm.sample(split='train')
    #data_test, target_test = scm.sample(split='test')
    majority_data = int(np.floor(data_size_train*train_proportion))
    minority_data = data_size_train - majority_data
    if log_reg:
        models = Log_reg()
    else:
        models = MLP_IRM_SIMULATED()
    models.to(device)
    if not ideal:
        train_acc, test_acc, worst, train_acc_weighted = train(models, 
                                                               env_train, env_test, irm, lr_irm,
                                                               groups_test,
                                                               l2_regularizer_weight,
                                                               penalty_weight, penalty_anneal_iters)
    
    if ideal:
        model2 = Log_reg()
        model2.to(device)
        train_acc_ideal, test_acc_ideal, worst_ideal, train_acc_weighted_ideal = train(
            model2, [env_test], env_test, irm, lr_irm,
            groups_test, l2_regularizer_weight,
            penalty_weight, penalty_anneal_iters)
        weights_ideal = model2.linear1.weight

    weights = models.linear1.weight
    
    if plot:
        logits = True
        plt = plotting_function(data_pool, target_pool, data_test,
                          target_test,  data_train,
                          target_train,  models, predicted=False, logits=logits)
        plt = plotting_function(data_pool, target_pool, data_test,
                          target_test,  data_train,
                          target_train,  models, predicted=True, logits=logits)

        image = wandb.Image(plt)
        wandb.log({"fig unc": image})

    if not ideal:
        wandb.run.summary.update({"test acc": test_acc,
                                  "train_acc": train_acc})
    if ideal:
        return train_acc_weighted_ideal, test_acc_ideal, worst_ideal, weights_ideal[0][0].detach().cpu().item(), weights_ideal[0][1].detach().cpu().item()
    
    return train_acc_weighted, test_acc, worst, weights[0][0].detach().cpu().item(), weights[0][1].detach().cpu().item(), env_amounts

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        '--data', choices=['rand_corr', 'anti_corr','scm_f','same', 'scm_i_two_env', 'scm_i_three_env', 'spu_corr', 'test1_mech_complex','test2_mech_complex', 'test3_mech_complex'], default='scm_i')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_size_train', type=int, default=1000)
    parser.add_argument('--data_size_pool', type=int, default=1000)
    parser.add_argument('--data_size_test', type=int, default=1000)
    parser.add_argument('--num_models', type=int, default=10) 
    parser.add_argument('--train_proportion', type=float, default=0.8)
    parser.add_argument('--test_proportion', type=float, default=0.2)
    parser.add_argument('--num_env2', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=3010)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--project_name', type=str, default='causal_al')
    parser.add_argument('--non_lin_entangle', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--score', choices=['ent', 'reg'], default='ent')
    parser.add_argument('--irm', action='store_true')
    parser.add_argument('--log_reg', action='store_true')

    parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
    parser.add_argument('--lr_irm', type=float, default=0.01)
    parser.add_argument('--n_restarts', type=int, default=1)
    parser.add_argument('--penalty_anneal_iters', type=int, default=100)
    parser.add_argument('--penalty_weight', type=float, default=10000.0)
    parser.add_argument('--steps', type=int, default=500)
    
    args = parser.parse_args()

train_acc_w, test_acc, worst, coeff1, coeff2, env_amount = main(
    data=args.data, seed=args.seed,
    data_size_train=args.data_size_train,
    data_size_pool=args.data_size_pool, data_size_test=args.data_size_test,
    num_models=args.num_models, train_proportion=args.train_proportion,
    test_proportion=args.test_proportion, num_env2=args.num_env2,
    num_epochs=args.num_epochs, lr=args.lr, project_name=args.project_name,
    non_lin_entangle=args.non_lin_entangle, plot=args.plot, score=args.score,
    irm=args.irm, log_reg=args.log_reg,
    l2_regularizer_weight=args.l2_regularizer_weight,
    lr_irm=args.lr_irm, n_restarts=args.n_restarts,
    penalty_anneal_iters=args.penalty_anneal_iters,
    penalty_weight=args.penalty_weight, steps=args.steps)

print_res = True
if print_res:
    column_names = ["proportion", "train ac w", "test acc", "worst acc", "coef1", "coef2"]
    df = pd.DataFrame(columns=column_names)
    pretty_print("proportion", "train ac w", "test acc", "worst acc", "coef1", "coef2", "env_amounts")
    df1 = pd.DataFrame({"proportion": 0, "train ac w":train_acc_w.detach().cpu().item(), "test acc": test_acc.detach().cpu().item(), "worst acc": worst, "coef1": coeff1, "coef2":coeff2}, index=[0])
    df = pd.concat([df, df1])
    pretty_print(
        0,
        train_acc_w.detach().cpu().item(),
        test_acc.detach().cpu().item(),
        worst,
        coeff1,
        coeff2,
        [0,0]
    )
    

