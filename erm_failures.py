import numpy as np
import torch 
import wandb
import json
from scm import scm_rand_corr, scm_anti_corr, scm_same, scm_f, scm_i, scm_i_sep, combine_envs, scm_i_one_env, combine_many_envs, standardise_envs
from models.drop_out_model import Model, train, test
from plotting import plotting_function, plotting_uncertainties
from models.model_reg import ModelReg, train_reg, test_reg
from irm.colored_mnist.main import MLP_IRM_SIMULATED, train, Log_reg
from sklearn.linear_model import LogisticRegression

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
         erm, irm, log_reg,
         l2_regularizer_weight,
         lr_irm, n_restarts,
         penalty_anneal_iters, penalty_weight, steps,ideal=False, print_=False):
    run = wandb.init(
        project=project_name,
        settings=wandb.Settings(start_method='fork')
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data == 'anti_corr':
        data_test, target_test = scm_anti_corr(
            seed+1, prop_1=(1-test_proportion),
            entangle=non_lin_entangle, flip_y=0, num_samples=data_size_test, device=device)
        if erm:
            # train data for erm is merged across environments
            data_train, target_train = scm_anti_corr(
                seed, prop_1=train_proportion, entangle=non_lin_entangle,
                flip_y=0, num_samples=data_size_train, device=device)
        else:
            # data from envs are separate for IRM
            data_train_e1, target_train_e1 = scm_anti_corr(
                seed, prop_1=1, entangle=non_lin_entangle,
                flip_y=0, num_samples=int(data_size_train*train_proportion), device=device)
            env1 = {'images': data_train_e1, 'labels': target_train_e1.float() }
            data_train_e2, target_train_e2 = scm_anti_corr(
                seed, prop_1=0, entangle=non_lin_entangle,
                flip_y=0, num_samples=int(data_size_train*(1-train_proportion)), device=device)
            env2 = {'images': data_train_e2, 'labels': target_train_e2.float()}
            env3 = {'images': data_test, 'labels': target_test.float()}
            envs = [env1, env2, env3]
            data_train = torch.cat((data_train_e1, data_train_e2))
            target_train = torch.cat((target_train_e1, target_train_e2))
        data_pool, target_pool = data_train, target_train

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


    if data == 'scm_i':
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
        data_e1, target_e1 = scm_i_sep(
            seed, num_samples_e1, device, environment=1, env1_noise=(0.3, 0.5), env2_noise=(0.4, 0.5))
        data_e2, target_e2 = scm_i_sep(seed, num_samples_e2, device, environment=2, env1_noise=(0.3, 0.5), env2_noise=(0.4, 0.5))
        data, target = combine_envs(data_e1, data_e2, target_e1, target_e2)
        data_e1_test, target_e1_test = scm_i_sep(seed+1, test_num_samples_e1, device, environment=1, env1_noise=(0.3, 0.5), env2_noise=(0.5, 0.1))
        data_e2_test, target_e2_test = scm_i_sep(seed+1, test_num_samples_e2, device, environment=2, env1_noise=(0.4, 0.5), env2_noise=(0.5, 0.1))
        if print_:
            print("num points in e1 test {}".format(len(target_e1_test)))
            print("num points in e2 test {}".format(len(target_e2_test)))
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

    if data == 'scm_i_many':
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
        num_samples_env1_test = 500
        num_samples_env2_test = 500
        num_samples_env3_test = 500

        data_env1, target_env1 = scm_i_one_env(
            seed+3, num_samples_env1_test, device, noise_x1=0.1, noise_x2=0.3)
        data_env2, target_env2 = scm_i_one_env(
            seed+4, num_samples_env2_test, device, noise_x1=0.1, noise_x2=0.4)
        data_env3, target_env3 = scm_i_one_env(
            seed+5, num_samples_env3_test, device, noise_x1=0.5, noise_x2=0.1)

        data_test, target_test = combine_many_envs(
            (data_env1, data_env2, data_env3), (target_env1, target_env2, target_env3))
        groups_test = [num_samples_env1_test, int(num_samples_env1_test+num_samples_env2_test)]
        env_test = {'images': data_test, 'labels': target_test.float()}
    mask = False
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
    
    if print_:
        print("non-causal coeff train: {:3.2f}, causal coeff: {:3.2f}".format(weights[0][0].detach().cpu().item(), weights[0][1].detach().cpu().item()))
        print("non-causal coeff train ideal: {:3.2f}, causal coeff: {:3.2f}".format(
        weights_ideal[0][0].detach().cpu().item(), weights_ideal[0][1].detach().cpu().item()))
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

        train_indices = [0 for i in range(len(data_train))]
        if score == 'reg':
            #plt_unc = plotting_uncertainties(data_pool, data_train, data_test,
            #                                 target_test, models, 'reg', train_indices, model_reg, proportion, show=True)
            pass
        else:
            pass
            #plt_unc = plotting_uncertainties(data_pool, data_train, data_test,
            #                                target_test, models, 'ent', train_indices, None, proportion, show=True)

    if not ideal:
        wandb.run.summary.update({"test acc": test_acc,
                                  "train_acc": train_acc})
    if ideal:
        return train_acc_weighted_ideal, test_acc_ideal, worst_ideal, weights_ideal[0][0].detach().cpu().item(), weights_ideal[0][1].detach().cpu().item()
    
    return train_acc_weighted, test_acc, worst, weights[0][0].detach().cpu().item(), weights[0][1].detach().cpu().item()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data', choices=['rand_corr', 'anti_corr','scm_f','same', 'scm_i', 'scm_i_many'], default='scm_i')
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
    parser.add_argument('--erm', action='store_true')
    parser.add_argument('--irm', action='store_true')
    parser.add_argument('--log_reg', action='store_true')

    parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
    parser.add_argument('--lr_irm', type=float, default=0.01)
    parser.add_argument('--n_restarts', type=int, default=1)
    parser.add_argument('--penalty_anneal_iters', type=int, default=100)
    parser.add_argument('--penalty_weight', type=float, default=10000.0)
    parser.add_argument('--steps', type=int, default=500)
    
    args = parser.parse_args()

    main(data=args.data, seed=args.seed,
         data_size_train=args.data_size_train,
         data_size_pool=args.data_size_pool, data_size_test=args.data_size_test,
         num_models=args.num_models, train_proportion=args.train_proportion,
         test_proportion=args.test_proportion, num_env2=args.num_env2,
         num_epochs=args.num_epochs, lr=args.lr, project_name=args.project_name,
         non_lin_entangle=args.non_lin_entangle, plot=args.plot, score=args.score,
         erm=args.erm, irm=args.irm, log_reg=args.log_reg,
         l2_regularizer_weight=args.l2_regularizer_weight,
         lr_irm=args.lr_irm, n_restarts=args.n_restarts,
         penalty_anneal_iters=args.penalty_anneal_iters,
         penalty_weight=args.penalty_weight, steps=args.steps)
