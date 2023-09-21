from cows_camels import CC
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch 
import random
import wandb

from data_gen import scm1_data, scm_ac_data, scm_band_data, scm1_noise_data, entangled_data, entangled_image_data
from scm import scm1_noise, scm_continuous_confounding

class ModelEnsemble(nn.Module):
    def __init__(self, input_size, num_models):
        super().__init__()
        self.model_backbone = nn.Sequential(
            nn.Linear(input_size, 24, True), nn.ReLU(), nn.Dropout(), nn.Linear(24,1))
        self.models = nn.ModuleList([self.model_backbone for model in range(num_models)])
        
    def forward(self, x):
        return torch.stack([model(x) for model in self.models])

class Model(nn.Module):
    def __init__(self, input_size, num_models):
        super().__init__()
        self.lin = nn.Linear(input_size, 12, True)
        self.rel =  nn.ReLU()
        self.mid = nn.Linear(12, 24, True)
        self.drop = nn.Dropout(p=0.3)
        self.lin2 = nn.Linear(24, 2,  True)
        
    def forward(self, x):
        x = self.lin(x)
        x = self.rel(x)
        x = self.drop(x)
        x = self.mid(x)
        x = self.rel(x)
        x = self.drop(x)
        x = self.lin2(x)
        x = self.drop(x)
        return x
    

class ModelEnsembleHet(nn.Module):
    def __init__(self, input_size, num_models):
        super().__init__()
        self.modelA = nn.Sequential(
            nn.Linear(input_size, 12, True), nn.ReLU(), nn.Linear(12, 1, True))
        self.modelB = nn.Sequential(
            nn.Linear(input_size, 12, True), nn.ReLU(), nn.Dropout(), nn.Linear(12,1, True))
        self.modelC = nn.Sequential(
            nn.Linear(input_size, 24, True), nn.ReLU(), nn.Dropout(), nn.Linear(24,12, True), nn.ReLU(), nn.Linear(12, 1, True))

        self.models = nn.ModuleList([self.modelA, self.modelB, self.modelC])
        
    def forward(self, x):
        return torch.stack([model(x) for model in self.models])


def ensemble_loss(output, target):
    losses = torch.stack(
        [F.binary_cross_entropy_with_logits(model_result, target) for model_result in output])
    total_loss = torch.sum(losses)
    return total_loss



def train(num_epochs, model, data, target, lr, device,  log_interval=2000, ensemble=False):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #data,target = torch.stack(data), torch.stack(target)
    #data, target = data.to(device), target.to(device)
    for epoch in range(num_epochs):
        #data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if ensemble:
            loss = ensemble_loss(output, target)
        else:
            loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(output, axis=1)
        if ensemble:
            num_models = preds.shape
        else:
            num_models = 1
        num_correct = ((preds - target).abs() < 1e-2).float().sum()
        correct = num_correct / (len(target)*num_models)
        if epoch % log_interval == 0:

            print('Train Epoch: {} Loss: {:.3f} Train correct: {:.3f}' .format(
                epoch, loss.item(), correct))
    return correct

def test(model, data, target, device, ensemble):
    #data, target = data.to(device), target.to(device)
    # data, target = torch.FloatTensor(data).to(device), torch.FloatTensor(target).to(device)
    model.eval()
    output = model(data)
    preds = torch.argmax(output, axis=1)
    num_correct = ((preds - target).abs() < 1e-2).float().sum()
    if ensemble:
        num_models = preds.shape[0]
    else:
        num_models = 1

    accuracy = num_correct / (len(target)*num_models)

    return accuracy

#array = np.concatenate(preds).reshape(num_models, data_size)
# each row represents a prediction from a model
def calc_score(preds, n_largest, train_indices, prop):
    # finds the max entropy points from predictions on
    # the whole dataset, and removes items which are no longer
    # in the poolset. 
    # returns index of data with respect to whole dataset
    majority_data = int(np.floor(preds[0].shape[0]*prop))
    minority_data = preds[0].shape[0] - majority_data
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
    return score_largest, mean_score_maj, mean_score_min,  

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


def al_loop(models, data, target, data_test, target_test,
            n_largest, al_iters, lr, num_epochs, device, prop, wandb,
            log_int = 1000, random_ac=False):
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
            n_largest_idx, mean_score_maj, mean_score_min = calc_score(
                preds, n_largest, train_indices, prop)

            n_largest_idx = random.sample(pool_idx, n_largest)
        else:
            models.train()
            preds = [F.softmax(models(data), dim=1).cpu().clone().detach().numpy() for _ in range(10)]
            n_largest_idx, mean_score_maj, mean_score_min = calc_score(
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
    print('final train size {}'.format(data_train.shape[0]))
    prop_minority = np.sum(
        np.where(np.asarray(train_indices) == 1)[0]<majority_data)/len(
            np.where(np.asarray(train_indices) ==1)[0])
    print('prop minority selected for train: {}'.format(prop_minority))
    prop_maj = np.sum(
            np.where(np.asarray(train_indices) == 1)[0]>majority_data)/len(
            np.where(np.asarray(train_indices) ==1)[0])
    print('prop majority selected for train: {}'.format(prop_maj))
    return train_acc, test_acc, prop_maj, prop_minority, data_train, data_pool, mean_score_maj, mean_score_min
    
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--entangle', action='store_true')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--no_confound_test', action='store_true')
    parser.add_argument('--scm_ac', action='store_true')
    parser.add_argument('--standard_train', action='store_true')
    parser.add_argument('--non_lin_entangle', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_size', type=int, default=200)
    parser.add_argument('--n_largest', type=int, default=2)
    parser.add_argument('--al_iters', type=int, default=10)
    parser.add_argument('--proportion', type=float, default=0.50)
    parser.add_argument('--num_epochs', type=int, default=3010)

    args = parser.parse_args()
    run = wandb.init(
        project='causal_al_r2',
        settings=wandb.Settings(start_method='fork')
    )
    wandb.config.update(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_size = args.data_size
    num_models = 4
    num_epochs = args.num_epochs
    n_largest = args.n_largest
    al_iters = args.al_iters
    lr = 1e-2
    non_lin = args.non_lin_entangle
    seed = 0
    entangled = False
    cc = True

    if entangled: 
        data, target, data_test, target_test = entangled_data(seed, num_samples=data_size)
        input_size = 2
        target = target.unsqueeze(1).float()
        target_test = target_test.unsqueeze(1).float()

    if cc: 
        scm = CC(5,5,2, non_lin=non_lin)
        if non_lin: 
            input_size = 5
        else:
            input_size = 10
        data, target, data_test, target_test, min_size = scm.mix_train_test(
            args.proportion, data_size, no_confounding_test=args.no_confound_test)
        target = target.squeeze(1).type(torch.LongTensor)
        target_test = target_test.squeeze(1).type(torch.LongTensor)

    if args.scm_ac:
        majority_data = int(np.floor(data_size*args.proportion))
        minority_data = data_size - majority_data

        data, target = scm1_noise(
            seed, 
            env1=(0.01, 0.01, 0.01), env2=(0.01, 0.01, 0.01), num_samples=(majority_data, minority_data), entangle=False)
        data_test, target_test = scm1_noise(seed+1, env1=(0.01, 0.01, 0.01), env2=(0.01, 0.01, 0.01), num_samples=(
            minority_data, majority_data), entangle=False)
        input_size = 2
    if args.cont:
        data, target = scm_continuous_confounding(
            seed, prop_1=args.proportion, entangle=args.non_lin_entangle, flip_y=0, num_samples=args.data_size)
        data_test, target_test = scm_continuous_confounding(
            seed+1, prop_1=(1-args.proportion), entangle=args.non_lin_entangle, flip_y=0, num_samples=args.data_size)
        input_size = 2
    lr = 1e-3
    standard_train = args.standard_train
    ensemble = False
        
    rand_ac = args.random

    #data, target = scm.sample(split='train')
    #data_test, target_test = scm.sample(split='test')
    majority_data = int(np.floor(data_size*args.proportion))
    minority_data = data_size - majority_data
    data = data.to(device)
    target = target.to(device)
    data_test = data_test.to(device)
    target_test = target_test.to(device)
    if ensemble: 
        models = ModelEnsemble(input_size, num_models)
    else:
        models = Model(input_size, num_models)
    models.to(device)
    preds = models(data)
    if standard_train:
        train_acc = train(num_epochs, models, data, target, lr, device, ensemble=ensemble)
        test_acc = test(models, data_test, target_test, device, ensemble=ensemble)
        print('test accuracy: {}'.format(test_acc))
    else:
        train_acc, test_acc, prop_maj, prop_minority, data_train, data_pool, mean_score_maj, mean_score_min = al_loop(
            models, data, target, data_test, target_test, n_largest, al_iters, lr, num_epochs, device,
            args.proportion, run, random_ac=rand_ac)
        print("random: ", args.random)
        wandb.run.summary.update({"test acc": test_acc,
                                  "train_acc": train_acc,
                                  "mean score majority env": mean_score_maj,
                                  "mean score minority env": mean_score_min,
                                  "points in train set": data_train.shape[0],
                                  "points in pool set": data_pool.shape[0],
                                  "prop minority selected for train": prop_minority,
                                  "prop majority selected for train": prop_maj,
                                  "minority data size":min_size })
