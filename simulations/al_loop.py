from cows_camels import CC
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch 
import random

class ModelEnsemble(nn.Module):
    def __init__(self, input_size, num_models):
        super().__init__()
        self.model_backbone = nn.Sequential(
            nn.Linear(input_size, 12, True), nn.ReLU(), nn.Dropout(), nn.Linear(12,1))
        self.models = nn.ModuleList([self.model_backbone for model in range(num_models)])
        
    def forward(self, x):
        return torch.stack([model(x) for model in self.models])

def ensemble_loss(output, target):
    losses = torch.stack(
        [F.binary_cross_entropy_with_logits(model_result, target) for model_result in output])
    total_loss = torch.sum(losses)
    return total_loss


def train(num_epochs, model, data, target, lr, device,  log_interval=10):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #data,target = torch.stack(data), torch.stack(target)
    #data, target = data.to(device), target.to(device)

    for epoch in range(num_epochs):
        #data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = ensemble_loss(output, target)
        loss.backward()
        optimizer.step()
        if epoch % log_interval == 0:
            preds = (output > 0.).float()
            num_correct = ((preds - target).abs() < 1e-2).float().sum()
            correct = num_correct 
            # accuracy = torch.sum(torch.argmax(output, axis=1) == torch.arxmax(target, axis=1)) /len(output)
            print('Train Epoch: {} Loss: {:.3f} Train correct: {:.3f}' .format(
                epoch, loss.item(), correct/(len(target)*len(preds))))

def test(model, data, target, device):
    model.eval()
    #data, target = data.to(device), target.to(device)
    # data, target = torch.FloatTensor(data).to(device), torch.FloatTensor(target).to(device)
    output = model(data)
    preds = (output > 0.).float()
    num_correct = ((preds - target).abs() < 1e-2).float().sum()

    accuracy = num_correct / (len(target)* len(preds))

    return accuracy

#array = np.concatenate(preds).reshape(num_models, data_size)
# each row represents a prediction from a model
def entropy(preds, n_largest, train_indices):
    # finds the max entropy points from predictions on
    # the whole dataset, and removes items which are no longer
    # in the poolset. 
    # returns index of data with respect to whole dataset
    means = preds.mean(axis=0).detach().numpy().squeeze(1)
    sd = preds.std(axis=0).detach().numpy().squeeze(1)
    score = sd + abs(means)
    score_masked = [score_val if idx < 1 else 0. for score_val, idx in zip(score, train_indices)]
    score_labeled = np.vstack([score_masked, [i for i in range(len(score))]]) 
    score_sorted = np.argsort(score_masked)
    score_selected = score_labeled[:,score_sorted]
    return score_selected[1,n_largest*-1:]

def al_loop(models, data, target, data_test, target_test,
            n_largest, al_iters, lr, num_epochs, device, log_int = 100, random_ac=False):
    assert(al_iters*n_largest < len(data))
    pool_indices = [1 for i in range(len(data))]
    train_indices = [0 for i in range(len(data))]
    for iter in range(al_iters):
        if random_ac:
            pool_idx = [idx for (idx, in_pool) in enumerate(pool_indices) if in_pool > 0 ]
            n_largest_idx = random.sample(pool_idx, n_largest)
        else:
            preds = models(data)
            n_largest_idx = entropy(preds, n_largest, train_indices)
            print(n_largest_idx)
        for idx in n_largest_idx:
            train_indices[int(idx)] = 1
            pool_indices[int(idx)] = 0
        data_pool = [data[idx] for idx, in_pool in enumerate(pool_indices) if in_pool > 0]
        data_train = [data[idx] for idx, in_train in enumerate(train_indices) if in_train > 0]

        target_pool = [target[idx] for idx, in_pool in enumerate(pool_indices) if in_pool > 0]
        target_train = [target[idx] for idx, in_train in enumerate(train_indices) if in_train > 0]
        data_pool, target_pool = torch.stack(data_pool).to(device), torch.stack(target_pool).to(device)
        data_train, target_train = torch.stack(data_train).to(device), torch.stack(target_train).to(device)

        train(num_epochs, models, data_train, target_train, lr, device, log_interval=log_int)
        test_acc = test(models, data_test, target_test, device)
        print('Al iter: {} test accuracy: {}'.format(iter, test_acc))
        print('Al iter: {} points in train set: {}'.format(iter, sum(train_indices)))
        print('Al iter: {} points in pool set: {}'.format(iter, sum(pool_indices)))
    print('total dataset size {}'.format(len(target)))

if __name__ == "__main__":
    device = 'cpu'
    input_size = 5
    data_size = 10_000
    input_size = 5
    num_models = 4
    num_epochs = 200
    n_largest = 50
    al_iters = 20
    lr = 1e-3
    scm = CC(5,5,6, non_lin=True)
    standard_train = False
    rand_ac = True
    data, target, data_test, target_test = scm.mix_train_test(0.7, data_size, no_confounding_test=True)
    data.to(device)
    models = ModelEnsemble(input_size, num_models)
    preds = models(data)
    #for i in range(al_loop_iters):
    if standard_train:
        train(num_epochs, models, data, target, lr, device)
        test_acc = test(models, data_test, target_test, device)
        print('test accuracy: {}'.format(test_acc))
    else:
        al_loop(models, data, target, data_test, target_test,
            n_largest, al_iters, lr, num_epochs, device, random_ac=rand_ac)
