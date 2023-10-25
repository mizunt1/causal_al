from collections import namedtuple
from argparse import ArgumentParser

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from data_gen import scm1_data, scm_ac_data, scm_band_data, scm1_noise_data, entangled_data, entangled_image_data
from cows_camels import CC

class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 12, True)
        self.relu = nn.ReLU()
        self.do = nn.Dropout()
        self.fc2 = nn.Linear(12, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.do(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train(num_epochs, model, data, target, lr, device,  log_interval=100):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()


        
        output = model(data)
        loss = F.binary_cross_entropy_with_logits(output, target)
        
        loss.backward()
        optimizer.step()
        if epoch % log_interval == 0:
            preds = (output > 0.).float()
            num_correct = ((preds - target).abs() < 1e-2).float().sum()
            correct = num_correct 
            
            # accuracy = torch.sum(torch.argmax(output, axis=1) == torch.arxmax(target, axis=1)) /len(output)
            print('Train Epoch: {} Loss: {:.3f} Train correct: {:.3f}' .format(
                epoch, loss.item(), correct/len(target)))

def test(model, data, target, device):
    model.eval()
    data, target = data.to(device), target.to(device)
    output = model(data)
    preds = (output > 0.).float()
    num_correct = ((preds - target).abs() < 1e-2).float().sum()

    accuracy = num_correct / len(target)

    return accuracy

def main(args):
    seed = 3
    torch.manual_seed(seed)
    device = torch.device('cuda')
    scm = args.scm
    option = args.option
    plot = args.plot
    input_size = 2        
    if scm == "scm_band":
        data, target, data_test, target_test = scm_band_data(seed)
    elif scm == "scm1":
        data, target, data_test, target_test = scm1_data(seed, option)
        if plot:
            plt.axhline(y=0.5, color='r')
            plt.ylabel("causal variable")
            plt.xlabel("non causal variable")
            plt.scatter(data[:,0], data[:,1])
            plt.show()

            plt.axhline(y=0.5, color='r')
            plt.ylabel("causal variable")
            plt.xlabel("non causal variable")
            plt.scatter(data_test[:,0], data_test[:,1])
            plt.show()
        lr = 1e-2
    elif scm == 'scm_noise':
        data, target, data_test, target_test = scm1_noise_data(seed)
    elif scm == "scm_ac":
        data, target, data_test, target_test = scm_ac_data(seed)
        lr = 1e-2
    elif scm == 'entangled':
        data, target, data_test, target_test = entangled_data(seed)
        target = target.unsqueeze(1).float()
        target_test = target_test.unsqueeze(1).float()
        input_size = 1
        lr = 1e-3
    elif scm == 'cow_camel':
        scm = CC(5,5,2, non_lin=False)
        
        #data, target = scm.sample()
        #data_test, target_test = scm.sample(split='test')
        data, target, data_test, target_test = scm.mix_train_test(0.5, 100)
        input_size = 10
        lr = 1e-3
        
    # MODEL SETUP
    model = Model(input_size).to(device)        
    test_accuracy = test(model, data_test, target_test, device)
    print('test accuracy before training: {}'.format(test_accuracy))
    log_interval = 1
    num_epochs = 10_000
    train(num_epochs, model, data, target, lr, device)

    test_accuracy = test(model, data_test, target_test, device)
    print('test accuracy: {}'.format(test_accuracy))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--scm', type=str)
    parser.add_argument('--option', type=str)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    main(args)
