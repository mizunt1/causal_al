from collections import namedtuple
from argparse import ArgumentParser

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from data_gen import scm1_data, scm_ac_data

class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 12, True)
        self.do = nn.Dropout()
        self.fc2 = nn.Linear(12, 2)
        self.sm = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.do(x)
        x = self.fc2(x)
        x = self.sm(x)
        return x


def train(num_epochs, model, data, target, lr, device,  log_interval=100):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if epoch % log_interval == 0:
            correct = torch.sum(torch.argmax(output, axis=1) == target) 
            
            # accuracy = torch.sum(torch.argmax(output, axis=1) == torch.arxmax(target, axis=1)) /len(output)
            print('Train Epoch: {} Loss: {:.3f} Train correct: {:.3f}' .format(
                epoch, loss.item(), correct))

def test(model, data, target, device):
    model.eval()
    data, target = data.to(device), target.to(device)
    output = model(data)
    accuracy = torch.sum(torch.argmax(output, axis=1) == target) 
    return accuracy

def main(args):
    seed = 3
    torch.manual_seed(seed)
    device = torch.device('cuda')
    scm = args.scm
    option = args.option
    plot = args.plot
    input_size = 2        

    if scm == "scm1":
        data, target, data_test, target_test = scm1_data(option, seed)
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
    elif scm == "scm_ac":
        data, target, data_test, target_test = scm_ac_data(seed, option)
        lr = 1e-2
    # MODEL SETUP
    model = Model(input_size).to(device)
    test_accuracy = test(model, data_test, target_test, device)
    print('test accuracy before training: {}'.format(test_accuracy))
    log_interval = 1

    train(1000, model, data, target, 1e-2, device)

    test_accuracy = test(model, data_test, target_test, device)
    print('test accuracy: {}'.format(test_accuracy))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--scm', type=str, choices=['scm1', 'scm_ac'])
    parser.add_argument('--option', type=str)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    main(args)
