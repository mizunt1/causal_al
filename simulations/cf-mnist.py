from torchvision import datasets
import numpy as np
import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

def flip_binary(seed, array, proportion):
    rng = np.random.default_rng(seed=seed)
    flipped = rng.choice([0,1], size=len(array), p=[1-proportion, proportion])
    flipped_array = [int(y^1) if z else int(y) for y,z in zip(array, flipped)]
    return flipped_array

def cf_mnist(seed, data, label, colour_flip, label_flip):
    class_split=0.5
    rng = np.random.default_rng(seed=seed)
    labels = (label < 5).float()
    target1_indices = torch.where(labels == 1)[0]
    target0_indices = torch.where(labels == 0)[0]
    num_target1 = int(np.floor(class_split*len(target1_indices)))
    target1_chosen = rng.choice(target1_indices, size=num_target1,replace=False)
    num_target0 = int(np.floor((1-class_split)*len(target1_indices)))
    target0_chosen = rng.choice(target0_indices, size=num_target0,replace=False)
    swap1 = rng.choice(target1_chosen, size = int(label_flip*len(label)))
    swap0 = rng.choice(target0_chosen, size = int(label_flip*len(label)))
    # split data in to class 1 and class 0. 
    for item1, item0 in zip(swap1, swap0):
        labels[item1] = 0
        labels[item0] = 1

    # sample classes in an inbalanced way
    colours_target1 = [1 for x in target1_chosen]
    colours_target1_flipped = flip_binary(seed, colours_target1, colour_flip)
    colours_target0 = [0 for x in target0_chosen]
    colours_target0_flipped = flip_binary(seed, colours_target0, colour_flip)
    # assign colours for each image, 
    data_target0 = torch.stack([data[target0_chosen,:,:], data[target0_chosen,:,:]], dim=1)    
    data_target0[torch.tensor(range(len(data_target0))), [x^1 for x in colours_target0_flipped], :, :] *= 0
    data_target1 = torch.stack([data[target1_chosen,:,:], data[target1_chosen,:,:]], dim=1)
    data_target1[torch.tensor(range(len(data_target1))),  [x^1 for x in colours_target1_flipped], :, :] *= 0
    # change colours of mnist digits
    return torch.cat((data_target1.float()/255., data_target0.float()/255.)), torch.cat((labels[target1_chosen], labels[target0_chosen]))
    
class CFMnist(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx, :, : ,:], self.label[idx]

class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super(MLP, self).__init__()
        lin1 = nn.Linear(2 * 28 * 28, hidden_dim)
        lin2 = nn.Linear(hidden_dim, hidden_dim)
        lin3 = nn.Linear(hidden_dim, 1)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
        
    def forward(self, input):
        out = input.view(input.shape[0], 2 * 28 * 28)
        out = self._main(out)
        return torch.squeeze(out, dim=1)

def test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        preds = (output > 0.).float()
        num_correct = ((preds - target).abs() < 1e-2).float().sum()
        correct += num_correct
        total += len(target)
    accuracy = correct / total
    return accuracy

def train(num_epochs, model, dataloader_train, dataloader_test, lr, device,  log_interval=100):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        for data, target in dataloader_train: 
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.binary_cross_entropy_with_logits(output, target)
            loss.backward()
            optimizer.step()
            
            preds = (output > 0.).float()
            num_correct = ((preds - target).abs() < 1e-2).float().sum()
            correct += num_correct
            total += len(target)
        # accuracy = torch.sum(torch.argmax(output, axis=1) == torch.arxmax(target, axis=1)) /len(output)
        print('Train Epoch: {} Loss: {:.3f} Train correct: {:.3f}' .format(
            epoch, loss.item(), correct/total))
        acc_test = test(model, dataloader_test, device)
        print('Test Epoch: {} Test accuracy: {:.3f}' .format(
            epoch, acc_test))

def setup_envs(mnist_data, majority_frac_train, majority_frac_test, train_size=50000, test_size=None):
    if test_size != None:
        assert train_size + test_size < len(mnist.targets)
    if test_size == None:
        test_size = train_size
    mnist_train = (mnist.data[:train_size], mnist.targets[:train_size])
    mnist_test = (mnist.data[test_size:], mnist.targets[test_size:])
    test_len = len(mnist_test[0])
    
    majority_env_train = int(train_size*majority_frac_train)
    minority_env_train = int(train_size-majority_env_train)
    majority_env_test = int(test_len*majority_frac_test)
    minority_env_test = int(test_len-majority_env_test)
    
    print(
        "Majority size train: {}, minority size train: {} Majority size test: {} minority size test: {}".format(
            majority_env_train, minority_env_train, majority_env_test, minority_env_test))
    majority_train_data = (mnist_train[0][:majority_env_train], mnist_train[1][:majority_env_train])
    minority_train_data = (mnist_train[0][majority_env_train:], mnist_train[1][majority_env_train:])

    majority_test_data = (mnist_test[0][:majority_env_test], mnist_test[1][:majority_env_test])
    minority_test_data = (mnist_test[0][majority_env_test:], mnist_test[1][majority_env_test:])

    data_env1_train, labels_env1_train = cf_mnist(0, majority_train_data[0], majority_train_data[1], colour_flip=0.9, label_flip=0.1)
    data_env0_train, labels_env0_train = cf_mnist(0, minority_train_data[0], minority_train_data[1], colour_flip=0.1, label_flip=0.1)

    data_env1_test, labels_env1_test = cf_mnist(0, minority_test_data[0], minority_test_data[1], colour_flip=0.9, label_flip=0.1)
    data_env0_test, labels_env0_test = cf_mnist(0, majority_test_data[0], majority_test_data[1], colour_flip=0.1, label_flip=0.1)
    
    data_train = torch.cat((data_env1_train, data_env0_train))
    labels_train = torch.cat((labels_env1_train, labels_env0_train))
    data_test = torch.cat((data_env1_test, data_env0_test))
    labels_test = torch.cat((labels_env1_test, labels_env0_test))

    # created equally size data for env1 and env2
    dataset_train = CFMnist(data_train, labels_train)
    dataloader_train = DataLoader(dataset_train, batch_size=100, shuffle=True)
    dataset_test = CFMnist(data_test, labels_test)
    dataloader_test = DataLoader(dataset_test, batch_size=100, shuffle=True)
    return dataloader_train, dataloader_test

if __name__ == "__main__":
    majority_frac_train = 0.99
    majority_frac_test = 0.5
    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    dataloader_train, dataloader_test = setup_envs(mnist, majority_frac_train, majority_frac_test)#, train_size=10000)
    device = torch.device('cuda')
    hidden_dim = 256
    model = MLP(hidden_dim).to(device)
    train(100, model, dataloader_train, dataloader_test, 1e-3, device, log_interval=5000)
    rng_state = np.random.get_state()

