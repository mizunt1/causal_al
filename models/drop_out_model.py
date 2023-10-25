import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch 

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
    
def train(num_epochs, model, data, target, lr, device,  log_interval=2000, ensemble=False):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #data,target = torch.stack(data), torch.stack(target)
    #data, target = data.to(device), target.to(device)
    for epoch in range(num_epochs):
        #data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(output, axis=1)
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
    num_models = 1    
    accuracy = num_correct / (len(target)*num_models)

    return accuracy
