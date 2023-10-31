import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch 

class ModelReg(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lin = nn.Linear(input_size, 12, True)
        self.rel =  nn.ReLU()
        self.mid = nn.Linear(12, 24, True)
        self.drop = nn.Dropout(p=0.3)
        self.lin2 = nn.Linear(24, 1,  True)
        
    def forward(self, x):
        x = self.lin(x)
        x = self.rel(x)
        x = self.mid(x)
        x = self.rel(x)
        x = self.drop(x)
        x = self.lin2(x)
        return x
    
def train_reg(num_epochs, model, data, target, lr, device,  log_interval=2000):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    #data,target = torch.stack(data), torch.stack(target)
    #data, target = data.to(device), target.to(device)
    for epoch in range(num_epochs):
        #data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        mse = ((output - target).abs()).float().sum()
        correct = mse / (len(target))
        if epoch % log_interval == 0:

            print('Train Epoch for reg model: {} Loss: {:.3f} Train mse: {:.3f}' .format(
                epoch, loss.item(), correct))
    return correct

def test_reg(model, data, target, device):
    #data, target = data.to(device), target.to(device)
    # data, target = torch.FloatTensor(data).to(device), torch.FloatTensor(target).to(device)
    model.eval()
    output = model(data)
    num_correct = ((output - target).abs() < 1e-2).float().sum()
    accuracy = num_correct / (len(target))

    return accuracy
