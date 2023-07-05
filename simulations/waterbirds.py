import torch

import numpy as np
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from matplotlib import pyplot as plt
torch.manual_seed(0)
normalize = transforms.Normalize(mean=0, std=255)
class ConvNet(nn.Module):
    def __init__(self):
      super(ConvNet, self).__init__()
      self.conv1 = nn.Conv2d(3, 20, 20, 1)
      self.conv2 = nn.Conv2d(20, 50, 10, 1)
      self.conv3 = nn.Conv2d(50, 20, 5, 2)
      self.conv4 = nn.Conv2d(20, 10, 5, 2)
      self.fc1 = nn.Linear(10 * 5 * 5, 250)
      self.fc2 = nn.Linear(250, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)    
        x = x.view(-1, 10*5*5)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def confusion(output, y, meta):
    output = output.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    meta = meta.cpu().detach().numpy()
    
    ww_pred = np.asarray([np.argmax(out) for out, ys, ms in zip(output, y, meta) if ys ==1 and ms==1])
    
    ww_correct = (ww_pred).sum()
    ww_total = len(ww_pred)
    # landbird on land
    
    ll_pred = np.asarray(
        [np.argmin(out) for out, ys, ms in zip(output, y, meta) if ys ==0 and ms==0])
    
    ll_correct = (ll_pred).sum()
    ll_total = len(ll_pred)
    
    #waterbird on land
    wbl_pred = np.asarray(
        [np.argmax(out) for out, ys, ms in zip(output, y, meta) if ys ==1 and ms==0])
    
    wbl_correct = (wbl_pred).sum()
    wbl_total = len(wbl_pred)
    
    # landbird on water
    
    lbw_pred = np.asarray(
        [np.argmin(out) for out, ys, ms in zip(output, y, meta) if ys == 0 and ms==1])
    
    lbw_correct = (lbw_pred).sum()
    lbw_total = len(lbw_pred)
    return ww_correct, ww_total, ll_correct, ll_total, wbl_correct, wbl_total, lbw_correct, lbw_total


# Load the full dataset, and download it if necessary
dataset = get_dataset(dataset="waterbirds", download=True)

# Get the training set
train_data = dataset.get_subset(
    "train",
    transform=transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor()]
    ),
)

# Prepare the standard data loader
train_loader = get_train_loader("standard", train_data, batch_size=16)

test_data = dataset.get_subset(
    "test",
    transform=transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor()]
    ),
)

# Prepare the standard data loader
test_loader = get_train_loader("standard", test_data, batch_size=16)

def test_loop(test_loader, device, model):
    ww = 0
    ww_total_sum = 0
    ll = 0
    ll_total_sum = 0
    wbl = 0
    wbl_total_sum = 0
    lbw = 0
    lbw_total_sum = 0
    correct = 0
    for x, y ,meta in test_loader:
        meta = meta[:,0]
        x, y, meta = x.to(device), y.to(device), meta.to(device)
        output = model(x)
        correct_batch = torch.argmax(output, axis=1) == y
        correct += correct_batch.sum()
        
        ww_correct, ww_total, ll_correct, ll_total, wbl_correct, wbl_total, lbw_correct, lbw_total = confusion(output, y, meta)
        ww += ww_correct
        ww_total_sum += ww_total
        ll += ll_correct
        ll_total_sum += ll_total
        wbl += wbl_correct
        wbl_total_sum += wbl_total
        lbw += lbw_correct
        lbw_total_sum += lbw_total

    print('Test accuracy: {:.4f}'.format(
        correct/len(test_loader)))
    print('test accuracy for ww: {:.3f}, ll: {:.3f}, wbl: {:.3f}, lbw: {:.3f}'.format(
        ww/ww_total_sum, ll/ll_total_sum, wbl/wbl_total_sum, lbw/lbw_total_sum))
        
# Train loop
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = ConvNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
num_epochs = 10
batch_idx = 0
ww = 0
ww_total_sum = 0
ll = 0
ll_total_sum = 0
wbl = 0
wbl_total_sum = 0
lbw = 0
lbw_total_sum = 0

for epoch in range(num_epochs):
    model.train()

    for x, y, meta in train_loader:
        meta = meta[:,0]
        batch_idx += 1
        #plt.imshow(x[0][0], interpolation='nearest')
        #plt.show()
        x, y, meta = x.to(device), y.to(device), meta.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        # waterbird on water
        ww_correct, ww_total, ll_correct, ll_total, wbl_correct, wbl_total, lbw_correct, lbw_total = confusion(output, y, meta)
        ww += ww_correct
        ww_total_sum += ww_total
        ll += ll_correct
        ll_total_sum += ll_total
        wbl += wbl_correct
        wbl_total_sum += wbl_total
        lbw += lbw_correct
        lbw_total_sum += lbw_total
        if batch_idx %10 == 0:
            #correct = torch.gt(torch.argmax(output), torch.Tensor([0.0]).to(device)) == y
            correct = torch.argmax(output, axis=1) == y
            correct = correct.sum()
            print('Train Epoch: {} Loss: {:.4f} Train accuracy: {:.4f}'.format(
            epoch, loss.item(), correct/len(y)))

            print('Train accuracy for ww: {:.3f}, ll: {:.3f}, wbl: {:.3f}, lbw: {:.3f}'.format(
                ww/ww_total_sum, ll/ll_total_sum, wbl/wbl_total_sum, lbw/lbw_total_sum))        
            test_loop(test_loader, device, model)
