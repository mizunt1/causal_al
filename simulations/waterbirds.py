import torch

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

# Train loop
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = ConvNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    batch_idx = 0
    for x, y, meta in train_loader:
        batch_idx += 1
        # plt.imshow(x[0][0], interpolation='nearest')
        # plt.show()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        if batch_idx %10 == 0:
            #correct = torch.gt(torch.argmax(output), torch.Tensor([0.0]).to(device)) == y
            correct = torch.argmax(output, axis=1) == y
            correct = correct.sum()
            print('Train Epoch: {} Loss: {:.4f} Train accuracy: {:.4f}'.format(
                epoch, loss.item(), correct/len(y)))
            #print(output)

