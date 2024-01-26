# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import numpy as np
import torch
import copy
from torchvision import datasets
from torch import nn, optim, autograd

def accuracy(predicted, target):
    return np.sum((predicted == target))/ len(target)

def worst_group_acc(predicted, target, groups):
    # groups correspond to list of indices, where each item corresponds
    # to a starting point for each group.
    accuracies = []
    current_idx = 0
    for idx in range(len(groups)):
        predicted_section = predicted[current_idx: groups[idx]]

        target_section = target[current_idx: groups[idx]]
        current_idx = groups[idx]
        acc = mean_accuracy(predicted_section, target_section)
        if len(target_section) != 0:
            accuracies.append(acc.detach().cpu().item())
    predicted_section = predicted[current_idx:]
    target_section = target[current_idx:]
    acc = mean_accuracy(predicted_section, target_section)
    accuracies.append(acc.detach().cpu().numpy())
    return min(np.array(accuracies))

def make_environment(images, labels, e):
  def torch_bernoulli(p, size):
    return (torch.rand(size) < p).float()
  def torch_xor(a, b):
    return (a-b).abs() # Assumes both inputs are either 0 or 1
  # 2x subsample for computational convenience
  images = images.reshape((-1, 28, 28))[:, ::2, ::2]
  # Assign a binary label based on the digit; flip label with probability 0.25
  labels = (labels < 5).float()
  labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
  # Assign a color based on the label; flip the color with probability e
  colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
  # Apply the color to the image by zeroing out the other color channel
  images = torch.stack([images, images], dim=1)
  images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
  return {
    'images': (images.float() / 255.).cuda(),
    'labels': labels[:, None].cuda()
  }

# Define and instantiate the model

class Log_reg(nn.Module):
    def __init__(self):
        super(Log_reg, self).__init__()
        self.linear1 = torch.nn.Linear(2, 1)
        #self.act = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.linear1(x)
        #x = self.act(x)
        return x.squeeze(1)


class MLP_IRM_MNIST(nn.Module):
  def __init__(self):
    super(MLP_IRM_MNIST, self).__init__()
    if flags.grayscale_model:
      lin1 = nn.Linear(14 * 14, flags.hidden_dim)
    else:
      lin1 = nn.Linear(2 * 14 * 14, flags.hidden_dim)
    lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
    lin3 = nn.Linear(flags.hidden_dim, 1)
    for lin in [lin1, lin2, lin3]:
      nn.init.xavier_uniform_(lin.weight)
      nn.init.zeros_(lin.bias)
    self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
  def forward(self, input):
    if flags.grayscale_model:
      out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
    else:
      out = input.view(input.shape[0], 2 * 14 * 14)
    out = self._main(out)
    return out


# Define loss function helpers
class MLP_IRM_SIMULATED(nn.Module):
  def __init__(self):
    super(MLP_IRM_SIMULATED, self).__init__()
    lin1 = nn.Linear(2, 12)
    lin2 = nn.Linear(12, 24)
    lin3 = nn.Linear(24, 1)
    for lin in [lin1, lin2, lin3]:
      nn.init.xavier_uniform_(lin.weight)
      nn.init.zeros_(lin.bias)
    self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

  def forward(self, input):
    out = self._main(input)
    return out.squeeze(1)

def mean_nll(logits, y):
  return nn.functional.binary_cross_entropy_with_logits(logits, y.float())

def mean_accuracy(logits, y):
  preds = (logits > 0.).float()
  return ((preds - y).abs() < 1e-2).float().mean()

def penalty(logits, y):
  scale = torch.tensor(1.).cuda().requires_grad_()
  loss = mean_nll(logits * scale, y)
  grad = autograd.grad(loss, [scale], create_graph=True)[0]
  return torch.sum(grad**2)

def pretty_print(*values):
  col_width = 13
  def format_val(v):
    if not isinstance(v, str):
      v = np.array2string(v, precision=5, floatmode='fixed')
    return v.ljust(col_width)
  str_values = [format_val(v) for v in values]
  print("   ".join(str_values))

# Train loop
def train(mlp, envs_train, env_test, irm, lr, groups_test,
          l2_regularizer_weight, penalty_weight, penalty_anneal_iters, print_=False):
  """ envs train is a list of dictionaries whereas env_test is one dictionary"""
  optimizer = optim.Adam(mlp.parameters(), lr=lr)
  if print_:
    pretty_print('step', 'train nll', 'train acc',
                 'train acc w', 'train penalty', 'test acc', 'worst test ac')
  non_empt_envs = []
  size_of_envs = []
  for step in range(500):
    for env in envs_train:
      logits = mlp(env['images'])
      env['nll'] = mean_nll(logits, env['labels'])
      env['acc'] = mean_accuracy(logits, env['labels'])
      env['penalty'] = penalty(logits, env['labels'])
      if len(env['labels']) != 0:
        non_empt_envs.append(env)
        size_of_envs.append(len(env['labels']))
    train_nll = torch.stack([non_empt_envs[i]['nll'] for i in range(len(non_empt_envs))]).mean()
    train_acc = torch.stack([non_empt_envs[i]['acc'] for i in range(len(non_empt_envs))]).mean()
    train_acc_weighted = torch.stack(
      [non_empt_envs[i]['acc']*size_of_envs[i] for i in range(len(non_empt_envs))]).sum()/sum(size_of_envs)
    train_nll_weighted = torch.stack(
      [non_empt_envs[i]['nll']*size_of_envs[i] for i in range(len(non_empt_envs))]).sum()/sum(size_of_envs)
        
    train_penalty = torch.stack([non_empt_envs[i]['penalty'] for i in range(len(non_empt_envs))]).mean()
    weight_norm = torch.tensor(0.).cuda()
    for w in mlp.parameters():
      weight_norm += w.norm().pow(2)  
    if irm:
      loss = train_nll.clone()
      loss += l2_regularizer_weight * weight_norm
      penalty_weight = (penalty_weight 
                        if step >= penalty_anneal_iters else 1.0)
      loss += penalty_weight * train_penalty
      if penalty_weight > 1.0:
        # Rescale the entire loss to keep gradients in a reasonable range
        loss /= penalty_weight
    else:
      loss = train_nll_weighted + l2_regularizer_weight * weight_norm

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    test_acc = mean_accuracy(mlp(env_test['images']),env_test['labels'])
    worst = worst_group_acc(mlp(env_test['images']),
                                env_test['labels'], groups_test)
    if step % 100 == 0:
      if print_:
        pretty_print(
          np.int32(step),
          train_nll.detach().cpu().numpy(),
          train_acc.detach().cpu().numpy(),
          train_acc_weighted.detach().cpu().numpy(),
          train_penalty.detach().cpu().numpy(),
          test_acc.detach().cpu().numpy(),
          worst
        )
  return train_acc, test_acc, worst, train_acc_weighted

def test_erm_irm(mlp, env_test, groups_test, mask):
    test_acc_standard = mean_accuracy(mlp(env_test['images']),env_test['labels'])
    
    env_test_copy = copy.deepcopy(env_test)
    if mask == 'causal':
        env_test_copy['images'][:, 1] = torch.normal(mean=torch.zeros(len(env_test['images'])))

    elif mask == 'sp':
        env_test_copy['images'][:, 0] = torch.normal(mean=torch.zeros(len(env_test['images'])))
    else:
        pass
    test_acc = mean_accuracy(mlp(env_test_copy['images']),env_test['labels'])
    worst = worst_group_acc(mlp(env_test_copy['images']),
                                env_test['labels'], groups_test)
    return test_acc, worst, test_acc_standard

def train_erm(mlp, flags, data_train, target_train):
    optimizer = optim.Adam(mlp.parameters(), lr=flags.lr_irm)
    for step in range(flags.steps):
      logits = mlp(data_train)
      loss = mean_nll(logits, target_train)
      mean_acc = mean_accuracy(logits, target_train)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    print("train mean acc: ", mean_acc)
    return mean_acc

def test_erm(mlp, data_test, target_test):
      logits = mlp(data_test)
      loss = mean_nll(logits, target_test)
      mean_acc = mean_accuracy(logits, target_test)
      print("test mean acc: ", mean_acc)
      return mean_acc


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Colored MNIST')
  parser.add_argument('--hidden_dim', type=int, default=256)
  parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
  parser.add_argument('--lr_irm', type=float, default=0.001)
  parser.add_argument('--n_restarts', type=int, default=1)
  parser.add_argument('--penalty_anneal_iters', type=int, default=100)
  parser.add_argument('--penalty_weight', type=float, default=10000.0)
  parser.add_argument('--steps', type=int, default=501)
  parser.add_argument('--grayscale_model', action='store_true')
  flags = parser.parse_args()
  
  print('Flags:')
  for k,v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))
  mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
  mnist_train = (mnist.data[:50000], mnist.targets[:50000])
  mnist_val = (mnist.data[50000:], mnist.targets[50000:])

  rng_state = np.random.get_state()
  np.random.shuffle(mnist_train[0].numpy())
  np.random.set_state(rng_state)
  np.random.shuffle(mnist_train[1].numpy())

  # Build environments


  mlp = MLP_IRM_MNIST().cuda()
  envs = [
    make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2),
    make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
    make_environment(mnist_val[0], mnist_val[1], 0.9)
  ]
  # envs is a list
  # make environment returns a dict containing dict_keys(['images', 'labels'])
   
  train_irm(mlp, flags, envs)
