from typing import NamedTuple, Tuple
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import torch
seed=0
torch.manual_seed(0)


Environments = namedtuple('Environments', ['means', 'sds'])
def scramble(data, dim_inv=1, dim_spu=1):
    torch.manual_seed(2)
    scramble_non_lin = torch.nn.Sequential(torch.nn.Linear((dim_inv + dim_spu), int((dim_inv + dim_spu))), torch.nn.ReLU())
    return scramble_non_lin(data)

def scm_f(seed, prop_1, num_samples, device):
    rng = np.random.default_rng(seed)
    if prop_1 >1:
        num_samples_env1 = int(prop_1)
        num_samples_env2 = int(num_samples - prop_1)
    else:
        num_samples_env1 = int(np.ceil(prop_1*num_samples))
        num_samples_env2 = num_samples - num_samples_env1

    env1 = 0.8
    u_1 = rng.choice([1, -1], size=num_samples_env1, p=[env1, 1-env1])
    x2_1 = rng.normal(u_1, 1)
    x1_1 = rng.normal(0, [1 for i in range(num_samples_env1)])
    y_1 = rng.normal(u_1*x1_1, 1)
    data_input_e1 = np.hstack((np.expand_dims(x1_1, axis=1), np.expand_dims(x2_1, axis=1)))

    env2 = 0.2
    u_2 = rng.choice([1, -1], size=num_samples_env2, p=[env2, 1-env2])
    x2_2 = rng.normal(u_2, 1)
    x1_2 = rng.normal(0, [1 for i in range(num_samples_env2)])
    y_2 = rng.normal(u_2*x1_2, 1)
    data_input_e2 = np.hstack((np.expand_dims(x1_2, axis=1), np.expand_dims(x2_2, axis=1)))
    data_input = np.append(data_input_e1, data_input_e2, axis=0)

    target_e1 = np.asarray([int(y_1val > 0) for y_1val in y_1])
    target_e2 = np.asarray([int(y_2val > 0) for y_2val in y_2])
    data_output = np.append(target_e1, target_e2)
    return torch.tensor(data_input).to(torch.float).to(device), torch.tensor(data_output.astype(int)).to(device)

def scm_anti_corr(seed, prop_1=0.2, num_samples=200, entangle=False, flip_y=0.01, device='cuda'):
    # in environment e1, x1 and x2 is correlated
    # in environment e2 the mean is flipped for sampling x1s.
    rng = np.random.default_rng(seed)
    if prop_1 >1:
        num_samples_env1 = int(prop_1)
        num_samples_env2 = int(num_samples - prop_1)
    else:
        num_samples_env1 = int(np.ceil(prop_1*num_samples))
        num_samples_env2 = num_samples - num_samples_env1

    means = rng.normal(0, 1, size=num_samples_env1)
    x1 = rng.normal(means, [0.5 for i in range(len(means))])
    x2 = rng.normal(means, [0.5 for i in range(len(means))])
    target_e1 = np.asarray([int(x2_val > 0) for x2_val in x2])
    flipped = rng.choice([0,1], size =num_samples_env1, p=[1-flip_y, flip_y])
    y_flipped_e1 = [int(y^1) if z else int(y) for y,z in zip(target_e1, flipped)]
    data_input_e1 = np.hstack((np.expand_dims(x1, axis=1), np.expand_dims(x2, axis=1)))

    e2 = -1
    means = rng.normal(0, 1, size=num_samples_env2)
    x1 = rng.normal(means, [1 for i in range(len(means))])
    x2 = rng.normal(e2*means, [1 for i in range(len(means))])
    target_e2 = np.asarray([int(x2_val > 0) for x2_val in x2])
    data_input_e2 = np.hstack((np.expand_dims(x1, axis=1), np.expand_dims(x2, axis=1)))
                            
    flipped = rng.choice([0,1], size =num_samples_env2, p=[1-flip_y, flip_y])
    y_flipped_e2 = [int(y^1) if z else int(y) for y,z in zip(target_e2, flipped)]
    data_input = np.append(data_input_e1, data_input_e2, axis=0)
    data_output = np.append(y_flipped_e1, y_flipped_e2)
    if entangle:
        data_input = scramble(torch.FloatTensor(data_input))
    return torch.tensor(data_input).to(torch.float).to(device), torch.tensor(data_output.astype(int)).to(device)

def scm_rand_corr(seed, prop_1=0.2, num_samples=200, entangle=False, flip_y=0.01, device='cuda'):
    # in environment e1, x1 and x2 is correlated
    rng = np.random.default_rng(seed)
    if prop_1 >1:
        num_samples_env1 = int(prop_1)
        num_samples_env2 = int(num_samples - prop_1)
    else:
        num_samples_env1 = int(np.ceil(prop_1*num_samples))
        num_samples_env2 = num_samples - num_samples_env1
    means = rng.normal(0, 1, size=num_samples_env1)
    x1 = rng.normal(means, [0.1 for i in range(len(means))])
    x2 = rng.normal(means, [0.1 for i in range(len(means))])
    target_e1 = np.asarray([int(x2_val > 0) for x2_val in x2])
    flipped = rng.choice([0,1], size =num_samples_env1, p=[1-flip_y, flip_y])
    y_flipped_e1 = [int(y^1) if z else int(y) for y,z in zip(target_e1, flipped)]
    data_input_e1 = np.hstack((np.expand_dims(x1, axis=1), np.expand_dims(x2, axis=1)))
    
    means_e2 = rng.normal(0, 1, num_samples_env2)
    means_unrelated = rng.normal(0,1, size=num_samples_env2)
    x1 = rng.normal(means_unrelated, [0.1 for i in range(len(means_unrelated))])

    x2 = rng.normal(means_e2, [0.1 for i in range(len(means_unrelated))])
    
    target_e2 = np.asarray([int(x2_val > 0) for x2_val in x2])
    data_input_e2 = np.hstack((np.expand_dims(x1, axis=1), np.expand_dims(x2, axis=1)))
                            
    flipped = rng.choice([0,1], size =num_samples_env2, p=[1-flip_y, flip_y])
    y_flipped_e2 = [int(y^1) if z else int(y) for y,z in zip(target_e2, flipped)]
    data_input = np.append(data_input_e1, data_input_e2, axis=0)
    data_output = np.append(y_flipped_e1, y_flipped_e2)
    if entangle:
        data_input = scramble(torch.FloatTensor(data_input))
    return torch.tensor(data_input).to(torch.float).to(device), torch.tensor(data_output.astype(int)).to(device)

def scm_same(seed, prop_1=0.2, num_samples=200, entangle=False, flip_y=0.01, device='cuda'):
    # in environment e1, x1 and x2 is correlated
    rng = np.random.default_rng(seed)
    num_samples_env1 = int(np.ceil(prop_1*num_samples))
    num_samples_env2 = num_samples - num_samples_env1
    means = rng.normal(0, 1, size=num_samples_env1)
    x1 = rng.normal(means, [0.5 for i in range(len(means))])
    x2 = rng.normal(means, [0.5 for i in range(len(means))])
    target_e1 = np.asarray([int(x2_val > 0) for x2_val in x2])
    flipped = rng.choice([0,1], size =num_samples_env1, p=[1-flip_y, flip_y])
    y_flipped_e1 = [int(y^1) if z else int(y) for y,z in zip(target_e1, flipped)]
    data_input_e1 = np.hstack((np.expand_dims(x1, axis=1), np.expand_dims(x2, axis=1)))
    
    means_e2 = rng.normal(0, 1, num_samples_env2)
    x1 = rng.normal(means_e2, [1 for i in range(len(means_e2))])

    x2 = rng.normal(means_e2, [1 for i in range(len(means_e2))])
    target_e2 = np.asarray([int(x2_val > 0) for x2_val in x2])
    data_input_e2 = np.hstack((np.expand_dims(x1, axis=1), np.expand_dims(x2, axis=1)))
                            
    flipped = rng.choice([0,1], size =num_samples_env2, p=[1-flip_y, flip_y])
    y_flipped_e2 = [int(y^1) if z else int(y) for y,z in zip(target_e2, flipped)]
    data_input = np.append(data_input_e1, data_input_e2, axis=0)
    data_output = np.append(y_flipped_e1, y_flipped_e2)
    if entangle:
        data_input = scramble(torch.FloatTensor(data_input))
    return torch.tensor(data_input).to(torch.float).to(device), torch.tensor(data_output.astype(int)).to(device)


def scm1(seed, env1, env2,
         num_samples):
    rng = np.random.default_rng(seed=seed)
    x_env1 = rng.multivariate_normal(
        mean=env1["means"], cov=np.eye(2)*([sd**2 for sd in env1["sds"]]), size=num_samples[0])
    x_env2 = rng.multivariate_normal(
        mean=env2["means"], cov=np.eye(2)*([sd**2 for sd in env2["sds"]]), size=num_samples[1])
    out = np.concatenate((x_env1, x_env2))
    target = np.asarray([int(x2 > 0.5) for x1, x2 in out])
    data = np.hstack((out, np.expand_dims(target, axis=1)))
    rng.shuffle(data)
    return out[:,0:2], out[:,-1].astype(int)


def scm1_noise(seed, env1, env2,
         num_samples, entangle=False):
    # env1[0] proportion of x1 to flip
    # env1[1] proportion of x2 to flip
    # env1[2] proportion of y to flip
    # there is a causal link between y and x2
    # x1 and x2 are confounded by e1
    # x1, x2 and y are all 0s or 1s
    e1 = 1
    rng = np.random.default_rng(seed=seed)
    x1 = [e1 for i in range(num_samples[0])]
    prop_x1 = env1[0]
    flipped = rng.choice([0,1], size=num_samples[0], p=[1-prop_x1, prop_x1])
    x1_flipped = [int(y^1) if z else int(y) for y,z in zip(x1, flipped)]
    prop_x2 = env1[1]
    flipped2 = rng.choice([0,1], size =num_samples[0], p=[1-prop_x2, prop_x2])
    x2 = [e1 for i in range(num_samples[0])]
    x2_flipped = [int(y^1) if z else int(y) for y,z in zip(x2, flipped2)]
    y = [x> 0.5 for x in x2_flipped]  
    prop_y = env1[2]
    flipped = rng.choice([0,1], size = num_samples[0], p=[1-prop_y, prop_y])
    y_flipped = [int(y^1) if z else int(y) for y,z in zip(y, flipped)]
    data_e1 = np.hstack((np.expand_dims(x1_flipped, axis=1), np.expand_dims(x2_flipped, axis=1), np.expand_dims(y_flipped, axis=1)))
 
    e2 = 1
    rng = np.random.default_rng(seed=seed)
    x1 = [0 for i in range(num_samples[1])]
    prop_x1 = env2[0]
    flipped = rng.choice([0,1], size = num_samples[1], p=[1-prop_x1, prop_x1])
    x1_flipped = [int(y^1) if z else int(y) for y,z in zip(x1, flipped)]
    prop_x2 = env2[1]
    flipped2 = rng.choice([0,1], size = num_samples[1], p=[1-prop_x2, prop_x2])

    x2 = [e2 for i in range(num_samples[1])]
    x2_flipped = [int(y^1) if z else int(y) for y,z in zip(x2, flipped2)]
    y = [x> 0.5 for x in x2_flipped]  
    prop_y = env2[2]
    flipped = rng.choice([0,1], size=num_samples[1], p=[1-prop_y, prop_y])
    y_flipped = [int(y^1) if z else int(y) for y,z in zip(y, flipped)]
    data_e2 = np.hstack((np.expand_dims(x1_flipped, axis=1), np.expand_dims(x2_flipped, axis=1), np.expand_dims(y_flipped, axis=1)))
    data = np.append(data_e1, data_e2, axis=0)
    if entangle:
        data_input = scramble(torch.FloatTensor(data[:, 0:2]))
        
    else:
        data_input = data[:, 0:2]
    return torch.tensor(data_input).to(torch.float), torch.tensor(data[:,-1].astype(int))



def entangled_image(seed, env1, env2,
         num_samples):
    rng = np.random.default_rng(seed=seed)
    x_env1 = rng.multivariate_normal(
        mean=env1["means"], cov=np.eye(2)*([sd**2 for sd in env1["sds"]]), size=num_samples[0])
    x_env2 = rng.multivariate_normal(
        mean=env2["means"], cov=np.eye(2)*([sd**2 for sd in env2["sds"]]), size=num_samples[1])
    out = np.concatenate((x_env1, x_env2))
    target = np.asarray([int(x2 > 0.5) for x1, x2 in out])
    data = np.hstack((out, np.expand_dims(target, axis=1)))
    rng.shuffle(data)

    return np.expand_dims(out, 1), target.astype(int)


def entangled(seed, env1, env2,
         num_samples, scramble):
    e1 = 1
    rng = np.random.default_rng(seed=seed)
    x1 = [e1 for i in range(num_samples[0])]
    prop_x1 = env1[0]
    flipped = rng.choice([0,1], size=num_samples[0], p=[1-prop_x1, prop_x1])
    x1_flipped = [int(y^1) if z else int(y) for y,z in zip(x1, flipped)]
    prop_x2 = env1[1]
    flipped2 = rng.choice([0,1], size =num_samples[0], p=[1-prop_x2, prop_x2])

    x2 = [e1 for i in range(num_samples[0])]
    x2_flipped = [int(y^1) if z else int(y) for y,z in zip(x2, flipped2)]
    y = [x> 0.5 for x in x2_flipped]  
    prop_y = env1[2]
    flipped = rng.choice([0,1], size = num_samples[0], p=[1-prop_y, prop_y])
    y_flipped = [int(y^1) if z else int(y) for y,z in zip(y, flipped)]
    data_e1 = np.hstack((np.expand_dims(x1_flipped, axis=1), np.expand_dims(x2_flipped, axis=1), np.expand_dims(y_flipped, axis=1)))
 
    e2 = 0
    rng = np.random.default_rng(seed=seed)
    x1 = [e2 for i in range(num_samples[1])]
    prop_x1 = env2[0]
    flipped = rng.choice([0,1], size = num_samples[1], p=[1-prop_x1, prop_x1])
    x1_flipped = [int(y^1) if z else int(y) for y,z in zip(x1, flipped)]
    prop_x2 = env2[1]
    flipped2 = rng.choice([0,1], size = num_samples[1], p=[1-prop_x2, prop_x2])

    x2 = [e2 for i in range(num_samples[1])]
    x2_flipped = [int(y^1) if z else int(y) for y,z in zip(x2, flipped2)]
    y = [x> 0.5 for x in x2_flipped]  
    prop_y = env2[2]
    flipped = rng.choice([0,1], size=num_samples[1], p=[1-prop_y, prop_y])
    y_flipped = [int(y^1) if z else int(y) for y,z in zip(y, flipped)]
    data_e2 = np.hstack((np.expand_dims(x1_flipped, axis=1), np.expand_dims(x2_flipped, axis=1), np.expand_dims(y_flipped, axis=1)))
    data = np.append(data_e1, data_e2, axis=0)
    rng.shuffle(data)
    data = data[:,0:2]
    out = scramble(torch.FloatTensor(data))
    target = data[:,-1]

    return out, target.astype(int)

def scm_band(seed, env1, env2,
         num_samples):
    rng = np.random.default_rng(seed=seed)
    x_env1 = rng.multivariate_normal(
        mean=env1["means"], cov=np.eye(2)*([sd**2 for sd in env1["sds"]]), size=num_samples[0])
    x_env2 = rng.multivariate_normal(
        mean=env2["means"], cov=np.eye(2)*([sd**2 for sd in env2["sds"]]), size=num_samples[1])
    out = np.concatenate((x_env1, x_env2))
    target = np.asarray([int(x2 > 0.5 and x2 < 1.0) for x1, x2 in out])
    data = np.hstack((out, np.expand_dims(target, axis=1)))
    rng.shuffle(data)
    return out[:,0:2], out[:,-1].astype(int)

def scm2(seed, env1: NamedTuple=Environments(means=(0.25, 0.4, 0.1), sds=(0.075, 0.075, 0.075)), 
        env2: NamedTuple=Environments(means=(0.75, 0.6, 0.8), sds=(0.075, 0.075, 0.075)),
        num_samples: Tuple=(100, 100)):
    rng = np.random.default_rng(seed=seed)
    x_env1 = rng.multivariate_normal(
        mean=env1.means, cov=np.eye(3)*([sd**2 for sd in env1.sds]), size=num_samples[0])
    x_env2 = rng.multivariate_normal(
        mean=env2.means, cov=np.eye(3)*([sd**2 for sd in env1.sds]), size=num_samples[1])
    out = np.concatenate((x_env1, x_env2))
    mean = np.sum(out[:,1] + out[:,2])/(len(out))
    rng.shuffle(out)
    target = [int(x2+x3 >mean) for x1, x2, x3 in out]
    return out, target    

def scm_ac(seed:int, env1: Tuple=(0.25, 0.20),
           env2: Tuple=(0.20, 0.25),
           prop_1 = 0.2, num_samples=200):
    # env1[0] is the proportion to flip y
    # env1[1] is the proportion to slip x2
    # x1 is causally linked to y and is sampled from a random normal
    # no confounding
    rng = np.random.default_rng(seed)
    num_samples_env1 = int(np.ceil(prop_1*num_samples))
    x1 = rng.normal(
        0.5, 0.075, size=num_samples_env1)
    target = np.asarray([(x > 0.5) for x in x1])
    proportion_flip = env1[0]
    flipped = rng.choice([0,1], size = len(target), p=[1-proportion_flip, proportion_flip])
    target_flipped = [int(y^1) if z else int(y) for y,z in zip(target, flipped)]
    target = target_flipped

    proportion = env1[1]
    to_flip = rng.choice([0,1], size = len(target), p=[1-proportion, proportion])
    x2_flipped = [int(y^1) if z else int(y) for y,z in zip(target, to_flip)]

    data_env1 = np.hstack((np.expand_dims(x1, axis=1), np.expand_dims(x2_flipped, axis=1), np.expand_dims(target, axis=1)))

    num_samples_env2 = num_samples - num_samples_env1
    x1_env2 = rng.normal(
        0.5, 0.075, size=num_samples_env2)
    target = [(x > 0.5) for x in x1_env2]
    proportion = env2[0]
    flipped = rng.choice([0,1], size = len(target), p=[1-proportion, proportion])
    target_flipped = [int(y^1) if z else int(y) for y,z in zip(target, flipped)]
    target = target_flipped
    proportion = env2[1]
    flipped_x2 = rng.choice([0,1], size = len(target), p=[1-proportion, proportion])
    x2_flipped = [int(y^1) if z else int(y) for y,z in zip(target, flipped_x2)]
    data_env2 = np.hstack((np.expand_dims(x1_env2, axis=1), np.expand_dims(x2_flipped, axis=1), np.expand_dims(target, axis=1)))
    data_combined = np.append(data_env1, data_env2, axis=0)
    rng.shuffle(data_combined)    
    return torch.tensor(data_combined[:,0:2]).to(torch.float), torch.tensor(data_combined[:,-1].astype(int))

if __name__ == '__main__':
    sd = 0.075
    env1 = Environments(means=(0.25, 0.4), sds=(sd, sd))
    env2 = Environments(means=(0.75, 0.6), sds=(sd, sd))
    out = scm1(0, env1, env2)
    plt.axhline(y=0.5, color='r')
    plt.ylabel("causal variable")
    plt.xlabel("non causal variable")
    plt.scatter(out[:,0], out[:,1])
    plt.show()
