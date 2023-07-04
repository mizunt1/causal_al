from typing import NamedTuple, Tuple
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
seed=0
Environments = namedtuple('Environments', ['means', 'sds'])

def scm1(seed, env1, env2,
         num_samples):
    rng = np.random.default_rng(seed=seed)
    x_env1 = rng.multivariate_normal(
        mean=env1["means"], cov=np.eye(2)*([sd**2 for sd in env1["sds"]]), size=num_samples[0])
    x_env2 = rng.multivariate_normal(
        mean=env2["means"], cov=np.eye(2)*([sd**2 for sd in env1["sds"]]), size=num_samples[1])
    out = np.concatenate((x_env1, x_env2))
    target = [int(x2 > 0.5) for x1, x2 in out]
    rng.shuffle(out)
    return out, target

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
    rng = np.random.default_rng(seed=seed)
    num_samples_env1 = int(np.ceil(prop_1*num_samples))
    x1 = rng.normal(
        0.5, 0.075, size=num_samples_env1)
    target = np.asarray([(x > 0.5) for x in x1])
    proportion_flip = env1[0]
    flipped = rng.choice([0,1], size = len(target), p=[1-proportion_flip, proportion_flip])
    target_flipped = [int(y^1) if z else int(y) for y,z in zip(target, flipped)]
    target =target_flipped
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
    return data_combined[:,0:2], (data_combined[:,-1]).astype(int)

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
