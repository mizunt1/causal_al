from scm import scm1, scm_ac
import torch
import numpy as np

def scm1_data(option, seed):
    rng = np.random.default_rng(seed)
    if option == "equal":
        # train and test distributions are equal
        data = torch.tensor(scm1(0, {"means":(0.25, 0.4), "sds":(0.075, 0.075)},
                                 {"means":(0.75, 0.6), "sds":(0.075, 0.075)},
                                 num_samples=[100, 100])).to(torch.float)
        target = [(x2 > 0.5) for x1, x2 in data]
        proportion = 0.48
        flipped = rng.choice([0,1], size = len(target), p=[1-proportion, proportion])
        target_flipped = [int(~y) if z else int(y) for y,z in zip(target, flipped)]
        
        target = torch.tensor(target_flipped)

        data_test = torch.tensor(scm1(1, num_samples=[100, 100])).to(torch.float)
        target_test = torch.tensor([int(x2 > 0.5) for x1, x2 in data_test])
        return data, target, data_test, target_test
    elif option == "shift":
        seed = 3
        # train distribution is unbalanced in their environments

        data = torch.tensor(scm1(2+seed, num_samples=[25, 175])).to(torch.float)
        target = torch.tensor([int(x2 > 0.5) for x1, x2 in data])

        data_test = torch.tensor(scm1(1+seed, num_samples=[100, 100])).to(torch.float)
        target_test = torch.tensor([int(x2 > 0.5) for x1, x2 in data_test])
        return data, target, data_test, target_test

    elif option == "no_cf": 
        # test distribution doesnt have cofounding whereas train does. Data is balanced
        data = torch.tensor(scm1(0, {"means":(0.25, 0.4), "sds":(0.075, 0.075)},
                                 {"means":(0.75, 0.6), "sds":(0.075, 0.075)}, 
                                 num_samples=(100, 100))).to(torch.float)
        target = torch.tensor([int(x2 > 0.5) for x1, x2 in data])

        data_test = torch.tensor(
            scm1(1, {"means":(0.5, 0.4), "sds":(0.075, 0.075)}, {"means":(0.5, 0.4), "sds":(0.075, 0.075)}, (100,100)))
        target_test = torch.tensor(np.array([x2 > 0.5 for x1, x2 in data_test]))
        return data, target, data_test, target_test
    elif option =="reversed":
        # cofounding is reversed for test
        data = torch.tensor(scm1(0, {"means":(0.25, 0.4), "sds":(0.075, 0.075)},
                                 {"means":(0.75, 0.6), "sds":(0.075, 0.075)}, 
                                 num_samples=(100, 100))).to(torch.float)

        target = [(x2 > 0.5) for x1, x2 in data]
        proportion = 0.40
        flipped = rng.choice([0,1], size = len(target), p=[1-proportion, proportion])
        target_flipped = [int(~y) if z else int(y) for y,z in zip(target, flipped)]

        target = torch.tensor(target_flipped)
        data_test = torch.tensor(
            scm1(1, {"means":(0.25, 0.6), "sds":(0.075, 0.075)}, {"means":(0.75, 0.4), "sds":(0.075, 0.075)})).to(torch.float)
        target_test = torch.tensor(np.array([x2 > 0.5 for x1, x2 in data_test]))
        print("proportion flipped {}".format(proportion))
        return data, target, data_test, target_test
    else:
        print("option does not exist")

def scm_ac_data(option, seed):
    train_dist1 = (0.40, 0.05)
    train_dist2 = (0.05, 0.40)
    prop = 0.5
    data, target = scm_ac(seed, train_dist1, train_dist2, prop_1=prop)
    data = torch.tensor(data).to(torch.float)
    target = torch.tensor(target)
    if option == "shift":
        test_dist1 = (0.05,0.40)
        test_dist2 = test_dist1
        prop = 0.5
    elif option == "same":
        test_dist1 = (0.25, 0.20)
        test_dist2= (0.10, 0.20)
    data_test, target_test = sem_ac(seed+1, test_dist1, test_dist2, prop)
    data_test = torch.tensor(data_test).to(torch.float)
    target_test = torch.tensor(target_test)
    lr = 1e-2
    input_size = 2

    
