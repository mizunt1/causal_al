from scm import scm1, scm_ac
import torch
import numpy as np

def scm1_data(option, seed):
    rng = np.random.default_rng(seed)
    if option == "equal":
        # train and test distributions are equal
        data, target = scm1(seed, {"means":(0.25, 0.4), "sds":(0.075, 0.075)},
                                 {"means":(0.75, 0.6), "sds":(0.075, 0.075)},
                                 num_samples=[100, 100])
        data, target = torch.tensor(data).to(torch.float), torch.tensor(target)

        data_test, target_test = scm1(seed+1, {"means":(0.25, 0.4), "sds":(0.075, 0.075)},
                                 {"means":(0.75, 0.6), "sds":(0.075, 0.075)},
                                 num_samples=[100, 100])
        
        data_test, target_test = torch.tensor(data_test).to(torch.float), torch.tensor(target)
        return data, target, data_test, target_test
    elif option == "shift":
        # train distribution is unbalanced in their environments
        data, target = scm1(seed, {"means":(0.25, 0.4), "sds":(0.075, 0.075)},
                            {"means":(0.75, 0.6), "sds":(0.075, 0.075)},
                                 num_samples=[25, 175])
        data, target = torch.tensor(data).to(torch.float), torch.tensor(target)
        data_test, target_test = scm1(seed, {"means":(0.25, 0.4), "sds":(0.075, 0.075)},
                            {"means":(0.75, 0.6), "sds":(0.075, 0.075)},
                            num_samples=[175, 25])
        data_test, target_test = torch.tensor(data_test).to(torch.float), torch.tensor(target)
        return data, target, data_test, target_test

    elif option == "no_cf": 
        # test distribution doesnt have cofounding whereas train does. Data is balanced
        data, target = torch.tensor(scm1(0, {"means":(0.25, 0.4), "sds":(0.075, 0.075)},
                                 {"means":(0.75, 0.6), "sds":(0.075, 0.075)}, 
                                 num_samples=(100, 100))).to(torch.float)
        data, target = torch.tensor(data).to(torch.float), torch.tensor(target)
        data_test,target = torch.tensor(
            scm1(1, {"means":(0.5, 0.4), "sds":(0.075, 0.075)}, {"means":(0.5, 0.4), "sds":(0.075, 0.075)}, (100,100)))
        data_test, target_test = torch.tensor(data_test).to(torch.float), torch.tensor(target)
        return data, target, data_test, target_test
    elif option =="reversed":
        # cofounding is reversed for test
        data, target = torch.tensor(scm1(0, {"means":(0.25, 0.4), "sds":(0.075, 0.075)},
                                 {"means":(0.75, 0.6), "sds":(0.075, 0.075)}, 
                                 num_samples=(100, 100))).to(torch.float)
        data, target = torch.tensor(data).to(torch.float), torch.tensor(target)
        data_test, target_test = torch.tensor(
            scm1(1, {"means":(0.25, 0.6), "sds":(0.075, 0.075)}, {"means":(0.75, 0.4), "sds":(0.075, 0.075)})).to(torch.float)
        data_test, target_test = torch.tensor(data_test).to(torch.float), torch.tensor(target)
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

    
