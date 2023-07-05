from scm import scm1, scm_ac, scm_band, scm1_noise, entangled, entangled_image
import torch
import numpy as np

def scm_band_data(seed):
    data, target = scm_band(seed, {"means":(0.25, 0.4), "sds":(0.075, 0.075)},
                                 {"means":(0.75, 0.6), "sds":(0.075, 0.075)},
                                 num_samples=[100, 100])
    data, target = torch.tensor(data).to(torch.float), torch.tensor(target)

    data_test, target_test = scm_band(seed+1, {"means":(10, 0.4), "sds":(0.075, 0.075)},
                                 {"means":(-2, 0.6), "sds":(0.075, 0.075)},
                                 num_samples=[100, 100])
        
    data_test, target_test = torch.tensor(data_test).to(torch.float), torch.tensor(target)
    return data, target, data_test, target_test

def scm1_noise_data(seed):
    data, target = scm1_noise(seed, (0.1, 0.1, 0.1), (0.1, 0.1, 0.1), (100, 100))
    data, target = torch.tensor(data).to(torch.float), torch.tensor(target)
    data_test, target_test = scm1_noise(seed+1, (0.4, 0.1, 0.1), (0.4, 0.1, 0.1), (100, 100))
    data_test, target_test = torch.tensor(data_test).to(torch.float), torch.tensor(target_test)
    return data, target, data_test, target_test

def entangled_data(seed):
    data, target = entangled(seed, (0.1, 0.1, 0.1), (0.1, 0.1, 0.1), (100, 100))
    data, target = torch.tensor(data).to(torch.float), torch.tensor(target)
    data_test, target_test = entangled(seed+1, (0.4, 0.1, 0.1), (0.4, 0.1, 0.1), (100, 100))
    data_test, target_test = torch.tensor(data_test).to(torch.float), torch.tensor(target_test)
    return data, target, data_test, target_test

def entangled_image_data(seed):
    data, target = entangled_image(seed, {'means': (0.1, 0.1), 'sds':(0.1, 0.1)}, {'means': (0.1, 0.1), 'sds':(0.1, 0.1)}, (100, 100))
    data, target = torch.tensor(data).to(torch.float), torch.tensor(target)
    data_test, target_test = entangled_image(seed+1, {'means':(0.4, 0.1, 0.1), 'sds':(0.4, 0.1, 0.1)}, (100, 100))
    data_test, target_test = torch.tensor(data_test).to(torch.float), torch.tensor(target_test)
    return data, target, data_test, target_test

def scm1_data(seed, option):
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
    elif option == "no_overlap":
        # train and test distributions are equal
        data, target = scm1(seed, {"means":(0.25, 0.4), "sds":(0.075, 0.075)},
                                 {"means":(0.75, 0.6), "sds":(0.075, 0.075)},
                                 num_samples=[100, 100])
        data, target = torch.tensor(data).to(torch.float), torch.tensor(target)

        data_test, target_test = scm1(seed+1, {"means":(10, -3), "sds":(0.075, 0.075)},
                                 {"means":(20, 10), "sds":(0.075, 0.075)},
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
            scm1(1, {"means":(0.25, 0.6), "sds":(0.075, 0.075)},
                 {"means":(0.75, 0.4), "sds":(0.075, 0.075)})).to(torch.float)
        data_test, target_test = torch.tensor(data_test).to(torch.float), torch.tensor(target)
        return data, target, data_test, target_test
    else:
        print("option does not exist")

def scm_ac_data(seed, option):
    train_dist1 = (0.40, 0.05)
    train_dist2 = (0.05, 0.40)
    prop = 0.5
    data, target = scm_ac(seed, train_dist1, train_dist2, prop_1=prop, num_samples=200)
    data = torch.tensor(data).to(torch.float)
    target = torch.tensor(target)
    if option == "shift":
        test_dist1 = (0.05,0.40)
        test_dist2 = test_dist1
        prop = 0.5
    elif option == "equal":
        test_dist1 = (0.25, 0.20)
        test_dist2= (0.10, 0.20)
    data_test, target_test = scm_ac(seed+1, test_dist1, test_dist2, prop_1=prop, num_samples=200)
    data_test = torch.tensor(data_test).to(torch.float)
    target_test = torch.tensor(target_test)
    return data, target, data_test, target_test
    
    
