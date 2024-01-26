import torch
from abc import ABC, abstractmethod

class SCM(ABC):
    def __init__(self, device, num_samples):
        self.device = device
        self.num_samples = num_samples
    
    @abstractmethod
    def generate_one_env(self):
        pass

    def standardise_envs(self, datas):
        """
        insert a tuple of datas and standardise them all, 
        returning another list of standardised list of datas. 
        """
        data_input = torch.cat(datas)
        mean = data_input.mean(dim=0, keepdim=True)
        std = data_input.std(dim=0, keepdim=True)
        data_ret = []
        for data in datas:
            (data - mean)/std
            data_ret.append(data)
        return data_ret

    def return_envs(self, data_train, target_train, data_test, target_test):
        """
        insert a list of data_trains, list of target_trains,
        list of data_test, list of target_test, returns
        list of dictionary, each dictionary corresponding to a 
        different environment.
        """
        data_train = self.standardise_envs(data_train)
        data_test = self.standardise_envs(data_test)
        env_train = []
        for data, target in zip(data_train, target_train): 
            env_train.append(
                {'images': data, 'labels': target.float()})
        env_test = {'images': data_test[0], 'labels': target_test[0].float()}
        return env_train, env_test


        
