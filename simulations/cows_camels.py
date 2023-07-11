import numpy as np
import torch
import math

torch.manual_seed(3)
class CC:
    """
    Cows and camels
    """
    def __init__(self, dim_inv, dim_spu, n_envs, non_lin=False):
        self.scramble = torch.eye(dim_inv + dim_spu)
        self.scramble_non_lin = torch.nn.Sequential(torch.nn.Linear((dim_inv + dim_spu), int((dim_inv + dim_spu)/2)), torch.nn.ReLU())
        self.dim_inv = dim_inv
        self.dim_spu = dim_spu
        self.dim = dim_inv + dim_spu
        self.non_lin = non_lin

        self.task = "classification"
        self.envs = {}

        if n_envs >= 2:
            self.envs = {
                'E0': {"p": 0.95, "s": 0.3},
                'E1': {"p": 0.97, "s": 0.5}
            }
        if n_envs >= 3:
            self.envs["E2"] = {"p": 0.99, "s": 0.7}
        if n_envs > 3:
            for env in range(3, n_envs):
                self.envs["E" + str(env)] = {
                    "p": torch.zeros(1).uniform_(0.9, 1).item(),
                    "s": torch.zeros(1).uniform_(0.3, 0.7).item()
                }
        print("Environments variables:", self.envs)

        # foreground is 100x noisier than background
        self.snr_fg = 1e-2
        self.snr_bg = 1

        # foreground (fg) denotes animal (cow / camel)
        cow = torch.ones(1, self.dim_inv)
        self.avg_fg = torch.cat((cow, cow, -cow, -cow))

        # background (bg) denotes context (grass / sand)
        grass = torch.ones(1, self.dim_spu)
        self.avg_bg = torch.cat((grass, -grass, -grass, grass))

    def sample(self, n=1000, env="E0", split="train"):
        p = self.envs[env]["p"]
        s = self.envs[env]["s"]
        w = torch.Tensor([p, 1 - p] * 2) * torch.Tensor([s] * 2 + [1 - s] * 2)
        i = torch.multinomial(w, n, True)
        x = torch.cat((
            (torch.randn(n, self.dim_inv) /
                math.sqrt(10) + self.avg_fg[i]) * self.snr_fg,
            (torch.randn(n, self.dim_spu) /
                math.sqrt(10) + self.avg_bg[i]) * self.snr_bg), -1)

        if split == "test":
            # for test, the spurrious features are somehow mixed around, so unrelated to correlated features
            x[:, self.dim_spu:] = x[torch.randperm(len(x)), self.dim_spu:]

        inputs = x @ self.scramble
        outputs = x[:, :self.dim_inv].sum(1, keepdim=True).gt(0).float()
        # inputs are spurious and invariant features scrambled together. ie the image
        # outputs is the invariant class. y. 
        if self.non_lin:
            inputs = self.scramble_non_lin(x).detach()
        return inputs, outputs

    def mix_train_test(self, proportion, total_samples, no_confounding_test=False):
        majority_data = int(np.floor(total_samples*proportion))
        minority_data = total_samples - majority_data
        data2, target2 = self.sample(n=minority_data, split='test')
        data1, target1 = self.sample(n=majority_data)
        print("minority env train: ",len(target2) )
        data_train = torch.cat((data1, data2))
        target_train = torch.cat((target1, target2))
        if no_confounding_test:
            minority_data = int(total_samples /2)
            majority_data = minority_data
        data1_test, target1_test = self.sample(n=minority_data)
        data2_test, target2_test = self.sample(n=majority_data, split='test')
        print("minority env test: ",len(target1_test) )
        data_test = torch.cat((data1_test, data2_test))
        target_test = torch.cat((target1_test, target2_test))
        return data_train, target_train, data_test, target_test
        

        
if __name__ == "__main__":
    scm = CC(5,5,2)
    
    inputs, outputs = scm.sample()
    inputs2, outputs2 = scm.sample()
    import pdb
    pdb.set_trace()
