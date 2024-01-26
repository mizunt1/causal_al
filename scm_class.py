import torch
import numpy as np
from scm_base_class import SCM

class piif_noise(SCM):
    def __init__(self, device, num_samples):
        super().__init__(device, num_samples)
    
    def generate_one_env(self, seed, noise_x2, noise_x1):
        rng = np.random.default_rng(seed)
        x2 = rng.normal(0, 0.2, size=self.num_samples)
        y_step = rng.normal(x2, noise_x2)
        y = np.asarray([int(y_val > 0) for y_val in y_step])
        x1 = rng.normal(y, [noise_x1 for i in range(len(x2))])
        data_input_e1 = np.hstack((np.expand_dims(x1, axis=1), np.expand_dims(x2, axis=1)))
        return torch.tensor(
            data_input_e1).to(torch.float).to(self.device), torch.tensor(y.astype(int)).to(self.device)


class fiif_noise(SCM):
    def __init__(self, device, num_samples):
        super().__init__(device, num_samples)

    def generate_one_env(self, seed, noise_x1, noise_x2):
        rng = np.random.default_rng(seed)
        means = rng.normal(0, 1, size=self.num_samples)
        x1 = rng.normal(means, [noise_x1 for i in range(len(means))])
        x2 = rng.normal(means, [noise_x1 for i in range(len(means))])
        y_step = rng.normal(x2, [noise_x2 for i in range(len(means))])
        y = np.asarray([int(x2_val > 0) for x2_val in y_step])
        data_input_e1 = np.hstack(
            (np.expand_dims(x1, axis=1), np.expand_dims(x2, axis=1)))
        return torch.tensor(
            data_input_e1).to(
                torch.float).to(self.device), torch.tensor(
                    y.astype(int)).to(self.device)


class piif_complex_mech(SCM):
    def __init__(self, device, num_samples1, num_samples2):
        self.num_samples1 = num_samples1
        self.num_samples2 = num_samples2
        super().__init__(device, num_samples1)

    def generate_one_env(self, seed, mech_type):
        if mech_type == 1:
            num_samples = self.num_samples1
        elif mech_type == 2:
            num_samples = self.num_samples2
        else:
            num_samples = self.num_samples1 + self.num_samples2
        noise_x1 = 0.0
        noise_x2 = 0.0
        rng = np.random.default_rng(seed)
        x2 = rng.normal(0, 0.2, size=num_samples)
        y_step = rng.normal(x2, noise_x2)
        y = np.asarray([int(y_val > 0) for y_val in y_step])
        if mech_type == 1:
            x1 = rng.normal(2*y + 2, [noise_x1 for i in range(len(x2))])
        elif mech_type == 2:
            x1 = rng.normal(-2*y +1 , [noise_x1 for i in range(len(x2))])
        else:
            x1 = rng.normal(-3*y -2 , [noise_x1 for i in range(len(x2))])
        data_input_e1 = np.hstack(
            (np.expand_dims(x1, axis=1), np.expand_dims(x2, axis=1)))
        return torch.tensor(
            data_input_e1).to(
                torch.float).to(self.device), torch.tensor(
                    y.astype(int)).to(self.device)

class piif_overlap_mech(SCM):
    def __init__(self, device, num_samples1, num_samples2):
        self.num_samples1 = num_samples1
        self.num_samples2 = num_samples2
        super().__init__(device, num_samples1)

    def generate_one_env(self, seed, mech_type):
        if mech_type == 1:
            num_samples = self.num_samples1
        elif mech_type == 2:
            num_samples = self.num_samples2
        else:
            num_samples = self.num_samples1 + self.num_samples2
        noise_x1 = 0.0
        noise_x2 = 0.0
        rng = np.random.default_rng(seed)
        x2 = rng.normal(0, 0.2, size=num_samples)
        y_step = rng.normal(x2, noise_x2)
        y = np.asarray([int(y_val > 0) for y_val in y_step])
        if mech_type == 1:
            x1_step = [1 if y_val == 1 else -1 for y_val in y]
            x1 = rng.normal(x1_step, [noise_x1 for i in range(len(x2))])
        elif mech_type == 2:
            x1_step = [1 if y_val == 0 else -2 for y_val in y]
            x1 = rng.normal(x1_step, [noise_x1 for i in range(len(x2))])
        else:
            x1_step = [2 if y_val == 1 else 3 for y_val in y]
            x1 = rng.normal(x1_step, [noise_x1 for i in range(len(x2))])

        data_input_e1 = np.hstack(
            (np.expand_dims(x1, axis=1), np.expand_dims(x2, axis=1)))
        return torch.tensor(
            data_input_e1).to(
                torch.float).to(self.device), torch.tensor(
                    y.astype(int)).to(self.device)

    
    def generate_one_env_test(self, seed):
        noise_x1 = 0.1
        noise_x2 = 0.2
        rng = np.random.default_rng(seed)
        x2 = rng.normal(0, 0.2, size=self.num_samples)
        y_step = rng.normal(x2, noise_x2)
        y = np.asarray([int(y_val > 0) for y_val in y_step])
        x1 = rng.normal(y, [noise_x1 for i in range(len(x2))])
        data_input_e1 = np.hstack(
            (np.expand_dims(x1, axis=1), np.expand_dims(x2, axis=1)))
        return torch.tensor(
            data_input_e1).to(
                torch.float).to(self.device), torch.tensor(
                    y.astype(int)).to(self.device)
    

