import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch 
import wandb
import pandas as pd
from dataclasses import dataclass

from scm_class import piif_noise, fiif_noise
from irm.colored_mnist.main import MLP_IRM_SIMULATED, train, test_erm_irm
from tools import pretty_print

@dataclass
class TrainConfig:
    data: str
    seed: int 
    data_size_train: int 
    data_size_test: int 
    lr: float 
    project_name: str
    irm: bool
    l2_regulariser_weight: float
    penalty_weight: float
    penalty_anneal_iters: float
    steps: int
    wandb: bool

class TrainConfigDefault:
    data: str
    seed: int = 0 
    data_size_train: int = 1000
    data_size_test: int = 1000
    lr: float = 1e-2
    project_name: str = 'test_adv'
    irm: bool = False
    l2_regulariser_weight: float = 1e-3
    penalty_weight: float = 1e4
    penalty_anneal_iters: float = 100
    steps: int = 500
    wandb: bool = False

def main(tc):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if tc.wandb: 
        run = wandb.init(
            project=tc.project_name,
            settings=wandb.Settings(start_method='fork')
    )
    if tc.data == 'piif_noise':
        data_class = piif_noise(device, tc.data_size_train)    
    if tc.data == 'fiif_noise':    
        data_class = fiif_noise(device, tc.data_size_train)

    data_train, target_train = data_class.generate_one_env(
        tc.seed, noise_x1=tc.noise_x1, noise_x2=tc.noise_x2)
    data_test, target_test = data_class.generate_one_env(
        tc.seed + 1, noise_x1=tc.noise_x1, noise_x2=tc.noise_x2)
    env_train, env_test = data_class.return_envs(
        [data_train], [target_train], [data_test], [target_test])
    groups_test = [tc.data_size_test]    

    model = MLP_IRM_SIMULATED()
    model.to(device)
    train_acc, test_acc, worst, train_acc_weighted = train(model, 
                                                           env_train, env_test,
                                                           tc.irm, tc.lr,
                                                           groups_test,
                                                           tc.l2_regulariser_weight,
                                                           tc.penalty_weight,
                                                           tc.penalty_anneal_iters)
    test_acc_causal_masked, test_acc_worst_causal_masked, ta_standard = test_erm_irm(
        model, env_test, groups_test, mask='causal')

    test_acc_sp_masked, test_acc_worst_sp_masked, ta_standard = test_erm_irm(
        model, env_test, groups_test, mask='sp')

    if tc.wandb: 
        wandb.run.summary.update({"test acc": test_acc,
                                  "test_acc_causal_masked": test_acc_causal_masked,
                                  "test_acc_sp_masked": test_acc_sp_masked,
                                  "train_acc": train_acc})
    
    return train_acc_weighted, test_acc, test_acc_causal_masked, test_acc_sp_masked, worst

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--data', default='scm_i')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_size_train', type=int, default=1000)
    parser.add_argument('--data_size_test', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--project_name', type=str, default='causal_al')
    parser.add_argument('--irm', action='store_true')
    parser.add_argument('--l2_regulariser_weight', type=float,default=0.001)
    parser.add_argument('--penalty_anneal_iters', type=int, default=100)
    parser.add_argument('--penalty_weight', type=float, default=10000.0)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--wandb',  action='store_true')
    
    args = parser.parse_args()
    config = TrainConfig(**vars(args))
    train_acc_w, test_acc, test_c_masked, test_sp_masked, worst = main(config)
    column_names = ["train ac w",
                    "test acc", "test acc c mskd", "test acc sp mskd", "worst acc"]
    df = pd.DataFrame(columns=column_names)
    df1 = pd.DataFrame({"train ac w":train_acc_w.detach().cpu().item(),
                        "test acc": test_acc.detach().cpu().item(),
                        "test acc c mskd": test_c_masked.detach().cpu().item(),
                        "test acc sp mskd": test_sp_masked.detach().cpu().item(),
                        "worst acc": worst}, index=[0])
    df = pd.concat([df, df1])
    # df for printing
    pretty_print("train ac w",
                 "test acc", "t/ac c mskd", "t/ac sp mskd", "worst")
    pretty_print(
        train_acc_w.detach().cpu().item(),
        test_acc.detach().cpu().item(),
        test_c_masked.detach().cpu().item(),
        test_sp_masked.detach().cpu().item(),
        worst,
    )
    

