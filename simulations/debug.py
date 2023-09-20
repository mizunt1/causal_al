from scm import scm1_noise, scm_continuous_confounding
from al_loop import train, Model,test, calc_score_debug
from argparse import ArgumentParser
import torch 
import torch.nn.functional as F

parser = ArgumentParser()
parser.add_argument('--entangle', action='store_true')
parser.add_argument('--random', action='store_true')
parser.add_argument('--cont', action='store_true')
parser.add_argument('--no_confound_test', action='store_true')
parser.add_argument('--scm_ac', action='store_true')
parser.add_argument('--standard_train', action='store_true')
parser.add_argument('--non_lin_entangle', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--data_size', type=int, default=200)
parser.add_argument('--n_largest', type=int, default=2)
parser.add_argument('--al_iters', type=int, default=10)
parser.add_argument('--proportion', type=float, default=0.50)
parser.add_argument('--num_epochs', type=int, default=3010)

args = parser.parse_args()
seed = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_models = 10
data, target = scm_continuous_confounding(
    seed, prop_1=1.0, entangle=args.non_lin_entangle, flip_y=0, num_samples=args.data_size)
data = data.to(device)
target = target.to(device)

data_test, target_test = scm_continuous_confounding(
    seed + 1, prop_1=1.0, entangle=args.non_lin_entangle, flip_y=0, num_samples=args.data_size)
data_test = data_test.to(device)
target_test = target_test.to(device)

data_ood, target_ood = scm_continuous_confounding(
    seed+1, prop_1=0, entangle=args.non_lin_entangle, flip_y=0, num_samples=args.data_size)

data_ood = data_ood.to(device)
target_ood = target_ood.to(device)

input_size = 2
lr = 1e-3
models = Model(input_size, num_models)
models.to(device)

train_acc = train(args.num_epochs, models, data, target, lr, device, ensemble=False)
test_acc = test(models, data_test, target_test, device, ensemble=False)

print('train accuracy: {}'.format(train_acc))
print('test accuracy: {}'.format(test_acc))

data_ood = data_ood.to(device)
target_ood = target_ood.to(device)

test_acc_ood = test(models, data_ood, target_ood, device, ensemble=False)
print('test accuracy ood: {}'.format(test_acc_ood))

models.train()

preds_ood = [F.softmax(models(data_ood), dim=1).cpu().clone().detach().numpy() for _ in range(10)]

preds = [F.softmax(models(data), dim=1).cpu().clone().detach().numpy() for _ in range(10)]
preds_test = [F.softmax(models(data_test), dim=1).cpu().clone().detach().numpy() for _ in range(10)]
var_b_ood, var_w_ood, mean_score_ood = calc_score_debug(preds_ood)
var_b, var_w, mean_score = calc_score_debug(preds)
var_b_test, var_w_test, mean_score_test = calc_score_debug(preds_test)
print("var between ood {}, var within ood {} mean score ood {} ".format(var_b_ood, var_w_ood, mean_score_ood))
print("var between train {}, var within train {} mean score train {}".format(var_b, var_w, mean_score))
print("var between test {}, var within test {} mean score test {}".format(var_b_test, var_w_test, mean_score_test))
