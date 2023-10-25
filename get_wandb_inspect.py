import wandb
import json
from collections import defaultdict
import pandas as pd

entity, project = "mizunt", "causal_al_rand_data"
api = wandb.Api(timeout=19)
runs = api.runs(entity + "/" + project)
arts = defaultdict(list)
arts2 = defaultdict(int)

def get_runs(proportion, al_iters, random):
    return api.runs(entity + "/" + project,
        filters={'$and': [{
            'state': {'$eq': 'finished'},
            'config.non_lin_entangle': {'$eq':True},
            'config.proportion': {'$eq': proportion},
            'config.al_iters': {'$eq': al_iters},
            'config.random': {'$eq': random},
#            'created_at': {'$gt': '2023-02-03T16:00:00'},
        }]}
    )
proportions = [0.5, 0.9, 0.8, 0.95]
al_iters = [ 5, 7, 10, 15, 20]
num_seeds = 3

for prop in proportions:
    for al_iter in al_iters:
        runs_random = get_runs(prop, al_iter, True)
        runs_al = get_runs(prop, al_iter, False)
        diction_r = defaultdict()
        diction_al = defaultdict()
        for runs_r in runs_random:
            inf = json.loads(runs_r.json_config)
            test_acc = runs_r.summary['test acc']
            seed = inf['seed']['value']
            diction_r[seed]=test_acc
        for run_al in runs_al:
            inf = json.loads(run_al.json_config)
            test_acc = run_al.summary['test acc']
            seed = inf['seed']['value']
            diction_al[seed] = test_acc
        i = 0
        for seed in range(num_seeds):
            if diction_al[seed] > diction_r[seed]:
                i += 1
                pass
            else:
                break
        if i == 3:
            print("prop {}, al_iter {}".format(prop, al_iter))
        
                
            
