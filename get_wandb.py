import wandb
import json
from collections import defaultdict
import pandas as pd


entity, project = "mizunt", "causal_al-simulations"
api = wandb.Api(timeout=19)
runs = api.runs(entity + "/" + project)
arts = defaultdict(list)
arts2 = defaultdict(int)

def get_runs():
    return api.runs(entity + "/" + project,
        filters={'$and': [{
            'state': {'$eq': 'finished'},
            'config.non_lin_entangle': {'$eq':True}
        }]}
    )
proportions = [0.5, 0.9, 0.8, 0.95]
al_iters = [3, 5, 7, 10, 15]
data_sizes = [200, 300, 500, 600, 1000]
num_seeds = 5
# data will be stored in a tree structure
# order is rand, proportion, al_iter, data_size, seed
data = {}
for train in ['rand', 'al']:
    data[train] = {}
    for proportion in proportions: 
        data[train][str(proportion)] = {}
        for al_iter in al_iters:
            data[train][str(proportion)][str(al_iter)] = {}
            for data_size in data_sizes: 
                seeds = [str(i) for i in range(num_seeds)]
                data[train][str(proportion)][str(al_iter)][str(data_size)] = {"test acc": dict(zip(seeds, [None for i in seeds])), "minority": dict(zip(seeds, [None for i in seeds]))}               

first = True
runs = get_runs()
import pdb
pdb.set_trace()
for run_ in runs:
    inf = json.loads(run_.json_config)
    try:
        random = inf['random']['value']
    except KeyError as e:
        continue
        
    if random:
        train = 'rand'
    else:
        train = 'al'
        #if first:
        #    import pdb
        #    pdb.set_trace()
        #    first = False
    if len(run_.summary.keys()) != 0:
        data[train][str(inf['proportion']['value'])][str(inf['al_iters']['value'])][str(inf['data_size']['value'])]['test acc'][str(inf['seed']['value'])] = run_.summary['test acc']
        data[train][str(inf['proportion']['value'])][str(inf['al_iters']['value'])][str(inf['data_size']['value'])]['minority'][str(inf['seed']['value'])] = run_.summary['prop minority selected for train']

proportions = [0.5, 0.9, 0.8, 0.95]
al_iters = [3, 5, 7, 10]
data_sizes = [200, 500, 1000]
num_seeds = 5


data_sizes_df = {}
for data_size in data_sizes:
    data_size = str(data_size)
    props = []
    iters = []
    test_actives = []
    test_randoms = []
    min_actives = []
    min_randoms = []

    for prop in proportions:
        prop = str(prop)
        for al_iter in al_iters:
            al_iter = str(al_iter)
            iters.append(al_iter)
            columns = ['prop', 'al iter', 'test active', 'test random', 'min active', 'min random']
            test_active = [i for i in data['al'][prop][al_iter][data_size]['test acc'].values() if i is not None]
            test_active = (sum(test_active)/(len(test_active) + 1e-15))
            test_random = [i for i in data['rand'][prop][al_iter][data_size]['test acc'].values() if i is not None]
            test_random = sum(test_random)/(len(test_random) + 1e-15)

            min_active = [i for i in data['al'][prop][al_iter][data_size]['minority'].values() if i is not None]
            min_active = sum(min_active)/ (len(min_active) +1e-15)

            min_random = [i for i in data['rand'][prop][al_iter][data_size]['minority'].values() if i is not None]
            min_random = sum(min_random)/ (len(min_random) +1e-15)
            props.append(prop)
            test_actives.append(test_active)
            test_randoms.append(test_random)
            min_actives.append(min_active)
            min_randoms.append(min_random)
        data_columns = [props, iters, test_actives, test_randoms, min_actives, min_randoms]
        df = pd.DataFrame(dict(zip(columns, data_columns)))
        data_sizes_df[data_size] = df
print(data_sizes_df['1000'].to_latex(index=False,
                  formatters={"name": str.upper},
                  float_format="{:.3f}".format,))


