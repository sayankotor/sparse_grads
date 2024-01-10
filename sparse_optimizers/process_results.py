import json
import os
import numpy as np
import pandas as pd

if __name__ == '__main__':
    tasks =  ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']
    random_seeds = [42, 3705, 2023, 7, 3612]

    task2metric_for_best_model = {
        'cola': 'matthews_correlation',
        'mnli': 'accuracy',
        'mrpc': 'accuracy',
        'qnli': 'accuracy',
        'qqp': 'accuracy',
        'rte': 'accuracy',
        'sst2': 'accuracy',
        'stsb': 'pearson',
        'wnli': 'accuracy'
    }

    log_dir = './logs/lora'

    metrics = {}

    for task in tasks:
        log_file = os.path.join(log_dir, f'{task}.json')

        with open(log_file, 'r') as f:
            logs = json.load(f)

        metrics[task] = np.mean([logs[str(seed)]['val'][f'eval_{task2metric_for_best_model[task]}'] for seed in random_seeds])

    print(metrics)
    print(np.mean(list(metrics.values())))