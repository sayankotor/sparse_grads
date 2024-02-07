import json
import os
import numpy as np
import pandas as pd

if __name__ == '__main__':
    tasks = ['cola',]
    # tasks =  ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']
    random_seeds = [42, 3705, 2023]

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

    log_dir = './logs'


    for model_path in ['roberta-base', 'bert-base-uncased']:
        print('\n\n', model_path)
        for run_type in ['ft', 'lora', 'sparse']:
            metrics = {}

            for task in tasks:
                log_file = os.path.join(log_dir, run_type, f'{model_path}_{task}.json')
                
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        logs = json.load(f)

                    try:
                        metrics[task] = np.mean([logs[str(seed)]['val'][f'eval_{task2metric_for_best_model[task]}'] for seed in random_seeds])
                    except:
                        pass

            print(run_type)
            print(metrics)
            print(np.mean(list(metrics.values())))