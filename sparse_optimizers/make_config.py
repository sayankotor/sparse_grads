from itertools import product

# tasks = ['stsb', 'cola', 'mrpc', 'qnli', 'rte', 'sst2', 'wnli']
# tasks = ['mnli', 'qqp']
tasks = ['mrpc']#, 'rte', 'sst2']
model_paths = ['roberta-large',]
# run_types = ['ft', 'lora', 'sparse']
run_types = ['lora',]

with open('config_large.txt', 'w') as f:
    f.write('ArrayTaskId\tTask\tmodel\ttype\n')
    for i, (task, model_path, run_type, _) in enumerate(product(tasks, model_paths, run_types, range(1))):
        f.write(f'{i}\t{task}\t{model_path}\t{run_type}\n')
