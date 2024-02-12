from itertools import product

# tasks = ['stsb', 'cola', 'mrpc', 'qnli', 'rte', 'sst2', 'wnli']
# tasks = ['mnli', 'qqp']
tasks = ['stsb', 'cola', 'qnli', 'sst2', 'wnli']
model_paths = ['roberta-large',]
run_types = ['ft', 'lora', 'sparse']
# run_types = ['sparse',]

with open('config_large.txt', 'w') as f:
    f.write('ArrayTaskId\tTask\tmodel\ttype\n')
    for i, (task, model_path, run_type) in enumerate(product(tasks, model_paths, run_types)):
        f.write(f'{i}\t{task}\t{model_path}\t{run_type}\n')
