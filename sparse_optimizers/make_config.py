from itertools import product

tasks = ['cola', 'mrpc', 'qnli', 'rte', 'sst2', 'stsb', 'wnli', 'mnli', 'qqp']
model_paths = ['roberta-base', 'bert-base-uncased']
run_types = ['ft', 'lora', 'sparse']

with open('config.txt', 'w') as f:
    f.write('ArrayTaskId\tTask\tmodel\ttype\n')
    for i, (task, model_path, run_type) in enumerate(product(tasks, model_paths, run_types)):
        f.write(f'{i}\t{task}\t{model_path}\t{run_type}\n')
