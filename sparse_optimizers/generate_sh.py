s = ''
device = 3
# tasks = ['cola', 'stsb', 'mrpc', 'qnli', 'rte', 'sst2']
# tasks = ['qqp',]
# tasks = ['stsb', 'mrpc', 'qnli', 'rte', 'sst2']
tasks = ['qqp',]
model_paths = ['bert-base-uncased', 'roberta-base']
run_types = ['sparse',]
n_params_list = [18000, 22000, 30000, 100000]

for task in tasks:
    for run_type in run_types:
        for model_path in model_paths:
            for n_params in n_params_list:
                s += f"""echo Running {run_type} {model_path} {task} {n_params}\n\
CUDA_VISIBLE_DEVICES='{device}' python run.py --task {task} --run_type {run_type} --model_path {model_path} --n_params {n_params}\n"""

print(s)