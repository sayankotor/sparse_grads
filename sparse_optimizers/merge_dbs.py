import optuna

for model_path in ['roberta-large']:
    # for task in ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']:
    for task in ['cola', 'mrpc', 'qnli', 'rte', 'sst2', 'stsb', 'wnli']:
    # for task in ['mnli', 'qqp']:
        for run_type in ['ft', 'lora', 'sparse']:
            study = f'{task}_{run_type}_{model_path}'

            optuna.copy_study(
                from_study_name=study,
                from_storage="postgresql+psycopg2://sparse-grad:BiW7oocu@10.1.3.21:5432/sparse-grad",
                to_storage="sqlite:///sparse-grad_final.db"
            )