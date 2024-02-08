import optuna

for model_path in ['bert-base-uncased', 'roberta-base']:
    for task in ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']:
        for run_type in ['ft', 'lora', 'sparse']:
            study = f'{task}_{run_type}_{model_path}'

            optuna.copy_study(
                from_study_name=study,
                from_storage="postgresql+psycopg2://sparse-grad:mooKah4i@doge.skoltech.ru/sparse-grad",
                to_storage="sqlite:///sparse-grad_final.db"
            )