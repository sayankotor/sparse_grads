import optuna


for study in ['mnli_ft_bert-base-uncased', 'sst2_ft_bert-base-uncased', 'sst2_lora_roberta-base',
              'mnli_lora_bert-base-uncased', 'sst2_lora_bert-base-uncased', 'sst2_sparse_bert-base-uncased',
              'sst2_sparse_roberta-base', 'qqp_sparse_roberta-base', 'mnli_lora_roberta-base', 'mnli_sparse_roberta-base',
              'qqp_lora_roberta-base', 'mnli_sparse_bert-base-uncased', 'qqp_sparse_bert-base-uncased', 'qqp_lora_bert-base-uncased']:


    optuna.copy_study(
        from_study_name=study,
        from_storage="sqlite:///sparse-grad.db",
        to_storage=f"postgresql+psycopg2://sparse-grad:mooKah4i@doge.skoltech.ru/sparse-grad",
    )