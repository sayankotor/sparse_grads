import re
from pathlib import Path

import torch

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, Value

from util import get_dataset, compute_metrics


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    trainable_params_names = []
    for param_name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_params_names.append(re.sub(r".\d+.", '._.', param_name))
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
    print('trainable params:', set(trainable_params_names))


def make_dataset(model_path_name, dataset_path, dataset_name):
    dataset = load_dataset(dataset_path, dataset_name)

    label_list = dataset["train"].features["label"]
    if isinstance(label_list, Value):
        num_labels = 1
    else:
        num_labels = len(label_list.names)

    tokenizer = AutoTokenizer.from_pretrained(model_path_name)

    tokenized_dataset = get_dataset(tokenizer, dataset, dset_type=dataset_name)
    return tokenized_dataset, num_labels


def make_model(model_path, enable_lora, num_labels,
    lora_rank, lora_alpha=1., lora_dropout=0.,
    verbose=False):

    config = AutoConfig.from_pretrained(
        model_path,
        num_labels=num_labels,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
    )

    if verbose:
        print_trainable_parameters(model)

    if enable_lora:
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_CLS
        )
        model = get_peft_model(model, config)

        if verbose:
            print_trainable_parameters(model)

    return model

def make_trainer(model, tokenized_dataset, output_dir, seed, batch_size=16, lr=2e-5, num_epoches=1, max_steps=-1):
    training_args = TrainingArguments(
        learning_rate=lr,
        num_train_epochs=num_epoches,
        max_steps=max_steps,
        evaluation_strategy="steps",
        skip_memory_metrics = False,
        eval_steps=100,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy='no',
        overwrite_output_dir=True,
        output_dir=output_dir,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=True,
        seed=seed,
        report_to='none',
        warmup_ratio=0.06,
        )

    validation_split_name = 'validation' if 'validation' in tokenized_dataset.keys() else 'validation_matched'

    # print('dir')
    # print(tokenized_dataset.__dict__)
    # print(tokenized_dataset.keys())
    # print(dir(tokenized_dataset))

    trainer = Trainer(  
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset[validation_split_name],
            compute_metrics=compute_metrics,
        )

    return trainer

def train(model_path, task, enable_lora, seed, batch_size, lr, num_epoches, max_steps, lora_rank=None, verbose=False):
    tokenized_dataset, num_labels = make_dataset(model_path, dataset_path='glue', dataset_name=task)
    model = make_model(model_path, enable_lora, num_labels, lora_rank, verbose=verbose).to('cuda')
    output_dir = str(Path('model') / f'glue-{task}')

    trainer = make_trainer(model, tokenized_dataset, output_dir, seed, batch_size=batch_size, lr=lr, num_epoches=num_epoches, max_steps=max_steps)
    
    train_result = trainer.train()
    eval_result = trainer.evaluate()

    trainer.log_metrics("train", train_result.metrics)
    trainer.log_metrics("eval", eval_result)

    # Report memory after training operation
    torch.cuda.synchronize()
    print("Memory after training: {} MB".format( \
            torch.cuda.memory_allocated()/1024./1024.))


    # Report memory after training iteration
    torch.cuda.synchronize()
    print("Peak memory usage: {} MB".format( \
            torch.cuda.max_memory_allocated()/1024./1024.))


if __name__ == '__main__':
    model_path_name = r"bert-base-uncased"
    dataset_path = 'glue'
    tasks = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']
    # tasks = ['cola',]
    seed = 34

    # make_model(model_path_name, enable_lora=True, num_labels=2, lora_rank=2, verbose=True)
    for task in tasks:
        train(model_path_name, task, enable_lora=True, seed=seed, batch_size=16, lr=5e-4, num_epoches=-1, max_steps=2, lora_rank=1, verbose=True)
    