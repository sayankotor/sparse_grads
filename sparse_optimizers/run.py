import re
import os
import json
from pathlib import Path
import numpy as np
import uuid
import argparse
from shutil import rmtree

import torch

import optuna
from optuna.pruners import BasePruner, NopPruner
from optuna.trial import TrialState
from optuna.study import StudyDirection

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback, EvalPrediction
import evaluate as ev
from datasets import load_dataset, Value

from utils import get_dataset, MetricsComputer, get_trainable_parameters
from lora_utils import convert_model as convert2lora
from sparse_utils import convert_model as convert2sparse, get_UV_dict
from sparse_grad_matrix_sparse import SparseGradLinearIntermediate, SparseGradLinearOutput
# from trainers_custom import TrainerBert2 as SparseTrainer
from trainers_custom import TrainerDoubleOpt as SparseTrainer


def make_dataset(model_path_name, dataset_path, dataset_name, max_length):
    dataset = load_dataset(dataset_path, dataset_name)

    label_list = dataset["train"].features["label"]
    if isinstance(label_list, Value):
        num_labels = 1
    else:
        num_labels = len(label_list.names)

    tokenizer = AutoTokenizer.from_pretrained(model_path_name)

    tokenized_dataset = get_dataset(tokenizer, dataset, dset_type=dataset_name, max_length=max_length)
    return tokenized_dataset, num_labels

def make_model(model_path, tokenized_dataset, lr, batch_size, seed, enable_lora, enable_sparse, output_modules_path, intermediate_modules_path, num_labels,
    lora_rank, sparse_n_params, verbose=False):

    assert not(enable_lora & enable_sparse)

    config = AutoConfig.from_pretrained(
        model_path,
        num_labels=num_labels,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
    )

    trainable_params, lora_params, all_param = get_trainable_parameters(model)

    if enable_lora:
        replace_modules_path = f'({output_modules_path}|{intermediate_modules_path})'
        model = convert2lora(model, replace_modules_path, lora_rank)
        trainable_params, lora_params, all_param = get_trainable_parameters(model)

    if enable_sparse:
        UV_dict = get_UV_dict(model, task, output_modules_path, intermediate_modules_path, n_stack_grads=360, tokenized_dataset=tokenized_dataset, lr=lr, batch_size=batch_size, seed=seed, max_steps=11)
        model = convert2sparse(model, output_modules_path, UV_dict['output'], SparseGradLinearOutput, n_params=sparse_n_params)
        model = convert2sparse(model, intermediate_modules_path, UV_dict['interm'], SparseGradLinearIntermediate, n_params=sparse_n_params)
        # model = replace_bert_layers(model, UV_dict)

    return model, trainable_params, lora_params, all_param

def make_trainer(model, task, enable_sparse, tokenized_dataset, output_dir, seed, metric_for_best_model, eval_steps, batch_size=16, lr=2e-5, num_epoches=1, max_steps=-1):
    training_args = TrainingArguments(
        learning_rate=lr,
        num_train_epochs=num_epoches,
        max_steps=max_steps,
        evaluation_strategy="steps",
        skip_memory_metrics = False,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        logging_steps=eval_steps // 2,
        metric_for_best_model=metric_for_best_model,
        load_best_model_at_end=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy='steps',
        save_total_limit=1,
        overwrite_output_dir=True,
        output_dir=output_dir,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=True,
        seed=seed,
        report_to='none',
        max_grad_norm=None,
        lr_scheduler_type='constant',
        warmup_ratio=0.,
        )
    validation_split_name = 'validation' if 'validation' in tokenized_dataset.keys() else 'validation_matched'
    if enable_sparse:
        trainer_cls = SparseTrainer
    else:
        trainer_cls = Trainer

    trainer = trainer_cls(  
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset[validation_split_name],
                compute_metrics=MetricsComputer(task),
                callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
            )

    if enable_sparse:
        trainer.make_grad_bank()

    return trainer

def train(model_path, task, enable_lora, enable_sparse, output_modules_path, intermediate_modules_path, seed, batch_size, eval_steps, max_length, lr, num_epoches, max_steps, metric_for_best_model, lora_rank=None, sparse_n_params=None, verbose=False):
    tokenized_dataset, num_labels = make_dataset(model_path, dataset_path='glue', dataset_name=task, max_length=max_length)
    model, trainable_params, lora_params, all_param = make_model(model_path, tokenized_dataset, lr, batch_size, seed, enable_lora, enable_sparse, output_modules_path, intermediate_modules_path, num_labels, lora_rank, sparse_n_params, verbose=verbose)
    model = model.to('cuda')
    output_dir = str(Path('model') / f'glue-{task}')
    checkpt_name = uuid.uuid4().hex
    checkpoint_dir = str(Path('model') / f'checkpt_{checkpt_name}')
    

    trainer = make_trainer(model, task, enable_sparse, tokenized_dataset, checkpoint_dir, seed, eval_steps=eval_steps, metric_for_best_model=metric_for_best_model, batch_size=batch_size, lr=lr, num_epoches=num_epoches, max_steps=max_steps)
    
    train_result = trainer.train()
    eval_result = trainer.evaluate()

    trainer.log_metrics("train", train_result.metrics)
    trainer.log_metrics("eval", eval_result)

    print(eval_result)

    # Report memory after training operation
    torch.cuda.synchronize()
    end_mem_bytes = torch.cuda.memory_allocated()
    print("Memory after training: {} MB".format( \
            end_mem_bytes/1024./1024.))


    # Report memory after training iteration
    torch.cuda.synchronize()
    peak_mem_bytes = torch.cuda.max_memory_allocated()
    print("Peak memory usage: {} MB".format( \
            peak_mem_bytes/1024./1024.))

    rmtree(checkpoint_dir)

    metrics = {'train': train_result.metrics,
               'val': eval_result,
               'end_mem_bytes': end_mem_bytes,
               'peak_mem_bytes': peak_mem_bytes,
               'trainable_params': trainable_params,
               'lora_params': lora_params,
               'all_param': all_param}

    return eval_result[f'eval_{metric_for_best_model}'], metrics


def optuna_objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    lr = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)

    eval_steps = int(task2evalsteps[task] * 16. / batch_size)

    val_metric, _ = train(model_path, task, enable_lora=enable_lora, enable_sparse=enable_sparse, output_modules_path=model2replace_modules_path[model_path]['output'], intermediate_modules_path=model2replace_modules_path[model_path]['intermediate'], seed=seed, batch_size=batch_size, eval_steps=eval_steps, max_length=max_length, lr=lr, num_epoches=20, max_steps=-1, metric_for_best_model=task2metric_for_best_model[task], lora_rank=model2params[model_path]['lora_rank'], sparse_n_params=model2params[model_path]['sparse_n_params'], verbose=verbose)
    return val_metric



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--run_type', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('-optimize', action='store_true')


    args = parser.parse_args()
    
    # model_path = r"bert-base-uncased"
    # model_path = r"roberta-base"
    model_path = args.model_path
    
    model2replace_modules_path = {'bert-base-uncased': {'output': '/bert/encoder/layer/\d+/output/dense',
                                                        'intermediate': '/bert/encoder/layer/\d+/intermediate/dense'},
                                  'roberta-base': {'output': '/roberta/encoder/layer/\d+/output/dense',
                                                   'intermediate': '/roberta/encoder/layer/\d+/intermediate/dense'},
                                  'roberta-large': {'output': '/roberta/encoder/layer/\d+/output/dense',
                                                   'intermediate': '/roberta/encoder/layer/\d+/intermediate/dense'}}

    model2params = {'bert-base-uncased': {'lora_rank': 7, 'sparse_n_params': 28000},
                    'roberta-base': {'lora_rank': 7, 'sparse_n_params': 28000},
                    'roberta-large': {'lora_rank': 10, 'sparse_n_params': 50000}}
    
    dataset_path = 'glue'
    
    # tasks, run =  ['cola', 'mnli', 'mrpc']
    # tasks, run = ['qnli', 'qqp', 'rte']
    # tasks, run = ['sst2', 'stsb', 'wnli']

    task = args.task

    max_length = 128
    verbose = True


    run_type = args.run_type

    if run_type == 'lora':
        enable_lora = True
        enable_sparse = False
    elif run_type == 'sparse':
        enable_lora = False
        enable_sparse = True
    elif run_type == 'ft':
        enable_lora = False
        enable_sparse = False
    else:
        raise ValueError('Wrong run_type')

    task2evalsteps = {
        'cola': 100,
        'mnli': 10000,
        'mrpc': 100,
        'qnli': 100,
        'qqp': 10000,
        'rte': 10,
        'sst2': 1000,
        'stsb': 100,
        'wnli': 10,
    }

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

    log_dir = f'./logs/{run_type}'

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    if args.optimize:
        # Finding optimal params
        seed = 34

        study_name = f'{task}_{run_type}_{model_path}'  # Unique identifier of the study.
        # storage_name = "postgresql+psycopg2://sparse-grad:mooKah4i@doge.skoltech.ru/sparse-grad"
        storage_name = "postgresql+psycopg2://sparse-grad:BiW7oocu@10.1.3.21:5432/sparse-grad"

        study = optuna.create_study(study_name=study_name, direction="maximize", storage=storage_name, load_if_exists=True)
        study.optimize(optuna_objective, n_trials=40, timeout=60*60*24*10, n_jobs=1, gc_after_trial=True)
    else:
        # # Calculating mean metrics with optimal params 

        task2hyperparams = \
        {
            'bert-base-uncased':
            {
                'ft': 
                    {
                        'cola': {'lr': 1.01e-5, 'batch_size': 16},
                        'mnli': {'lr': 1.51e-5, 'batch_size': 32},
                        'mrpc': {'lr': 1.90e-5, 'batch_size': 16},
                        'qnli': {'lr': 1.91e-5, 'batch_size': 16},
                        'qqp': {'lr': 5.11e-6, 'batch_size': 16},
                        'rte': {'lr': 3.05e-5, 'batch_size': 32},
                        'sst2': {'lr': 1.33e-5, 'batch_size': 16},
                        'stsb': {'lr': 2.70e-5, 'batch_size': 16},
                        'wnli': {'lr': 9.13e-4, 'batch_size': 32},
                    },
                'lora':
                    {
                        'cola': {'lr': 2.60e-5, 'batch_size': 32},
                        'mnli': {'lr': 1.00e-5, 'batch_size': 16},
                        'mrpc': {'lr': 3.11e-5, 'batch_size': 16},
                        'qnli': {'lr': 1.73e-5, 'batch_size': 16},
                        'qqp': {'lr': 9.25e-6, 'batch_size': 32},
                        'rte': {'lr': 3.84e-5, 'batch_size': 16},
                        'sst2': {'lr': 4.07e-5, 'batch_size': 32},
                        'stsb': {'lr': 1.12e-4, 'batch_size': 32},
                        'wnli': {'lr': 1.57e-5, 'batch_size': 32},
                    },
                'sparse':
                    {
                        'cola': {'lr': 3.15e-5, 'batch_size': 32},
                        'mnli': {'lr': 6.07e-6, 'batch_size': 32},
                        'mrpc': {'lr': 1.22e-5, 'batch_size': 32},
                        'qnli': {'lr': 1.94e-5, 'batch_size': 16},
                        'qqp': {'lr': 1.41e-5, 'batch_size': 32},
                        'rte': {'lr': 6.81e-5, 'batch_size': 16},
                        'sst2': {'lr': 1.47e-5, 'batch_size': 32},
                        'stsb': {'lr': 1.24e-4, 'batch_size': 32},
                        'wnli': {'lr': 2.52e-6, 'batch_size': 32},
                    }
            },
            'roberta-base':
            {
                'ft': 
                    {
                        'cola': {'lr': 1.85e-5, 'batch_size': 32},
                        'mnli': {'lr': 1.15e-6, 'batch_size':   16},
                        'mrpc': {'lr': 2.47e-5, 'batch_size': 32},
                        'qnli': {'lr': 8.38e-6, 'batch_size': 16},
                        'qqp': {'lr': 7.20e-6, 'batch_size': 32},
                        'rte': {'lr': 1.74e-5, 'batch_size': 32},
                        'sst2': {'lr': 1.02e-5, 'batch_size': 32},
                        'stsb': {'lr': 7.71e-5, 'batch_size': 32},
                        'wnli': {'lr': 2.34e-5, 'batch_size': 16},
                    }, 
                'lora':
                    {
                        'cola': {'lr': 3.20e-5, 'batch_size': 32},
                        'mnli': {'lr': 1.04e-5, 'batch_size': 32},
                        'mrpc': {'lr': 7.02e-5, 'batch_size': 32},
                        'qnli': {'lr': 5.54e-5, 'batch_size': 32},
                        'qqp': {'lr': 5.32e-6, 'batch_size': 16},
                        'rte': {'lr': 1.61e-5, 'batch_size': 16},
                        'sst2': {'lr': 1.49e-5, 'batch_size': 32},
                        'stsb': {'lr': 4.87e-5, 'batch_size': 16},
                        'wnli': {'lr': 3.11e-5, 'batch_size': 16},
                    }, 
                'sparse':
                    {
                        'cola': {'lr': 2.61e-5, 'batch_size': 32},
                        'mnli': {'lr': 1.00e-5, 'batch_size': 32},
                        'mrpc': {'lr': 4.63e-5, 'batch_size': 32},
                        'qnli': {'lr': 1.66e-5, 'batch_size': 16},
                        'qqp': {'lr': 1.40e-5, 'batch_size': 32},
                        'rte': {'lr': 1.81e-5, 'batch_size': 16},
                        'sst2': {'lr': 2.30e-5, 'batch_size': 32},
                        'stsb': {'lr': 1.87e-5, 'batch_size': 16},
                        'wnli': {'lr': 6.48e-5, 'batch_size': 32},
                    }
            },
            'roberta-large':
            {
                'ft': 
                    {
                        'cola': {'lr': 1.06e-5, 'batch_size': 16},
                        'mnli': {'lr': -1, 'batch_size': -1},
                        'mrpc': {'lr': 6.79e-6, 'batch_size': 16},
                        'qnli': {'lr': 2.35e-6, 'batch_size': 32},
                        'qqp': {'lr': -1, 'batch_size': -1},
                        'rte': {'lr': 6.49e-6, 'batch_size': 16},
                        'sst2': {'lr': 8.77e-6, 'batch_size': 32},
                        'stsb': {'lr': 1.11e-5, 'batch_size': 32},
                        'wnli': {'lr': 1.57e-6, 'batch_size': 16},
                    }, 
                'lora':
                    {
                        'cola': {'lr': 5.86e-6, 'batch_size': 16},
                        'mnli': {'lr': -1, 'batch_size': -1},
                        'mrpc': {'lr': 1.56e-5, 'batch_size': 16},
                        'qnli': {'lr': 1.19e-5, 'batch_size': 16},
                        'qqp': {'lr': -1, 'batch_size': -1},
                        'rte': {'lr': 4.17e-5, 'batch_size': 16},
                        'sst2': {'lr': 5.01e-6, 'batch_size': 16},
                        'stsb': {'lr': 2.96e-5, 'batch_size': 16},
                        'wnli': {'lr': 4.14e-5, 'batch_size': 32},
                    }, 
                'sparse':
                    {
                        'cola': {'lr': 1.27e-5, 'batch_size': 32},
                        'mnli': {'lr': 4.34e-6, 'batch_size': 32},
                        'mrpc': {'lr': 1.96e-5, 'batch_size': 16},
                        'qnli': {'lr': 5.56e-6, 'batch_size': 16},
                        'qqp': {'lr': 2.89e-6, 'batch_size': 32},
                        'rte': {'lr': 2.10e-5, 'batch_size': 32},
                        'sst2': {'lr': 1.46e-6, 'batch_size': 32},
                        'stsb': {'lr': 2.11e-5, 'batch_size': 16},
                        'wnli': {'lr': 1.32e-6, 'batch_size': 16},
                    }
            }
        }

        hyperparams = task2hyperparams[model_path][run_type][task]

        random_seeds = [42, 3705, 2023]
        # random_seeds = [37, 7777]
        log_file = os.path.join(log_dir, f'{model_path}_{task}.json')
        for seed in random_seeds:
            _, metrics = train(model_path, task,
                enable_lora=enable_lora, enable_sparse=enable_sparse,
                output_modules_path=model2replace_modules_path[model_path]['output'], intermediate_modules_path=model2replace_modules_path[model_path]['intermediate'],
                seed=seed, metric_for_best_model=task2metric_for_best_model[task],
                batch_size=hyperparams['batch_size'],
                eval_steps=int(task2evalsteps[task] * 16. / hyperparams['batch_size']),
                max_length=max_length,
                lr=hyperparams['lr'],
                num_epoches=20, max_steps=-1,
                lora_rank=model2params[model_path]['lora_rank'], sparse_n_params=model2params[model_path]['sparse_n_params'], verbose=True)

            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = {}

            with open(log_file, 'w') as f:
                logs.update({seed: metrics})
                json.dump(logs, f)
    