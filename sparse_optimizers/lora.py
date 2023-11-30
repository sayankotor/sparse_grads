import re
import os
import json
from pathlib import Path
from math import sqrt
from functools import partial, wraps
from typing import Callable, Optional
import numpy as np

import torch
from torch.nn import Parameter

import optuna
from optuna.pruners import BasePruner, NopPruner
from optuna.trial import TrialState
from optuna.study import StudyDirection

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback, EvalPrediction
import evaluate as ev
from datasets import load_dataset, Value

from util import get_dataset#, compute_metrics


class MetricsComputer:
    def __init__(self, task):
        self.task = task
        self.metric = ev.load("glue", task)

    def __call__(self, p: EvalPrediction):
            preds_ = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            if self.task != 'stsb':
                preds_ = np.argmax(preds_, axis=1)
            
            result = self.metric.compute(predictions=preds_, references=p.label_ids)
            return result

    
class LoRALinear(torch.nn.Module):
    r"""A reimplementation of LoRA linear for parameter efficient training.

    .. note::
       In the original paper author initialize the first factor with zeros and
       the second one from Gaussian distribution. We use uniform Kaiming
       initialization for the second factor instead.

    .. note::

       Factorized representation of additive correction has order unlike the
       one reported in the original paper. Factors are stored as list of
       :math:`A`, :math:`B^\top` to make application to input more convenient.
    """

    def __init__(self, in_features: int, out_features: int, rank: int,
                 bias: bool = True, device=None, dtype=None,
                 scale: float = 1.0):
        super().__init__()
        self.rank = rank
        self.scale = scale / rank

        opts = {'dtype': dtype, 'device': device}

        # Create frozen linear layer.
        self.linear = torch.nn.Linear(in_features, out_features, bias, **opts)
        self.linear.requires_grad_(False)

        # Create trainable factorized coorection.
        self.factors = torch.nn.ParameterList([
            Parameter(torch.empty((in_features, rank), **opts)),
            Parameter(torch.empty((out_features, rank), **opts)),
        ])

        # Initialize only correction.
        self.reset_parameters(False)

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    def reset_parameters(self, recursively=True) -> None:
        if recursively:
            self.linear.reset_parameters()
        torch.nn.init.kaiming_uniform_(self.factors[0], a=sqrt(5))
        torch.nn.init.zeros_(self.factors[1])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert not self.linear.weight.requires_grad, \
            'Weights of kernel must be freezed.'
        output = self.linear(input)
        correction = (input @ self.factors[0]) @ self.factors[1].T
        return output + self.scale * correction

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear, rank: int, **kwargs):
        """Creates linear layer with additive lowr-rank term from linear layer.
        """
        kwargs['bias'] = kwargs.get('bias', linear.bias is not None)
        self = cls(linear.in_features, linear.out_features, rank, **kwargs)
        self.linear = linear
        self.linear.weight.requires_grad = False
        return self

    def to_linear(self) -> torch.nn.Linear:
        """Merge an additive term to a basic weight matrix."""
        raise NotImplementedError

def _map_module(root: torch.nn.Module,
                func: Callable[[torch.nn.Module, str], torch.nn.Module], patt: re.Pattern,
                path: str) -> torch.nn.Module:
    for name, child in root.named_children():
        node = _map_module(child, func, patt, f'{path}/{name}')
        if node != child:
            setattr(root, name, node)
    if patt.match(path or '/'):
        root = func(root, path or '/')
    return root

def map_module(root: torch.nn.Module,
               func: Callable[[torch.nn.Module, str], torch.nn.Module],
               patt: Optional[str] = None) -> torch.nn.Module:
    """Function ``map_module`` applies a function to each leaf of module tree
    which matches to a specified pattern.

    Parameters
    ----------
    root : torch.nn.Module
        Module to modify.
    func : callable
        Function to be applied to every module (or matched to pattern) in
        module tree.
    patt : str, optional
        Pattern to filter modules by path in module tree.

    Returns
    -------
    torch.nn.Module
        Module modified in-place.
    """
    @wraps(func)
    def func_safe(*args, **kwargs):
        node = func(*args, **kwargs)
        if not isinstance(node, torch.nn.Module):
            raise ValueError('Mapped result must be toch.nn.Module type '
                             f'but given {type(node)}.')
        return node

    return _map_module(root, func_safe, re.compile(patt or r'.*'), '')

def convert_low_rank(rank: int, module: torch.nn.Module, path: str):
    if not isinstance(module, torch.nn.Linear):
        raise ValueError('Only linear layer can be converted: '
                         f'type={type(module)}.')
    return LoRALinear.from_linear(module, rank)

def convert_model(model, path: str, rank: int):
    return map_module(
        model, partial(convert_low_rank, rank), path)

def get_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    trainable_params_names = []
    lora_params = 0
    for param_name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_params_names.append(re.sub(r".\d+.", '._.', param_name))
            if re.match(r".*\.factors\.\d", param_name):
                lora_params += param.numel()
    print(
        f"lora params: {lora_params} || trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
    print('trainable params:', set(trainable_params_names))

    return trainable_params, lora_params, all_param

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

def make_model(model_path, enable_lora, lora_modules_path, num_labels,
    lora_rank, verbose=False):

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
        model = convert_model(model, lora_modules_path, lora_rank)

        trainable_params, lora_params, all_param = get_trainable_parameters(model)

    return model, trainable_params, lora_params, all_param

def make_trainer(model, task, tokenized_dataset, output_dir, seed, metric_for_best_model, eval_steps, batch_size=16, lr=2e-5, num_epoches=1, max_steps=-1):
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
        lr_scheduler_type='constant',
        warmup_ratio=0.,
        )

    validation_split_name = 'validation' if 'validation' in tokenized_dataset.keys() else 'validation_matched'

    trainer = Trainer(  
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset[validation_split_name],
            compute_metrics=MetricsComputer(task),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )

    return trainer

def train(run, model_path, task, enable_lora, lora_modules_path, seed, batch_size, eval_steps, max_length, lr, num_epoches, max_steps, metric_for_best_model, lora_rank=None, verbose=False):
    tokenized_dataset, num_labels = make_dataset(model_path, dataset_path='glue', dataset_name=task, max_length=max_length)
    model, trainable_params, lora_params, all_param = make_model(model_path, enable_lora, lora_modules_path, num_labels, lora_rank, verbose=verbose)
    model = model.to('cuda')
    output_dir = str(Path('model') / f'glue-{task}')
    checkpoint_dir = str(Path('model') / f'checkpt_{run}')

    trainer = make_trainer(model, task, tokenized_dataset, checkpoint_dir, seed, eval_steps=eval_steps, metric_for_best_model=metric_for_best_model, batch_size=batch_size, lr=lr, num_epoches=num_epoches, max_steps=max_steps)
    
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

    metrics = {'train': train_result.metrics,
               'val': eval_result,
               'end_mem_bytes': end_mem_bytes,
               'peak_mem_bytes': peak_mem_bytes,
               'trainable_params': trainable_params,
               'lora_params': lora_params,
               'all_param': all_param}

    return eval_result[f'eval_{metric_for_best_model}'], metrics


def optuna_objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
    lr = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)

    eval_steps = int(task2evalsteps[task] * 16. / batch_size)

    val_metric, _ = train(run, model_path, task, enable_lora=True, lora_modules_path=lora_modules_path, seed=seed, batch_size=batch_size, eval_steps=eval_steps, max_length=max_length, lr=lr, num_epoches=1, max_steps=-1, metric_for_best_model=task2metric_for_best_model[task], lora_rank=lora_rank, verbose=verbose)
    return val_metric



if __name__ == '__main__':
    model_path = r"bert-base-uncased"
    lora_modules_path = '/bert/encoder/layer/\d+/(output/dense|intermediate/dense)'
    dataset_path = 'glue'
    lora_rank = 7
    
    # tasks, run =  ['cola', 'mnli', 'mrpc'], 0
    # tasks, run = ['qnli', 'qqp', 'rte'], 1
    tasks, run = ['sst2', 'stsb', 'wnli'], 2

    # seed = 34
    max_length = 128
    verbose = True

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

    task2hyperparams = {
        'cola': {'lr': 4e-5, 'batch_size': 8},
        'mnli': {'lr': 2e-5, 'batch_size': 8},
        'mrpc': {'lr': 1.2e-5, 'batch_size': 4},
        'qnli': {'lr': 4e-5, 'batch_size': 16},
        'qqp': {'lr': 2e-5, 'batch_size': 16},
        'rte': {'lr': 3.5e-5, 'batch_size': 8},
        'sst2': {'lr': 1e-5, 'batch_size': 8},
        'stsb': {'lr': 2e-5, 'batch_size': 4},
        'wnli': {'lr': 5e-3, 'batch_size': 32},
    }

    random_seeds = [42, 3705, 2023, 7, 3612]

    log_dir = './logs/lora'

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    for task in tasks:
        log_file = os.path.join(log_dir, f'{task}.json')
        
        for seed in random_seeds:
            _, metrics = train(run, model_path, task, enable_lora=True, lora_modules_path=lora_modules_path, seed=seed, metric_for_best_model=task2metric_for_best_model[task],
            batch_size=task2hyperparams[task]['batch_size'], eval_steps=int(task2evalsteps[task] * 16. / task2hyperparams[task]['batch_size']), max_length=max_length, lr=task2hyperparams[task]['lr'], num_epoches=1, max_steps=-1, lora_rank=lora_rank, verbose=True)

            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = {}

            with open(log_file, 'w') as f:
                logs.update({seed: metrics})
                json.dump(logs, f)


        # study_name = task  # Unique identifier of the study.
        # storage_name = f"sqlite:///optuna_new.db"

        # study = optuna.create_study(study_name=study_name, direction="maximize", storage=storage_name, load_if_exists=True)
        # study.optimize(optuna_objective, n_trials=40, timeout=60*60*40, n_jobs=1, gc_after_trial=True)
    