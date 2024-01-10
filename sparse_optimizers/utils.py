import re
from typing import Callable, Optional
from functools import wraps

import numpy as np
import torch

from transformers import EvalPrediction
import evaluate as ev

from sparse_grad_matrix_sparse import Tucker_Decomposition, replace_bert_layers

def sparse_grad_linear(model, UV_dict):
    print ("create bert with sparse grads")
    model = replace_bert_layers(model, UV_dict)
    print ("created bert with sparse grads")
    return model

def get_dataset(tokenizer, raw_dataset, dset_type = 'cola', max_length=128):
    task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    }
    sentence1_key, sentence2_key = task_to_keys[dset_type]

    def preprocess_function(examples):
            # Tokenize the texts
            args = [examples[sentence1_key],] if sentence2_key is None else [examples[sentence1_key], examples[sentence2_key]]

            result = tokenizer(*args, max_length=max_length, truncation=True, padding="max_length")

            result["label"] = examples["label"]
            return result

    tokenized_dataset = raw_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=False
            )  
    return tokenized_dataset


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


def _map_module(root: torch.nn.Module,
                func: Callable[[torch.nn.Module, str], torch.nn.Module], patt: re.Pattern,
                path: str) -> torch.nn.Module:
    for name, child in root.named_children():
        node = _map_module(child, func, patt, f'{path}/{name}')
        if node != child:
            setattr(root, name, node)
    if patt.match(path or '/'):
        print(path)
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


def _get_weight_grads(root, patt,
                path):
    grads = []
    for name, child in root.named_children():
        grads += _get_weight_grads(child, patt, f'{path}/{name}')
    if patt.match(path):
        grads.append(root.weight.grad)
    return grads

def get_weight_grads(root,
               patt = None):
    return _get_weight_grads(root, re.compile(patt or r'.*'), '')
