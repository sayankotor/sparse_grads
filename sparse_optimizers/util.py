from sparse_grad_matrix_sparse_new import Tucker_Decomposition, replace_bert_layers
import torch

from transformers import EvalPrediction
import numpy as np

import evaluate as ev

def sparse_grad_linear(model, UV_dict):
    print ("create bert with sparse grads")
    model = replace_bert_layers(model, UV_dict)
    print ("created bert with sparse grads")
    return model


def get_dataset(tokenizer, raw_dataset, dset_type = 'cola'):
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

    def convert_to_stsb_features(example_batch):
        inputs = list(zip(example_batch['sentence1'], example_batch['sentence2']))
        features = tokenizer.batch_encode_plus(
            inputs, max_length=512, truncation=True, padding="max_length")
        features["labels"] = example_batch["label"]
        return features

    def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer.batch_encode_plus(*args, max_length=128, truncation=True, padding="max_length")

            result["label"] = examples["label"]
            return result

    if (dset_type == 'stsb'):
        tokenized_dataset = raw_dataset.map(
            convert_to_stsb_features,
            batched=True,
            load_from_cache_file=False,
            )
    else:    
        tokenized_dataset = raw_dataset.map(
                    preprocess_function,
                    batched=True,
                    load_from_cache_file=False
                )  
    return tokenized_dataset


metric = ev.load("glue", 'cola')

def compute_metrics(p: EvalPrediction):
        preds_ = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds_ = np.argmax(preds_, axis=1)
        
        result = metric.compute(predictions=preds_, references=p.label_ids)
        if True:
            result["combined_score"] = np.mean(list(result.values())).item()
            return result
        else:
            return {"accuracy": (preds_ == p.label_ids).astype(np.float32).mean().item()}
        



