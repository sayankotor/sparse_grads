import argparse
import torch
import transformers
import torch

from datasets import load_dataset
import pandas as pd

from transformers import Trainer
import evaluate as ev
import numpy as np

# Trainers
from transformers import TrainingArguments, Trainer, EvalPrediction

# Custom trainers
from trainers_custom import TrainerBert1, TrainerBert2, TrainerDoubleOpt

# Custom needed function

from util_trainer import pre_trainer_function, trainer_function
from util import get_dataset, sparse_grad_linear


# Create argument parser
parser = argparse.ArgumentParser(prog="Test suite for GPT models")
parser.add_argument("--sparse_grad", action="store_true")
parser.add_argument("--cuda", type=int)

# Parse arguments
args = parser.parse_args()

# Select device
device = torch.device("cuda:{}".format(args.cuda))

# Load the model and set it into training mode
from transformers import AutoConfig, BertConfig, AutoModelForSequenceClassification, AutoTokenizer

path_name = r"bert-base-uncased"

#dataset

dataset_cola = load_dataset('glue', 'cola')

label_list = dataset_cola["train"].features["label"].names
num_labels = len(label_list)


config = AutoConfig.from_pretrained(
    path_name,
    num_labels=num_labels,
)

model = AutoModelForSequenceClassification.from_pretrained(
    path_name,
    config=config,
)

tokenizer = AutoTokenizer.from_pretrained(path_name)


#dataset

tokenized_dataset = get_dataset(tokenizer, dataset_cola, dset_type = 'cola')


# metrics

metric = ev.load("glue", 'cola')

import pickle



        
## training arguments

training_args1 = TrainingArguments(
    learning_rate=5e-5,
    num_train_epochs=1,
    evaluation_strategy="steps",
    eval_steps=100,
    max_steps = 11,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=128,
    save_steps=1000,
    overwrite_output_dir=True,
    output_dir="./bert_stsb_128",
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=True,
    seed=297104,
    report_to='none',
    )

training_args2 = TrainingArguments(
    learning_rate=2e-5,
    num_train_epochs=3,
    evaluation_strategy="steps",
    skip_memory_metrics = False,
    eval_steps=100,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=128,
    save_steps=1000,
    overwrite_output_dir=True,
    output_dir="./bert_stsb_128",
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=True,
    seed=297104,
    report_to='none',
    )
    

model.to(device)
# Report memory before training iteration
torch.cuda.synchronize()
print("Memory for the model and input dataset: {} MB".format( \
        torch.cuda.max_memory_allocated(args.cuda)/1024./1024.))


if args.sparse_grad:
    UV_dict = pre_trainer_function(model, training_args1, tokenized_dataset)
    with open('filename.pickle', 'wb') as handle:
        pickle.dump(UV_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    torch.cuda.synchronize()
    print("Memory after pre-training sparse grad: {} MB".format( \
            torch.cuda.max_memory_allocated(args.cuda)/1024./1024.)) 
    model = sparse_grad_linear(model, UV_dict)
    torch.cuda.synchronize()
    print("Memory after re-creating model: {} MB".format( \
        torch.cuda.max_memory_allocated(args.cuda)/1024./1024.))
    
    trainer_function(model, training_args2, tokenized_dataset, is_sparse_grad = True)


else:
    trainer_function(model, training_args2, tokenized_dataset)

# Report memory after training operation
torch.cuda.synchronize()
print("Memory after training: {} MB".format( \
        torch.cuda.memory_allocated(args.cuda)/1024./1024.))


# Report memory after training iteration
torch.cuda.synchronize()
print("Peak memory usage: {} MB".format( \
        torch.cuda.max_memory_allocated(args.cuda)/1024./1024.))
