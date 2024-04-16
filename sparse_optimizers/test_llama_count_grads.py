import argparse
import torch
import transformers
import torch

from datasets import load_dataset
import pandas as pd

import pickle

from sparse_grad_matrix_sparse import Tucker_Decomposition

from transformers import TrainerCallback, TrainerCallback, TrainerState, TrainerControl

from transformers import Trainer
import evaluate as ev
import numpy as np

# Trainers
from transformers import TrainingArguments, Trainer, EvalPrediction
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, Trainer

# Custom trainers
from trainers_custom import TrainerBert2, TrainerDoubleOpt, TrainerLlama1

# Custom needed function

#from util_trainer import pre_trainer_function, trainer_function
from util import get_dataset, sparse_grad_linear

import timeit


UV_dict = {}

# Create argument parser
parser = argparse.ArgumentParser(prog="Test suite for GPT models")
parser.add_argument("--sparse_grad", action="store_true")
parser.add_argument("--cuda", type=int)

# Parse arguments
args = parser.parse_args()

# Select device
#DEVICE = torch.device("cuda:{}".format(args.cuda))

# Load the model and set it into training mode

grads1 = torch.load('grads1.pth', map_location = 'cpu')
#grads2 = torch.load('grads2.pth', map_location = 'cpu')

print (grads1.shape)
#print (grads2.shape)

start = timeit.timeit()

print ("Tucker")
_, VT, U = Tucker_Decomposition(grads1)
UV_dict.update({"up":tuple((U, VT))})
#_, VT, U = Tucker_Decomposition(grads2)
#UV_dict.update({"down":tuple((U, VT))})

end = timeit.timeit()
print(end - start)

print (U)

#with open('llama.pickle', 'wb') as handle:
        #pickle.dump(UV_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     model = sparse_grad_linear(model, UV_dict)
#del grads1, grads2


#base_model.to(device)
# Report memory before training iteration
#torch.cuda.synchronize()
#print("Memory for the model and input dataset: {} MB".format( \
        #torch.cuda.max_memory_allocated(torch.device("cpu"))/1024./1024.))


#if args.sparse_grad:
#    UV_dict = pre_trainer_function(base_model, training_args1, training_data)
#    with open('llama.pickle', 'wb') as handle:
#        pickle.dump(UV_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#    torch.cuda.synchronize()
#    print("Memory after pre-training sparse grad: {} MB".format( \
#            torch.cuda.max_memory_allocated(args.cuda)/1024./1024.))
    
#    model = sparse_grad_linear(model, UV_dict)
#    torch.cuda.synchronize()
#    print("Memory after re-creating model: {} MB".format( \
#        torch.cuda.max_memory_allocated(args.cuda)/1024./1024.))
    
    #trainer_function(model, training_args2, tokenized_dataset, is_sparse_grad = True)


#else:
#    trainer_function(model, training_args2, training_data)

# Report memory after training operation
#torch.cuda.synchronize()
#print("Memory after training: {} MB".format( \
#        torch.cuda.memory_allocated(args.cuda)/1024./1024.))


# Report memory after training iteration
#torch.cuda.synchronize()
#print("Peak memory usage: {} MB".format( \
        #torch.cuda.max_memory_allocated(args.cuda)/1024./1024.))
