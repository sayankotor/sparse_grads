import argparse
import torch
import transformers
import torch

from datasets import load_dataset
import pandas as pd

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


# Create argument parser
parser = argparse.ArgumentParser(prog="Test suite for GPT models")
parser.add_argument("--sparse_grad", action="store_true")
parser.add_argument("--cuda", type=int)

# Parse arguments
args = parser.parse_args()

# Select device
DEVICE = torch.device("cuda:{}".format(args.cuda))

# Load the model and set it into training mode
base_model_name = "TheBloke/Llama-2-7B-fp16"
refined_model = "7b_opst"
# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, max_length=512, truncation=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "left",  # Fix for fp16
# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)
# Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map=DEVICE, torch_dtype=torch.bfloat16
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

llama_tokenizer.add_special_tokens({
    "eos_token": llama_tokenizer.convert_ids_to_tokens(base_model.config.eos_token_id),
    "bos_token": llama_tokenizer.convert_ids_to_tokens(base_model.config.bos_token_id),
    "unk_token": llama_tokenizer.convert_ids_to_tokens(
        base_model.config.pad_token_id if base_model.config.pad_token_id != -1 else tokenizer.pad_token_id
    ),
})

import datasets
dataset = datasets.load_dataset("OpenAssistant/oasst1", split="validation")
    
dataset = dataset.map(lambda sample: {
        "message_id": sample["message_id"],
        "parent_id": sample["parent_id"],
        "text": sample["text"],
        },
        batched=True,
        remove_columns=list(dataset.features),)


from custom_dataset import get_custom_dataset
training_data = get_custom_dataset(llama_tokenizer, split="train")
val_data = get_custom_dataset(llama_tokenizer, split="validation")

for param in base_model.parameters():
    param.requires_grad = False

for i in range(len(base_model.model.layers)):
    if (i%3 == 0):
        for param in base_model.model.layers[i].mlp.parameters():
            param.requires_grad = True


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for pname, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            print(pname, param.numel())
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

print_trainable_parameters(base_model)

import pickle



        
## training arguments


trainer_config =  {
    "evaluation_strategy": "steps",
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 32,
    "eval_steps": 75,
    # "eval_steps": 3,
    "save_steps": 300,
    # "save_steps": 3,
    "logging_steps": 5,
    "learning_rate": 0.00009,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 50,
    # "warmup_steps": 5,
    "num_train_epochs": 1,
    # "max_steps": 15,
    "fp16": False,
    "bf16": True,
    "torch_compile": False,
    "optim": "adamw_torch"
}

training_args1 = TrainingArguments(
        output_dir="/home/jovyan/team_code/big_models",
        save_total_limit=5,
        max_steps = 10,
        local_rank = 1,
        logging_first_step=True,
        eval_delay=0,
        seed=42,
        data_seed=42,
        load_best_model_at_end=True,
        report_to=None,
        ddp_find_unused_parameters=None,
        # deepspeed=deepspeed_config,
        **trainer_config
    )
training_args1.local_rank = -1

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        
        peft_model_path = "/home/jovyan/team_code/big_models2/"
        
        kwargs["model"].save_pretrained(peft_model_path)
        return control

callbacks = [SavePeftModelCallback]


trainer = TrainerLlama1(
        model=base_model,
        args=training_args1,
        train_dataset=training_data,
        eval_dataset=val_data,
        callbacks =callbacks
    )


trainer.make_grad_bank()
trainer.train()
UV_dict = {}

grads1 = torch.stack(trainer.grads1[:120]).float()
grads2 = torch.stack(trainer.grads2[:120]).float()
del trainer

torch.cuda.synchronize()
print("Memory after del trainer: {} MB".format( \
        torch.cuda.max_memory_allocated(args.cuda)/1024./1024.))

torch.save(grads1, 'grads1.pth')
torch.save(grads2, 'grads2.pth')

u1, VT, U = Tucker_Decomposition(grads1.to(DEVICE))
UV_dict.update({"up":tuple((U, VT))})
u1, VT, U = Tucker_Decomposition(grads2.to(DEVICE))
UV_dict.update({"down":tuple((U, VT))})

with open('llama.pickle', 'wb') as handle:
        pickle.dump(UV_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     model = sparse_grad_linear(model, UV_dict)
del grads1, grads2


#base_model.to(device)
# Report memory before training iteration
torch.cuda.synchronize()
print("Memory for the model and input dataset: {} MB".format( \
        torch.cuda.max_memory_allocated(args.cuda)/1024./1024.))


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
torch.cuda.synchronize()
print("Peak memory usage: {} MB".format( \
        torch.cuda.max_memory_allocated(args.cuda)/1024./1024.))
