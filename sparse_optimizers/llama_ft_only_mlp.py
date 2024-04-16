import os, torch, logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, Trainer
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

DEVICE = "cuda:0"

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
    device_map=DEVICE,torch_dtype=torch.bfloat16
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

from transformers import TrainerCallback, TrainerCallback, TrainerState, TrainerControl

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

training_args = TrainingArguments(
        output_dir="/home/jovyan/team_code/big_models",
        save_total_limit=5,
        local_rank = 1,
        logging_first_step=True,
        eval_delay=0,
        seed=42,
        max_steps=1,
        data_seed=42,
        load_best_model_at_end=True,
        report_to=None,
        ddp_find_unused_parameters=None,
        # deepspeed=deepspeed_config,
        **trainer_config
    )
training_args.local_rank = -1


trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=training_data,
        skip_memory_metrics = False,
        callbacks = callbacks,
        eval_dataset=val_data,
        tokenizer=llama_tokenizer)

#metrics = trainer.evaluate()
#print (metrics)
trainer.train()

torch.cuda.synchronize()
print("Peak memory usage: {} MB".format( \
        torch.cuda.max_memory_allocated()/1024./1024.))


#fine_tuning.model.save_pretrained("/home/jovyan/team_code/big_models/llama-2-7b_8bit.pth")