from sparse_grad_matrix_sparse import Tucker_Decomposition
import torch


# Trainers
from transformers import TrainingArguments, Trainer
from util import compute_metrics

# Custom trainers
from trainers_custom import TrainerBert2, TrainerDoubleOpt, TrainerLlama1

trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=training_data,
        callbacks = callbacks,
        eval_dataset=val_data,
        tokenizer=llama_tokenizer)

def pre_trainer_function(model, training_args1, training_data):
    
    trainer = TrainerLlama1(
        model=model,
        args=training_args1,
        train_dataset=training_data,
        max_steps=30,
        #eval_dataset=tokenized_dataset["validation"],
        compute_metrics = compute_metrics,
    )
    trainer.make_grad_bank()
    trainer.train()
    UV_dict = {}

    grads1 = torch.stack(trainer.grads1[:450])
    grads2 = torch.stack(trainer.grads2[:450])
    del trainer
    u1, VT, U = Tucker_Decomposition(grads1)
    UV_dict.update({"up":tuple((U, VT))})
    u1, VT, U = Tucker_Decomposition(grads2)
    UV_dict.update({"down":tuple((U, VT))})
#     model = sparse_grad_linear(model, UV_dict)
    del grads1, grads2
    return UV_dict


def trainer_function(model, training_args2, tokenized_dataset, is_sparse_grad = False):
    #print (model.device, tokenized_dataset["train"].device)
    if (is_sparse_grad):
        trainer = TrainerDoubleOpt(
            model=model,
            args=training_args2,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            compute_metrics = compute_metrics,
        )
        trainer.make_grad_bank()
    else:
        trainer = Trainer(
            model=model,
            args=training_args2,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            compute_metrics = compute_metrics,
        )
        
    train_result = trainer.train()
    trainer.evaluate()
    trainer.log_metrics("train", train_result.metrics)
        