from sparse_grad_matrix_sparse import Tucker_Decomposition
import torch


# Trainers
from transformers import TrainingArguments, Trainer
from utils import MetricsComputer

# Custom trainers
from trainers_custom import TrainerBert1, TrainerBert2, TrainerDoubleOpt

def pre_trainer_function(model, training_args1, tokenized_dataset):
    
    trainer = TrainerBert1(
        model=model,
        args=training_args1,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics = MetricsComputer('cola'),
    )
    trainer.make_grad_bank()
    trainer.train()
    UV_dict = {}

    grads1 = torch.stack(trainer.grads1[:360])
    grads2 = torch.stack(trainer.grads2[:360])
    del trainer
    u1, VT, U = Tucker_Decomposition(grads1)
    UV_dict.update({"output":tuple((U, VT))})
    u1, VT, U = Tucker_Decomposition(grads2)
    UV_dict.update({"interm":tuple((U, VT))})
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
            compute_metrics = MetricsComputer('cola'),
        )
        trainer.make_grad_bank()
    else:
        trainer = Trainer(
            model=model,
            args=training_args2,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            compute_metrics = MetricsComputer('cola'),
        )
        
    train_result = trainer.train()
    trainer.evaluate()
    trainer.log_metrics("train", train_result.metrics)
        