from sparse_grad_matrix_sparse import Tucker_Decomposition
import torch


# Trainers
from transformers import TrainingArguments, Trainer
from util import compute_metrics

# Custom trainers
from trainers_custom import TrainerBert1, TrainerBert2, TrainerDoubleOpt

def freeze_bert(model):
    for param in model.parameters():
        param.requires_grad = False
    
    for ind in range(len(model.bert.encoder.layer)):
        for param in model.bert.encoder.layer[ind].output.dense.parameters():
            param.requires_grad = True
            
        for param in model.bert.encoder.layer[ind].intermediate.dense.parameters():
            param.requires_grad = True
            
def unfreeze_bert(model):
    for param in model.parameters():
        param.requires_grad = True

def pre_trainer_function(model, training_args1, tokenized_dataset):

    #freeze_bert(model)
    torch.cuda.synchronize()
    trainer = TrainerBert1(
        model=model,
        args=training_args1,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics = compute_metrics,
    )
    trainer.make_grad_bank()
    trainer.train()
    UV_dict = {}
    unfreeze_bert(model)
    grads1 = torch.stack(trainer.grads1[:480])
    grads2 = torch.stack(trainer.grads2[:480])
    del trainer
    torch.cuda.empty_cache()
    u1, VT, U = Tucker_Decomposition(grads1)
    UV_dict.update({"output":tuple((U, VT))})
    u1, VT, U = Tucker_Decomposition(grads2)
    UV_dict.update({"interm":tuple((U, VT))})
    #model = sparse_grad_linear(model, UV_dict)
    del grads1, grads2
    torch.cuda.empty_cache()
    return UV_dict


def trainer_function(model, training_args2, tokenized_dataset, is_sparse_grad = False):
    #print (model.device, tokenized_dataset["train"].device)

    #freeze_bert(model)
    if (is_sparse_grad):
        trainer = TrainerDoubleOpt(#TrainerDoubleOpt(
            model=model,
            args=training_args2,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            compute_metrics = compute_metrics,
        )
        trainer.make_grad_bank(show_out_grads = False, show_acts = False)
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
        