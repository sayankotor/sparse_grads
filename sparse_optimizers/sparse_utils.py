from functools import partial
from math import sqrt

import torch
from torch.nn import Parameter

from transformers import Trainer, TrainingArguments

from utils import map_module, get_weight_grads
from sparse_grad_matrix_sparse import SparseGradLinearIntermediate, Tucker_Decomposition
from trainers_custom import TrainerBert1


def convert_sparse(UV_dict, target_cls, n_params: int, module: torch.nn.Module, path: str):
    if not isinstance(module, torch.nn.Linear):
        raise ValueError('Only linear layer can be converted: '
                         f'type={type(module)}.')
    token_dim, hidden_dim = module.weight.shape
    new_module = target_cls(token_dim, hidden_dim, n_params=n_params)
    new_module.from_linear(module, UV_dict)
    return new_module

def convert_model(model, path: str, UV_dict, target_cls, n_params):
    print('Converting these layers to sparse:')
    return map_module(
        model, partial(convert_sparse, UV_dict, target_cls, n_params), path)


class SparsePretrainer(Trainer):
    def __init__(self, output_layers_path, intermediate_layers_path, n_stack_grads, *args, **kwargs,):
        super().__init__(*args, **kwargs)
        self.n_stack_grads = n_stack_grads
        self.output_layers_path = output_layers_path
        self.intermediate_layers_path = intermediate_layers_path

    def make_grad_bank(self):
        # self.n_show = 3
        self.losses = []
        self.n_steps = 0
        self.grads1 = []
        self.grads2 = []
    
    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        loss.backward()
        self.n_steps += 1
        for grad in get_weight_grads(model, self.output_layers_path):
            if (len(self.grads1)) < self.n_stack_grads:
                self.grads1.append(torch.empty_like(grad).copy_(grad))
            else:
                break

        for grad in get_weight_grads(model, self.intermediate_layers_path):
            if (len(self.grads2)) < self.n_stack_grads:
                self.grads2.append(torch.empty_like(grad).copy_(grad))
            else:
                break
            
        self.losses.append(loss.cpu().detach().numpy())
        return loss.detach()


def get_UV_dict(model, task, output_layers_path, intermediate_layers_path, n_stack_grads, tokenized_dataset, lr, batch_size, seed, max_steps=11):
    training_args = TrainingArguments(
            learning_rate=lr,
            num_train_epochs=1,
            evaluation_strategy="no",
            max_steps = max_steps,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            save_strategy='no',
            # The next line is important to ensure the dataset labels are properly passed to the model
            remove_unused_columns=True,
            output_dir='./model',
            overwrite_output_dir=True,
            seed=seed,
            report_to='none',
            lr_scheduler_type='constant', ## NOT AS IT WAS BEFORE
            warmup_ratio=0.,
            )

    trainer = SparsePretrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation_matched" if task == "mnli" else "validation"],
        output_layers_path=output_layers_path,
        intermediate_layers_path=intermediate_layers_path,
        n_stack_grads=n_stack_grads
    )

    trainer.make_grad_bank()
    trainer.train()
    UV_dict = {}

    grads1 = torch.stack(trainer.grads1[:n_stack_grads])
    u1, VT, U = Tucker_Decomposition(grads1)
    UV_dict.update({"output":tuple((U, VT))})
    grads2 = torch.stack(trainer.grads2[:n_stack_grads])
    u1, VT, U = Tucker_Decomposition(grads2)
    UV_dict.update({"interm":tuple((U, VT))})
    return UV_dict
