import numpy as np
import pandas as pd
import transformers
from transformers import AutoModel, BertTokenizerFast, BertConfig, BertModel
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn

from functools import lru_cache

import matplotlib.pyplot as plt
import torch.autograd
import torch.nn.functional as F
import torch
from torch import Tensor
from torch.nn.parameter import Parameter,   UninitializedParameter
from torch.nn import functional as F

def func_collecting_tensors(step, tensor1, tensor2=None):
    '''собирает в тензор     '''
    if step == 0:
        return tensor1.unsqueeze(0)
    else:
        return torch.concatenate((tensor1, tensor2),0)



def Tucker_Decomposition(tensor):
    n1, n2, n3 = tensor.shape
    u1, _, _ = torch.svd(torch.reshape(tensor, (n1, -1)))
    u2, _, _ = torch.svd(torch.reshape(torch.permute(tensor, [1, 2, 0]), (n2, -1)))
    u3, _, _ = torch.svd(torch.reshape(torch.permute(tensor, [2, 0, 1]), (n3, -1)))
    return u1, u2, u3

def get_tucker_tensors(dict_layers):
    '''делает словарь где ключом будет слой, а значением будет тензор'''        
    dict_tensor = dict(zip(range(12), [[]]*12))
    for key in dict_layers.keys():
        dict_tensor[key].append(torch.cat(dict_layers[key], 2 ))
    return dict_tensor

def func_collecting_tensors(step, tensor1, tensor2=None):
    '''собирает в тензор     '''
    if step == 0:
        return tensor1.unsqueeze(0)
    else:
        return torch.concatenate((tensor1, tensor2),0)

@lru_cache
def get_I_matrix(b, r):
    return  torch.eye(b*r).to('cuda')

def Tucker_Decomposition(tensor):
    n1, n2, n3 = tensor.shape
    u1, _, _ = torch.svd(torch.reshape(tensor, (n1, -1)))
    u2, _, _ = torch.svd(torch.reshape(torch.permute(tensor, [1, 2, 0]), (n2, -1)))
    u3, _, _ = torch.svd(torch.reshape(torch.permute(tensor, [2, 0, 1]), (n3, -1)))
    return u1, u2, u3

def get_tucker_tensors(dict_layers):
    '''делает словарь где ключом будет слой, а значением будет тензор'''        
    dict_tensor = dict(zip(range(12), [[]]*12))
    for key in dict_layers.keys():
        dict_tensor[key].append(torch.cat(dict_layers[key], 2 ))
    return dict_tensor

class LinearFunctionSparseGrad(torch.autograd.Function):

        # Note that forward, setseup_context, and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, weight, bias, U, VT):
        input = input @ U.T ## Here change
        ctx.save_for_backward(input, weight, bias, U, VT) # space 2
        ctx.size = input.shape[0]
        return  input @ weight.T @ VT.T + bias # space 2  # HERE change
    


    @staticmethod
    def backward(ctx, grad_output):

        input, weight, bias, U, VT = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        # print (grad_output.shape, VT.T.shape, U.shape)
        grad_output = grad_output @ VT# !!!! HERE change
        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ weight @ U
        if ctx.needs_input_grad[1]:
            grad_weight = torch.einsum('ijk,kjl->il', grad_output.T, input)#grad_output.T @input
            #grad_weight = VT.T @ grad_weight  # !!!! HERE change
            grad_weight = torch.where(torch.abs(grad_weight) >= 0.001, grad_weight, torch.tensor(0.0).to('cuda')).to_sparse()  ## возвращаем градиент в каком пространстве?? VERY IMPORTANT
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output
            
        return grad_input, grad_weight, grad_bias, None, None, None, None, None

class SparseGradLinear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.treshold = 1e-3
        self.U = None
        self.VT = None
        self.is_sparse = False
                
        
    def from_linear(self, linear: nn.Linear, tuple_UV = tuple(), transpose=False):
        if transpose:
            self.weight = torch.empty_like(linear.weight.data.T).copy_(linear.weight.data.T)#torch.nn.Parameter(linear.weight.data.T)
        else:
            self.weight = torch.empty_like(linear.weight.data).copy_(linear.weight.data)#torch.nn.Parameter(linear.weight.data)
        self.bias = torch.nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        self.U = tuple_UV[0]
        self.VT = tuple_UV[1]
        
    def rewert_to_linear(self):
        self.weight = torch.nn.Parameter(self.VT@self.weight@self.U)
        self.bias = torch.nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        self.is_sparse = False
        
        
        
    def forward(self, x):
        
        return LinearFunctionSparseGrad.apply(x, self.weight, self.bias, self.U, self.VT)

            

def replace_bert_layers(model, UV_dict):
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        encoder = model.bert.encoder
    elif hasattr(model, "encoder"):
        encoder = model.encoder
    else:
        raise ValueError("Expected model to have attribute 'encoder' or 'bert.encoder'.")

    for i, layer in enumerate(model.bert.encoder.layer):
        token_dim, hidden_dim = layer.intermediate.dense.weight.shape
        #print ("dense")
        #print ("old shape", token_dim, hidden_dim)
        
        new_layer = SparseGradLinear(token_dim, hidden_dim)

        new_layer.from_linear(layer.intermediate.dense, UV_dict['interm'])

        model.bert.encoder.layer[i].intermediate.dense = new_layer
          
        #print ("new shape", layer.intermediate.dense.weight.shape)
        
        token_dim, hidden_dim = layer.output.dense.weight.shape
        #print ("output")
        #print ("old shape", token_dim, hidden_dim)
        
        new_layer = SparseGradLinear(token_dim, hidden_dim)

        new_layer.from_linear(layer.output.dense, UV_dict['output'])

        model.bert.encoder.layer[i].output.dense = new_layer
          
        #print ("new shape", layer.output.dense.weight.shape)
        #print ("\n\n")

    return model








