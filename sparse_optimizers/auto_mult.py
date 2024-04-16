import numpy as np
import pandas as pd
import transformers
from transformers import AutoModel, BertTokenizerFast, BertConfig, BertModel
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from functools import lru_cache

import torch.autograd
import torch.nn.functional as F
import torch
from torch import Tensor
from torch.nn.parameter import Parameter,   UninitializedParameter
from torch.nn import functional as F



@lru_cache()
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



def sparse_to_dense(input, weight):
    b, r, c = input.shape
    rows, cols = torch.nonzero(input.reshape(b*r, c), as_tuple=True)
    rows, cols = torch.unique(rows), torch.unique(cols)
    res_indirect = input.reshape(b*r, c)[rows, :][:, cols] @ weight[cols, :]

    # res_indirect = res_indirect.reshape(b,r, weight.T.shape[1])

    I = torch.eye(b*r).to('cuda')
    res = I[:, rows] @ res_indirect
    res = res.reshape(b, -1,weight.shape[1] )
  
    return res


def sparse_to_dense_prep(input, weight):
    b, r, c = input.shape
    rows, cols = torch.nonzero(input.reshape(b*r, c), as_tuple=True)
    rows, cols = torch.unique(rows), torch.unique(cols)
    res_indirect = input.reshape(b*r, c)[rows, :][:, cols] @ weight[cols, :]
    return res_indirect, rows, cols, b, r, c

def sparse_precount_to_dense_T(input, weight, rows):
    #print (input.shape)
    #print (weight.shape)
    res_indirect = input @ weight.reshape(-1, weight.shape[2])[rows, :]
    return res_indirect


def sparse_precount_to_dense(input, weight, rows, cols, b, r, c):
    res_indirect = input @ weight
    I = torch.eye(b*r,).to(weight.dtype).to('cuda')
    res = I[:, rows] @ res_indirect
    res = res.reshape(b, -1, weight.shape[1] )
    return res
    

class LinearFunctionSparseMult(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, weight, bias, treshold):
        ctx.save_for_backward(input,weight, bias)
        ctx.size = input.shape[0]
        
        if bias is not None:
            return  input @ weight.T  + bias 
        else:
            return  input @ weight.T 


    @staticmethod
    def backward(ctx, grad_output):

        input, weight, bias, layer_type = ctx.saved_tensors
        
        b, r, c = grad_output.shape
        
        grad_input = grad_weight = grad_bias = None

        rows, cols = torch.nonzero(grad_output.reshape(-1, grad_output.shape), as_tuple=True)
        rows, cols = torch.unique(rows), torch.unique(cols)
        
        if ctx.needs_input_grad[0]:
           grad_input = sparse_precount_to_dense(grad_output, weight, rows, cols, b, r, c)
        
        if ctx.needs_input_grad[1]:
            grad_weight = sparse_precount_to_dense_T(grad_output.T, input, rows)
                
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output
            
        return grad_input, grad_weight, grad_bias, None
            
        #return grad_input, grad_weight, grad_bias, None, None, None


# -

class LinearSpMult(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    _U = {}
    _V = {}
    _out_grads = []
    _acts = []
    _weight = []
    _grad_weight = []
    

    @property
    def U(self):
        return SparseGradLinearIntermediate._U
    
    def VT(self):
        return SparseGradLinearIntermediate._VT
    

    @staticmethod
    def set_UV(tuple_UV):
        SparseGradLinearIntermediate._U = tuple_UV[0]
        SparseGradLinearIntermediate._VT = tuple_UV[1]

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

        
    def from_linear(self, linear: nn.Linear, transpose=False):
        if transpose:
            self.weight = torch.nn.Parameter(linear.weight.data.T)
        else:
            self.weight = torch.nn.Parameter(linear.weight.data)
         
        self.bias = torch.nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        
        
    def forward(self, x):
        return LinearFunctionSparseMult.apply(x, self.weight, self.bias, self.treshold)
    
    


def replace_llama_layers(model, UV_dict):
    device = model.device
    
    for i, layer in enumerate(model.model.layers):
        if (i%3 == 0):
            token_dim, hidden_dim = model.model.layers[i].mlp.up_proj.weight.shape
            #print ("dense")
            print ("old shape up", token_dim, hidden_dim)
            
            new_layer = LinearSpMult(token_dim, hidden_dim)
    
            new_layer.from_linear(model.model.layers[i].mlp.up_proj)
    
            model.model.layers[i].mlp.up_proj = new_layer

            print ("new shape up", model.model.layers[i].mlp.up_proj.weight.shape)
              
            #print ("new shape", layer.intermediate.dense.weight.shape)
            
            token_dim, hidden_dim = model.model.layers[i].mlp.down_proj.weight.shape
            #print ("output")
            print ("old shape down", token_dim, hidden_dim)
            
            new_layer = LinearSpMult(token_dim, hidden_dim)
    
            new_layer.from_linear(model.model.layers[i].mlp.down_proj)
    
            model.model.layers[i].mlp.down_proj = new_layer

            print ("new shape down", model.model.layers[i].mlp.down_proj.weight.shape)
              
            #print ("new shape", layer.output.dense.weight.shape)
            #print ("\n\n")
    model.to(device)
    return model







