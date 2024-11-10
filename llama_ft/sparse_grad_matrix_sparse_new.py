import numpy as np
import pandas as pd
import transformers
from transformers import AutoModel, BertTokenizerFast, BertConfig, BertModel
import matplotlib.pyplot as plt


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
    #u1, _, _ = torch.svd(torch.reshape(tensor, (n1, -1)))
    print ("1", flush=True)
    u2, _, _ = torch.svd(torch.reshape(torch.permute(tensor, [1, 2, 0]), (n2, -1)))
    print ("2", flush=True)
    u3, _, _ = torch.svd(torch.reshape(torch.permute(tensor, [2, 0, 1]), (n3, -1)))
    print ("3", flush=True)
    return _, u2, u3

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

def sparsemat_mat(smat, mat):
    a, b, c = smat.shape
    d, e = mat.shape
    def sparsemat_vec(  vec):
    # a, b, c = smat.shape
        mat = smat.reshape(a*b, c)
        print('sparse operations start')
        inds = mat.to_sparse().indices()
        vals = mat.to_sparse().values()
        print('sparse operations end')
        try:
           
            x_senders = vec[0, inds.T[:, 1]]
            print('senders creation 1')
        except IndexError:
            vec = vec.unsqueeze(0)
            x_senders = vec[0, inds.T[:, 1]]
            print('senders creation 2')
        print('x senders ', vals.shape, x_senders.shape)
        print('vals.shape',  vals.shape)
        x_senders_m_vals = vals *x_senders
        dt = x_senders_m_vals.dtype
        print('before 1st return')

        print 
        
        return torch.zeros(max(mat.shape[0], inds.T[:, 0].shape[0]),dtype=dt).to('cuda').scatter_add(0, inds.T[:, 0], x_senders_m_vals)[:mat.shape[0]]
#     print('in big func')
    res = torch.vmap(sparsemat_vec)(mat.T).T
    print('vmap ended')
    return res.reshape(a,b,e)

# +



class LinearFunctionSparseGrad(torch.autograd.Function):

        # Note that forward, setseup_context, and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, weight, bias, treshold, layer_type):
        if (layer_type == torch.tensor(0)):
            U, VT = SparseGradLinearUp._U, SparseGradLinearUp._VT
        else:
            U, VT = SparseGradLinearDown._U, SparseGradLinearDown._VT
        treshold = treshold
        input = input @ U.T 
        
        ctx.save_for_backward(input,weight, bias, layer_type) # space 2
        ctx.size = input.shape[0]
        
        #return  input @ weight.T + bias # space 2  # HERE change
        #print ("U, VT", U, VT)
        #print ("input @ weight.T @ VT.T", input @ weight.T @ VT.T)
        #print ("bias", bias)
        if bias is not None:
            return  input @ weight.T @ VT.T + bias # space 2  # HERE change
        else:
            return  input @ weight.T @ VT.T
            


    @staticmethod
    def backward(ctx, grad_output):

        input, weight, bias, layer_type = ctx.saved_tensors
        
        if (layer_type == torch.tensor(0)):
            U, VT = SparseGradLinearUp._U, SparseGradLinearUp._VT
        else:
            U, VT = SparseGradLinearDown._U, SparseGradLinearDown._VT
        
        grad_input = grad_weight = grad_bias = None
        
        if (layer_type == torch.tensor(0)):
            grad_output = grad_output @ SparseGradLinearUp._VT# !!!! HERE change
        else:
            grad_output = grad_output @SparseGradLinearDown._VT
            
        if ctx.needs_input_grad[0]:
            if (layer_type == torch.tensor(0)):
                grad_input = grad_output @ weight @ SparseGradLinearUp._U
            else:
                grad_input = grad_output @ weight @ SparseGradLinearDown._U
        if ctx.needs_input_grad[1]:
            grad_weight = torch.einsum('ijk,kjl->il', grad_output.T, input)#grad_output.T @input
            #grad_weight = VT.T @ grad_weight  # !!!! HERE change
            trhld = torch.topk(torch.flatten(grad_weight), 100000).values[99999]
            grad_weight = torch.where(torch.abs(grad_weight) >= trhld, grad_weight, torch.tensor(0.0).to('cuda'))  ## возвращаем градиент в каком пространстве?? VERY IMPORTANT
            #if (grad_weight.is_coalesced()):
                #print (grad_weight.indices().shape)
                #print ("\n number of nonzero")
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output
            
        return grad_input, grad_weight, grad_bias, None, None
            
        #return grad_input, grad_weight, grad_bias, None, None, None


# -

class SparseGradLinearUp(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    _U = {}
    _V = {}

    @property
    def U(self):
        return SparseGradLinearUp._U
    
    def VT(self):
        return SparseGradLinearUp._VT
    

    @staticmethod
    def set_UV(tuple_UV, device):
        SparseGradLinearUp._U = tuple_UV[0].to(torch.bfloat16).to(device)
        SparseGradLinearUp._VT = tuple_UV[1].to(torch.bfloat16).to(device)

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
        #self.U = None
        #self.VT = None
        self.is_sparse = False
        self.acts = None
                
        
    def from_linear(self, linear: nn.Linear, tuple_UV = tuple(), transpose=False):
        if transpose:
            self.weight = torch.nn.Parameter(linear.weight.data.T)
        else:
            self.weight = torch.nn.Parameter(linear.weight.data)
         
        self.bias = torch.nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        #self.U = tuple_UV[0]
        #self.VT = tuple_UV[1]
        SparseGradLinearUp.set_UV(tuple_UV, self.weight.device)
        self.weight = torch.nn.Parameter(SparseGradLinearUp._VT.T@self.weight@SparseGradLinearUp._U.T)   
        
    def rewert_to_linear(self):
        self.weight = torch.nn.Parameter(SparseGradLinearUp._VT@self.weight@SparseGradLinearUp._U)
        self.bias = torch.nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        self.is_sparse = False
        
        
        
    def forward(self, x):
        # self.acts = x@self.U.T
        return LinearFunctionSparseGrad.apply(x, self.weight, self.bias, self.treshold, torch.tensor(0))


class SparseGradLinearDown(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
        
    _U = {}
    _V = {}

    @property
    def U(self):
        return SparseGradLinearDown._U
    
    def VT(self):
        return SparseGradLinearDown._VT
    

    @staticmethod
    def set_UV(tuple_UV, device):
        SparseGradLinearDown._U = tuple_UV[0].to(torch.bfloat16).to(device)
        SparseGradLinearDown._VT = tuple_UV[1].to(torch.bfloat16).to(device)

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
        #self.U = None
        #self.VT = None
        self.is_sparse = False
        self.acts = None
                
        
    def from_linear(self, linear: nn.Linear, tuple_UV = tuple(), transpose=False):
        if transpose:
            self.weight = torch.nn.Parameter(linear.weight.data.T)
        else:
            self.weight = torch.nn.Parameter(linear.weight.data)
         
        self.bias = torch.nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        #self.U = tuple_UV[0]
        #self.VT = tuple_UV[1]
        SparseGradLinearDown.set_UV(tuple_UV, self.weight.device)
        self.weight = torch.nn.Parameter(SparseGradLinearDown._VT.T@self.weight@SparseGradLinearDown._U.T)   
        
    def rewert_to_linear(self):
        self.weight = torch.nn.Parameter(SparseGradLinearDown._VT@self.weight@SparseGradLinearDown._U)
        self.bias = torch.nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        self.is_sparse = False
        
        
        
    def forward(self, x):
        # self.acts = x@self.U.T
        return LinearFunctionSparseGrad.apply(x, self.weight, self.bias, self.treshold, torch.tensor(1))


def replace_bert_layers(model, UV_dict):
    device = model.device
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
        
        new_layer = SparseGradLinearIntermediate(token_dim, hidden_dim)

        new_layer.from_linear(layer.intermediate.dense, UV_dict['interm'])

        model.bert.encoder.layer[i].intermediate.dense = new_layer
          
        #print ("new shape", layer.intermediate.dense.weight.shape)
        
        token_dim, hidden_dim = layer.output.dense.weight.shape
        #print ("output")
        #print ("old shape", token_dim, hidden_dim)
        
        new_layer = SparseGradLinearOutput(token_dim, hidden_dim)

        new_layer.from_linear(layer.output.dense, UV_dict['output'])

        model.bert.encoder.layer[i].output.dense = new_layer
          
        #print ("new shape", layer.output.dense.weight.shape)
        #print ("\n\n")
        model.to(device)
    return model


def replace_llama_layers(model, UV_dict):
    device = model.device
    
    for i, layer in enumerate(model.model.layers):
        if (i%3 == 0):
            token_dim, hidden_dim = model.model.layers[i].mlp.up_proj.weight.shape
            #print ("dense")
            print ("old shape up", token_dim, hidden_dim)
            
            new_layer = SparseGradLinearUp(token_dim, hidden_dim)
    
            new_layer.from_linear(model.model.layers[i].mlp.up_proj, UV_dict['up'])
    
            model.model.layers[i].mlp.up_proj = new_layer

            print ("new shape up", model.model.layers[i].mlp.up_proj.weight.shape)
              
            #print ("new shape", layer.intermediate.dense.weight.shape)
            
            token_dim, hidden_dim = model.model.layers[i].mlp.down_proj.weight.shape
            #print ("output")
            print ("old shape down", token_dim, hidden_dim)
            
            new_layer = SparseGradLinearDown(token_dim, hidden_dim)
    
            new_layer.from_linear(model.model.layers[i].mlp.down_proj, UV_dict['down'])
    
            model.model.layers[i].mlp.down_proj = new_layer

            print ("new shape down", model.model.layers[i].mlp.down_proj.weight.shape)
              
            #print ("new shape", layer.output.dense.weight.shape)
            #print ("\n\n")
    model.to(device)
    return model








