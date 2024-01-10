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
    tensor.req
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

def sparsemat_mat(smat, mat):
    a, b, c = smat.shape
    d, e = mat.shape
    def sparsemat_vec(vec):
    # a, b, c = smat.shape
        mat = smat.reshape(a*b, c)
        #print('sparse operations start')
        inds = mat.to_sparse().indices()
        vals = mat.to_sparse().values()
        #print('sparse operations end')
        try:
           
            x_senders = vec[0, inds.T[:, 1]]
            #print('senders creation 1')
        except IndexError:
            vec = vec.unsqueeze(0)
            x_senders = vec[0, inds.T[:, 1]]
            #print('senders creation 2')
        #print('x senders ', vals.shape, x_senders.shape)
        #print('vals.shape',  vals.shape)
        x_senders_m_vals = vals *x_senders
        dt = x_senders_m_vals.dtype
        #print('before 1st return')
        return torch.zeros(max(mat.shape[0], inds.T[:, 0].shape[0]),dtype=dt).to('cuda').scatter_add(0, inds.T[:, 0], x_senders_m_vals)[:mat.shape[0]]
#     print('in big func')
    res = torch.vmap(sparsemat_vec)(mat.T).T
    #print('vmap ended')
    return res.reshape(a,b,e)

# +

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

    


class LinearFunctionSparseGrad(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, weight, bias, treshold, layer_type):
        if (layer_type == torch.tensor(0)):
            U, VT = SparseGradLinearIntermediate._U, SparseGradLinearIntermediate._VT
        else:
            U, VT = SparseGradLinearOutput._U, SparseGradLinearOutput._VT
        treshold = treshold
        input = input @ U.T 

        if (layer_type == torch.tensor(0)):
            SparseGradLinearIntermediate._acts = input # ONLY FOR SHOW MODE
        else:
            SparseGradLinearOutput._acts = input # ONLY FOR SHOW MODE
            
        
        ctx.save_for_backward(input,weight, bias, layer_type) # space 2
        ctx.size = input.shape[0]
        
        #return  input @ weight.T + bias # space 2  # HERE change
        return  input @ weight.T @ VT.T + bias # space 2  # HERE change


    @staticmethod
    def backward(ctx, grad_output):

        input, weight, bias, layer_type = ctx.saved_tensors
        
        if (layer_type == torch.tensor(0)):
            U, VT = SparseGradLinearIntermediate._U, SparseGradLinearIntermediate._VT
        else:
            U, VT = SparseGradLinearOutput._U, SparseGradLinearOutput._VT
        
        grad_input = grad_weight = grad_bias = None
        
        if (layer_type == torch.tensor(0)):
            grad_output = grad_output @ SparseGradLinearIntermediate._VT# !!!! HERE change
            #SparseGradLinearIntermediate._out_grads =  grad_output # ONLY FOR SHOW MODE
            #SparseGradLinearIntermediate._weight = weight @ SparseGradLinearIntermediate._U
        else:
            grad_output = grad_output @SparseGradLinearOutput._VT
            #SparseGradLinearOutput._out_grads =  grad_output # ONLY FOR SHOW MODE
            #SparseGradLinearOutput._weight = weight @ SparseGradLinearOutput._U
            
        if ctx.needs_input_grad[0]:
           if (layer_type == torch.tensor(0)):
                #grad_input = grad_output @ weight @ SparseGradLinearIntermediate._U
                weight = weight @ SparseGradLinearIntermediate._U
               
           else:
                #grad_input = grad_output @ weight @ SparseGradLinearOutput._U
                weight = weight @ SparseGradLinearOutput._U
           #print ("grad_output.shape", grad_output.shape)    
           grad_input = sparse_to_dense(grad_output, weight)
           #print ("grad_input.shape", grad_input.shape)
        
        #print ("grad_input", torch.count_nonzero(grad_input), grad_input.shape)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.einsum('ijk,kjl->il', grad_output.T, input)#grad_output.T @input
            #grad_weight = VT.T @ grad_weight  # !!!! HERE change
            trhld = torch.topk(torch.flatten(grad_weight), 28000).values[27999]
            grad_weight = torch.where(torch.abs(grad_weight) >= trhld, grad_weight, torch.tensor(0.0).to('cuda')).to_sparse()  ## возвращаем градиент в каком пространстве?? VERY IMPORTANT
            #if (grad_weight.is_coalesced()):
                #print (grad_weight.indices().shape)
                #print ("\n number of nonzero")

            if (layer_type == torch.tensor(0)):
                SparseGradLinearIntermediate._grad_weight = grad_weight
            else:
                SparseGradLinearOutput._grad_weight = grad_weight
                
                
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output
            
        return grad_input, grad_weight, grad_bias, None, None
            
        #return grad_input, grad_weight, grad_bias, None, None, None


# -

class SparseGradLinearIntermediate(torch.nn.Module):
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
        SparseGradLinearIntermediate.set_UV(tuple_UV)
        self.weight = torch.nn.Parameter(SparseGradLinearIntermediate._VT.T@self.weight@SparseGradLinearIntermediate._U.T)   
        
    def rewert_to_linear(self):
        self.weight = torch.nn.Parameter(SparseGradLinearIntermediate._VT@self.weight@SparseGradLinearIntermediate._U)
        self.bias = torch.nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        self.is_sparse = False
        
        
        
    def forward(self, x):
        return LinearFunctionSparseGrad.apply(x, self.weight, self.bias, self.treshold, torch.tensor(0))
    
    
class SparseGradLinearOutput(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
        
    _U = {}
    _V = {}
    _out_grads = []
    _acts = []
    _weight = []
    _grad_weight = []

    @property
    def U(self):
        return SparseGradLinearOutput._U
    
    def VT(self):
        return SparseGradLinearOutput._VT
    

    @staticmethod
    def set_UV(tuple_UV):
        SparseGradLinearOutput._U = tuple_UV[0]
        SparseGradLinearOutput._VT = tuple_UV[1]

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
       
        SparseGradLinearOutput.set_UV(tuple_UV)
        self.weight = torch.nn.Parameter(SparseGradLinearOutput._VT.T@self.weight@SparseGradLinearOutput._U.T)   
        
    def rewert_to_linear(self):
        self.weight = torch.nn.Parameter(SparseGradLinearOutput._VT@self.weight@SparseGradLinearOutput._U)
        self.bias = torch.nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        self.is_sparse = False
        
        
        
    def forward(self, x):
        return LinearFunctionSparseGrad.apply(x, self.weight, self.bias, self.treshold, torch.tensor(1))

            

def replace_bert_layers(model, UV_dict):
    device = model.device
    if hasattr(model, "roberta") and hasattr(model.roberta, "encoder"):
        encoder = model.roberta.encoder
    elif hasattr(model, "encoder"):
        encoder = model.encoder
    else:
        raise ValueError("Expected model to have attribute 'encoder' or 'roberta.encoder'.")

    for i, layer in enumerate(model.roberta.encoder.layer):
        token_dim, hidden_dim = layer.intermediate.dense.weight.shape
        #print ("dense")
        #print ("old shape", token_dim, hidden_dim)
        
        new_layer = SparseGradLinearIntermediate(token_dim, hidden_dim)

        new_layer.from_linear(layer.intermediate.dense, UV_dict['interm'])

        model.roberta.encoder.layer[i].intermediate.dense = new_layer
          
        #print ("new shape", layer.intermediate.dense.weight.shape)
        
        token_dim, hidden_dim = layer.output.dense.weight.shape
        #print ("output")
        #print ("old shape", token_dim, hidden_dim)
        
        new_layer = SparseGradLinearOutput(token_dim, hidden_dim)

        new_layer.from_linear(layer.output.dense, UV_dict['output'])

        model.roberta.encoder.layer[i].output.dense = new_layer
          
        #print ("new shape", layer.output.dense.weight.shape)
        #print ("\n\n")
        model.to(device)
    return model


def replace_roberta_layers(model, UV_dict):
    device = model.device
    if hasattr(model, "roberta") and hasattr(model.roberta, "encoder"):
        encoder = model.roberta.encoder
    elif hasattr(model, "encoder"):
        encoder = model.encoder
    else:
        raise ValueError("Expected model to have attribute 'encoder' or 'roberta.encoder'.")

    for i, layer in enumerate(model.roberta.encoder.layer):
        token_dim, hidden_dim = layer.intermediate.dense.weight.shape
        #print ("dense")
        #print ("old shape", token_dim, hidden_dim)
        
        new_layer = SparseGradLinearIntermediate(token_dim, hidden_dim)

        new_layer.from_linear(layer.intermediate.dense, UV_dict['interm'])

        model.roberta.encoder.layer[i].intermediate.dense = new_layer
          
        #print ("new shape", layer.intermediate.dense.weight.shape)
        
        token_dim, hidden_dim = layer.output.dense.weight.shape
        #print ("output")
        #print ("old shape", token_dim, hidden_dim)
        
        new_layer = SparseGradLinearOutput(token_dim, hidden_dim)

        new_layer.from_linear(layer.output.dense, UV_dict['output'])

        model.roberta.encoder.layer[i].output.dense = new_layer
          
        #print ("new shape", layer.output.dense.weight.shape)
        #print ("\n\n")
        model.to(device)
    return model








