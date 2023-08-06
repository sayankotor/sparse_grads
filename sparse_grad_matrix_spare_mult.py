import numpy as np
import pandas as pd
import transformers
from transformers import AutoModel, BertTokenizerFast, BertConfig, BertModel
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn

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
        return torch.zeros(max(mat.shape[0], inds.T[:, 0].shape[0]),dtype=dt).to('cuda').scatter_add(0, inds.T[:, 0], x_senders_m_vals)[:mat.shape[0]]
#     print('in big func')
    res = torch.vmap(sparsemat_vec)(mat.T).T
    print('vmap ended')
    return res.reshape(a,b,e)

class LinearFunctionSparseGrad(torch.autograd.Function):

        # Note that forward, setseup_context, and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, weight, bias, len_grads, treshold, U, VT):
        treshold = treshold
        if (len_grads < 30 ):    # space 1
            ctx.save_for_backward(input, weight, bias)
            return  input @ weight.T + bias
        else:
            #u1, U, VT = Tucker_Decomposition(torch.cat(MyLayer.grads))
#             print('input.shape', input.shape)
            input = input @ U.T ## Here change
#             print('input.shape', input.shape)
            
            #BATCH IS 4 HERREREEEE
            b, r, c = input.shape
            #ii = input.view(b, 1, -1)
            #inds = torch.argsort(-abs(ii))
            #ii_sorted = torch.gather(-abs(ii), 2, inds)
            #print(ii_sorted)
            thresholds = 0.001 # abs(ii_sorted[:, :, 101])#.to(input.device)
#             ii = abs(ii[0])[inds[0]]
#             print(ii[:50])
#             threshold = abs(ii[11]).item()
#             print('threshold ', thresholds)
            input = torch.where(abs(input)> thresholds.unsqueeze(1), input, torch.tensor(0.).to(input.device))
#             print('input shape ', input.shape)
#             print(torch.nonzero(input, as_tuple=True))
            rows, cols = torch.nonzero(input.reshape(b*r, c), as_tuple=True)
            rows, cols = torch.unique(rows), torch.unique(cols)
            print (len(rows), len(cols))
            input_reshaped = input.reshape(b*r, c)[rows, :]
#             print('SHAPES ', input.shape, weight.T.shape)
            res_indirect = input.reshape(b*r, c)[rows, :][:, cols] @ weight.T[cols, :]
            
#             print('reshaped input shape ',input.reshape(input.shape[0]*input.shape[1], input.shape[2]).shape)
#             print('res shape ', res_indirect.shape)
            res_ = res_indirect  @ VT.T
            I = torch.eye(b*r).to('cuda')
            res = I[:, rows] @ res_
            res = res.reshape(b, -1,VT.T.shape[1] )
            tup = torch.tensor([int(b*r) ,b,r,c])
#             res = torch.zeros((b*r, VT.T.shape[1])
#             res_ = torch.zeros((b*r, weight.T.shape[1]))
#             res_[rows, :] =  res_indirect
#             rd = input  @ weight.T 
#             print('res direct shape ', res.shape)
#             input = torch.where(abs(input)> threshold, input, 0.)
#             im= plt.imshow(input[0, :20, :20].cpu().detach().numpy(), cmap='jet', aspect='auto')#not in spy
#             print ("number of nonzero ", torch.count_nonzero(input[0, :,:].cpu().detach()))
#             plt.title('input '+'20x20 ' ) 
#             plt.colorbar(im) #not in spy
#             plt.show()
            ctx.save_for_backward(input_reshaped, rows, tup ,weight, bias, U, VT) # space 2
        ctx.size = input.shape[0]
        return    res + bias
        
        
        #print (weight.shape)
        #print (bias.shape)
        #print (len(grads)),
        #print (treshold)
        #print (U.shape, VT.shape)
#         print('weight.T@ U', weight.T.shape, U.shape, VT.shape)
        
#         return  input  @ weight.T  @ VT.T + bias # space 2  # HERE change
        
#         return sparsemat_mat(input, weight.T  @ VT.T) + bias
    


    @staticmethod
    def backward(ctx, grad_output):

        if len(ctx.saved_tensors) == 3:  # space 1
            input, weight, bias = ctx.saved_tensors
            grad_input = grad_weight = grad_bias = None
            if ctx.needs_input_grad[0]:
                grad_input = grad_output @ weight
            if ctx.needs_input_grad[1]:
                #print (grad_output.T.shape)
                #print (input.shape)
                grad_weight =  torch.einsum('ijk,kjl->il', grad_output.T, input)#grad_output.T @input
  
            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output
            
        elif len(ctx.saved_tensors) == 7: # space 2

            input_reshaped, rows,(num_rows, b , r , c) , weight, bias, U, VT = ctx.saved_tensors
            grad_input = grad_weight = grad_bias = None
            # print (grad_output.shape, VT.T.shape, U.shape)
            grad_output = grad_output # !!!! HERE change
            if ctx.needs_input_grad[0]:
                grad_input = grad_output @ weight
            if ctx.needs_input_grad[1]:
                I = torch.eye(b*r).to('cuda')
                input = I[:, rows] @ input_reshaped
                input = input.reshape(b,r,c)
                grad_weight = torch.einsum('ijk,kjl->il', grad_output.T, input)#grad_output.T @input
                #print('grad_weight', grad_weight.shape, U.shape, VT.shape)
                grad_weight = VT.T @ grad_weight  # !!!! HERE change
                grad_weight = torch.where(torch.abs(grad_weight) >= 0.0, grad_weight, torch.tensor(0.0).to('cuda'))  ## возвращаем градиент в каком пространстве?? VERY IMPORTANT
            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output
            
        return grad_input, grad_weight, grad_bias, None, None, None, None



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
        self.grads = []
        self.treshold = 1e-3
        self.U = None
        self.VT = None
        self.len_grads = len(self.grads)
        
        
    def create_UV(self):
        #print ("self.len_grads", self.len_grads)
        if (self.len_grads >= 30):
            self.grads = torch.stack(self.grads[:30])
            u1, VT, U = Tucker_Decomposition(self.grads)
            self.U = U.T
            self.VT = VT.T
            self.U.requires_grad = False
            self.VT.requires_grad = False
            #print ("self.U", self.U.shape, "self.VT", self.VT.shape)
            self.weight = torch.nn.Parameter(self.VT.T@self.weight@self.U.T)#torch.nn.Parameter(self.U@self.weight@self.VT)
        else:
            print ("please do 30 optimizer steps")
            
        
        
    def from_linear(self, linear: nn.Linear, transpose=False):
        if transpose:
            self.weight = torch.nn.Parameter(linear.weight.data.T)
        else:
            self.weight = torch.nn.Parameter(linear.weight.data)
        self.bias = torch.nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        
        
        
    def forward(self, x):
        
        if ((self.U is None) and (self.VT is None) and len(self.grads)>=30):
            print ("created matrix")
            self.create_UV()
            
        
        return LinearFunctionSparseGrad.apply(x, self.weight, self.bias, self.len_grads, self.treshold, self.U, self.VT)

            

def replace_bert_layers(model):
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

        new_layer.from_linear(layer.intermediate.dense)

        model.bert.encoder.layer[i].intermediate.dense = new_layer
          
        #print ("new shape", layer.intermediate.dense.weight.shape)
        
        token_dim, hidden_dim = layer.output.dense.weight.shape
        #print ("output")
        #print ("old shape", token_dim, hidden_dim)
        
        new_layer = SparseGradLinear(token_dim, hidden_dim)

        new_layer.from_linear(layer.output.dense)

        model.bert.encoder.layer[i].output.dense = new_layer
          
        #print ("new shape", layer.output.dense.weight.shape)
        #print ("\n\n")

    return model








