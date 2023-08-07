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
        x_senders_m_vals = vals *x_senders
        dt = x_senders_m_vals.dtype
        return torch.zeros(max(mat.shape[0], inds.T[:, 0].shape[0]),dtype=dt).to('cuda').scatter_add(0, inds.T[:, 0], x_senders_m_vals)[:mat.shape[0]]
#     print('in big func')
    res = torch.vmap(sparsemat_vec)(mat.T).T
    print('vmap ended')
    return res.reshape(a,b,e)

class LinearFunctionSparseGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, len_grads, treshold, U, VT):
        treshold = treshold
        if (len_grads < 30 ):    # space 1
            ctx.save_for_backward(input, weight, bias)
            print (input.shape, weight.T.shape)                 
            return  input @ weight.T + bias
        else:
            input = input @ U.T ## Here change
            #print ("input.shape", input.shape)
            b, r, c = input.shape
            ii = input.view(b, 1, -1)
            inds = torch.argsort(-abs(ii))
            ii_sorted = torch.gather(-abs(ii), 2, inds)
           
            thresholds = abs(ii_sorted[:, :, 101])#.to(input.device)
            input = torch.where(abs(input)> thresholds.unsqueeze(1), input, torch.tensor(0.).to(input.device))

            rows, cols = torch.nonzero(input.reshape(b*r, c), as_tuple=True)
            rows, cols = torch.unique(rows), torch.unique(cols)
            
            input_reshaped = input.reshape(b*r, c)[rows, :]

            res_indirect = input.reshape(b*r, c)[rows, :][:, cols] @ weight.T[cols, :]
            
            res_ = res_indirect  @ VT.T
            I = get_I_matrix(b, r)
            res = I[:, rows] @ res_
            res = res.reshape(b, -1,VT.T.shape[1] )
            tup = torch.tensor([int(b*r) ,b,r,c])

            ctx.save_for_backward(input_reshaped, rows, tup ,weight, bias, U, VT) # space 2
        ctx.size = input.shape[0]
        return res + bias

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
            grad_output = grad_output @ VT # !!!! HERE change
            if ctx.needs_input_grad[0]:
                grad_input = grad_output @ weight @ U
            if ctx.needs_input_grad[1]:
                I = get_I_matrix(b, r)
                input = I[:, rows] @ input_reshaped
                input = input.reshape(b,r,c)
                grad_weight = torch.einsum('ijk,kjl->il', grad_output.T, input)
                grad_weight = VT.T @ grad_weight  # !!!! HERE change
                grad_weight = torch.where(torch.abs(grad_weight) >= 0.0, grad_weight, torch.tensor(0.0).to('cuda'))  ## возвращаем градиент в каком пространстве?? VERY IMPORTANT
            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output
            
        return grad_input, grad_weight, grad_bias, None, None, None, None
            

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








