import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter

from functools import partial
from utils import map_module, get_weight_grads


class LinearFunctionMeProp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias,  layer_type, n_params):
        ctx.n_params = n_params
        
        ctx.save_for_backward(input,weight, bias, layer_type) 
        ctx.size = input.shape[0]
        
        return  input @ weight.T + bias


    @staticmethod
    def backward(ctx, grad_output):

        input, weight, bias, layer_type = ctx.saved_tensors        
        grad_input = grad_weight = grad_bias = None
         
        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ weight

        if ctx.needs_input_grad[1]:
            grad_weight = torch.einsum('ijk,kjl->il', grad_output.T, input)
            trhld = torch.topk(torch.abs(torch.flatten(grad_weight)), ctx.n_params).values[ctx.n_params - 1]
            grad_weight = torch.where(torch.abs(grad_weight) >= trhld, grad_weight, torch.tensor(0.0).to('cuda'))#.to_sparse()
            # if (layer_type == torch.tensor(0)):
            #     torch.save(grad_weight.to_sparse(), '0.pth')
            # else:
            #     torch.save(grad_weight.to_sparse(), '1.pth')
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output
            
        return grad_input, grad_weight, grad_bias, None, None, None


class MePropLinearIntermediate(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, n_params: int, bias: bool = True,
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
        self.n_params = n_params
        
    def from_linear(self, linear: nn.Linear, tuple_UV = tuple(), transpose=False):
        if transpose:
            self.weight = torch.nn.Parameter(linear.weight.data.T)
        else:
            self.weight = torch.nn.Parameter(linear.weight.data)
         
        self.bias = torch.nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
            
    def forward(self, x):
        return LinearFunctionMeProp.apply(x, self.weight, self.bias, torch.tensor(0), self.n_params)
    
    
class MePropLinearOutput(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, n_params: int, bias: bool = True,
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

        self.n_params = n_params
                
        
    def from_linear(self, linear: nn.Linear,  transpose=False):
        if transpose:
            self.weight = torch.nn.Parameter(linear.weight.data.T)
        else:
            self.weight = torch.nn.Parameter(linear.weight.data)
         
        self.bias = torch.nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        
        
    def forward(self, x):
        return LinearFunctionMeProp.apply(x, self.weight, self.bias, torch.tensor(1), self.n_params)


def convert_me_prop(target_cls, n_params: int, module: torch.nn.Module, path: str):
    if not isinstance(module, torch.nn.Linear):
        raise ValueError('Only linear layer can be converted: '
                         f'type={type(module)}.')
    token_dim, hidden_dim = module.weight.shape
    new_module = target_cls(token_dim, hidden_dim, n_params=n_params)
    new_module.from_linear(module)
    return new_module

def convert_model(model, path: str, target_cls, n_params):
    print('Converting these layers to MeProp:')
    return map_module(
        model, partial(convert_me_prop, target_cls, n_params), path)
