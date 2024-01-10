from functools import partial
from math import sqrt

import torch
from torch.nn import Parameter

from utils import map_module


class LoRALinear(torch.nn.Module):
    r"""A reimplementation of LoRA linear for parameter efficient training.

    .. note::
       In the original paper author initialize the first factor with zeros and
       the second one from Gaussian distribution. We use uniform Kaiming
       initialization for the second factor instead.

    .. note::

       Factorized representation of additive correction has order unlike the
       one reported in the original paper. Factors are stored as list of
       :math:`A`, :math:`B^\top` to make application to input more convenient.
    """

    def __init__(self, in_features: int, out_features: int, rank: int,
                 bias: bool = True, device=None, dtype=None,
                 scale: float = 1.0):
        super().__init__()
        self.rank = rank
        self.scale = scale / rank

        opts = {'dtype': dtype, 'device': device}

        # Create frozen linear layer.
        self.linear = torch.nn.Linear(in_features, out_features, bias, **opts)
        self.linear.requires_grad_(False)

        # Create trainable factorized coorection.
        self.factors = torch.nn.ParameterList([
            Parameter(torch.empty((in_features, rank), **opts)),
            Parameter(torch.empty((out_features, rank), **opts)),
        ])

        # Initialize only correction.
        self.reset_parameters(False)

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    def reset_parameters(self, recursively=True) -> None:
        if recursively:
            self.linear.reset_parameters()
        torch.nn.init.kaiming_uniform_(self.factors[0], a=sqrt(5))
        torch.nn.init.zeros_(self.factors[1])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert not self.linear.weight.requires_grad, \
            'Weights of kernel must be freezed.'
        output = self.linear(input)
        correction = (input @ self.factors[0]) @ self.factors[1].T
        return output + self.scale * correction

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear, rank: int, **kwargs):
        """Creates linear layer with additive lowr-rank term from linear layer.
        """
        kwargs['bias'] = kwargs.get('bias', linear.bias is not None)
        self = cls(linear.in_features, linear.out_features, rank, **kwargs)
        self.linear = linear
        self.linear.weight.requires_grad = False
        return self

    def to_linear(self) -> torch.nn.Linear:
        """Merge an additive term to a basic weight matrix."""
        raise NotImplementedError

def convert_low_rank(rank: int, module: torch.nn.Module, path: str):
    if not isinstance(module, torch.nn.Linear):
        raise ValueError('Only linear layer can be converted: '
                         f'type={type(module)}.')
    return LoRALinear.from_linear(module, rank)

def convert_model(model, path: str, rank: int):
    return map_module(
        model, partial(convert_low_rank, rank), path)
