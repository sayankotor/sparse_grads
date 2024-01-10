# sparse_grads
Sparse version of weight gradients in MLP layers.

The modifications of TorchAutograd and Forward and Backward  operation are in sparse_grads/sparse_optimizers/sparse_grad_matrix_sparse.py
The modifications of Hugginface Trainer class for model with semi-sparse gradients are in sparse_grads/sparse_optimizers/trainer_custom.py

1. To apply Sparse Grad method to fine-tune BERT model on Cola dataset:
**cd bert/**
**CUDA_VISIBLE_DEVICES=0 python3 test_bert.py --cuda 0 --sparse_grad**

2. To do the same with RoBERTa:
**cd roberta/**
**CUDA_VISIBLE_DEVICES=0 python3 test_roberta.py --cuda 0 --sparse_grad**
