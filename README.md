# SparseGrad: A Selective Method for Efficient Fine-tuning of MLP Layers
Sparse version of weight gradients in MLP layers.

Using Tucker decomposition, we found a space where the weights of the linear layers is highly sparse (about 1% of all parameters remain significant). By rewriting Torch Autograd weight in this space, we reduced the number of trainablr parameters, and consequently, the memory usage. With the same memory consumption, our model outperforms LoRA and baselines that select the most significant parameters in the linear layer and train only those.

The modifications of TorchAutograd and Forward and Backward  operation are in sparse_grads/sparse_optimizers/sparse_grad_matrix_sparse.py

The modifications of Hugginface Trainer class for model with semi-sparse gradients are in sparse_grads/sparse_optimizers/trainer_custom.py

# Experiments with BERT and RoBERTa

1. To apply Sparse Grad method to fine-tune BERT model on Cola dataset:
   
**cd bert/**

**CUDA_VISIBLE_DEVICES=0 python3 test_bert.py --cuda 0 --sparse_grad**

2. To do the same with RoBERTa:
   
**cd roberta/**

**CUDA_VISIBLE_DEVICES=0 python3 test_roberta.py --cuda 0 --sparse_grad**

3. To employ the full fine-tuning benchmark
switch branch from main to lora_benchmark

**cd sparse_grad**

**run.py** 

--task task in GLUE 

--run_type ft (regular fine-tune), lora (Low Rank Adaptation), sparse (Sparse Grad), meprop (MeProp) 

--model_path Path to model on HF or local storage 

--optimize It now involves a search for optimal parameters. 

--n_params The number of parameters in the linear layer remained trainable.

# Experiments with LLaMa 2 7B

LLaMa 2 7B fine-tuned on openassistant dataset are in

LLaMa 2 7B fine-tuned on openassistant dataset using LoRA are in

LLaMa 2 7B fine-tuned on openassistant dataset using Sparse Grad are in

The example of usage:




