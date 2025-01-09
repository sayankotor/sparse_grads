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


3. To employ the full fine-tuning benchmark:
   
switch branch from main to lora_benchmark

**cd sparse_grad**

**run.py --task 'stsb' --run_type sparse --model_path 'roberta-base' --optimize False --n_params 280000** 

--task task in GLUE 

--run_type ft (regular fine-tune), lora (Low Rank Adaptation), sparse (Sparse Grad), meprop (MeProp) 

--model_path Path to model on HF or local storage 

--optimize It now involves a search for optimal parameters. 

--n_params The number of parameters in the linear layer remained trainable.


## News and Updates

* ```2024.12.12``` 👏👏👏 SparseGrad has been presented by EMNLP 2024 Main!


## <a name="Citing"></a>Citation
Consider giving this repository a star and cite SparseGrad in your publications if it helps your research.

```
@inproceedings{chekalina-etal-2024-sparsegrad,
    title = "{S}parse{G}rad: A Selective Method for Efficient Fine-tuning of {MLP} Layers",
    author = "Chekalina, Viktoriia A.  and
      Rudenko, Anna  and
      Mezentsev, Gleb  and
      Mikhalev, Aleksandr  and
      Panchenko, Alexander  and
      Oseledets, Ivan",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.831/",
    doi = "10.18653/v1/2024.emnlp-main.831",
    pages = "14929--14939",
    abstract = "The performance of Transformer models has been enhanced by increasing the number of parameters and the length of the processed text. Consequently, fine-tuning the entire model becomes a memory-intensive process. High-performance methods for parameter-efficient fine-tuning (PEFT) typically work with Attention blocks and often overlook MLP blocks, which contain about half of the model parameters. We propose a new selective PEFT method, namely SparseGrad, that performs well on MLP blocks. We transfer layer gradients to a space where only about 1{\%} of the layer`s elements remain significant. By converting gradients into a sparse structure, we reduce the number of updated parameters. We apply SparseGrad to fine-tune BERT and RoBERTa for the NLU task and LLaMa-2 for the Question-Answering task. In these experiments, with identical memory requirements, our method outperforms LoRA and MeProp, robust popular state-of-the-art PEFT approaches."
}





