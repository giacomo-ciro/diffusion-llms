# Diffusion LLMs

Davide Beltrame, Giacomo Cirò, Luca Gandolfi, Vittorio Rossi

## Abstract

  Diffusion language models (DLMs) offer faster inference than autoregressive methods. We present a more efficient approach for variable-length text generation with LLaDa, an 8B-parameter DLM. Instead of generating fixed-size blocks until an EOS token, our method predicts an upper bound on output length from the input context, allowing truncation to the nearest power of two. This reduces unnecessary tokens and can reduce computational overhead compared to block-based strategies depending on the generation setup. We evaluate both length prediction accuracy and downstream performance under this new paradigm. *We propose a benchmark for evaluating the upper bound on the EoS classification.*

## Overview

The authors of the DiffuGPT paper adapted an Autoregressive Language Model (ARM), namely GPT-2, to obtain a Diffusion Language Model (DLM) while leveraging the pre-trained weights with almost no loss in performance. 

However, they did not address the issue of fixed output length which comes with discrete diffusion. In fact, DiffuGPT is only capable of generating output whose length is the full diffusion context size, which is not always the best choice and can also hinder performance in some cases. For example, in the case of a yes/no question, the model is forced to generate a full 512 tokens output, even if the answer is only 2 tokens long.

In this project, we aim to further improve DiffuGPT, making it capable of variable length generation.

First, we test an approach similar to the one proposed in the Llada paper: we continue pre-training with a custom dataset of [text + eos + pad], to give DiffuGPT the ability to generate a pad token, allowing it to limit the length of its ouput.

Then, we test how confident our model is in predicting the EoS token at the first step of the diffusion generation, which is fundamental to avoid computing all the subsequent pad tokens.

Finally, we propose a method to improve the model capacity of predicting the EoS token: fine-tuning on [text + mask + EoS + mask] examples to improve the model's accuracy in predicting the eos token at the first diffusion step by only looking at the provided context.

---

We implement and extend Diffusion Language Models (DLMs), focusing on improving variable-length text generation efficiency. DLMs offer significant advantages over traditional autoregressive language models, particularly in terms of inference speed, but face challenges with fixed-output length constraints.

### Core Concepts

**DiffuGPT Extension**: We build upon the DiffuGPT approach, which adapts GPT-2 to create a Diffusion Language Model while preserving performance. However, DiffuGPT generates fixed-length outputs regardless of content needs, wasting computational resources.

**Variable-Length Generation**: Our key innovation is implementing efficient variable-length generation for diffusion models through:

1. **EOS Token Prediction**: Developing models that can accurately predict the appropriate End-of-Sequence (EOS) token position early in the diffusion process
2. **Early Termination**: Implementing techniques to stop the diffusion process once meaningful content has been generated
3. **Length Prediction**: Incorporating length prediction models to estimate required output length based on input prompts

### Implementation Approaches

We explore multiple methods for improving EOS prediction:

1. **Fine-tuning with [text + EOS + PAD]**: Training models to recognize padding patterns and generate EOS tokens at appropriate positions
2. **EOS Classification**: Developing specialized classification heads to predict EOS placement
3. **Length Regression**: Using regression models to predict output sequence lengths directly from input contexts
4. **DistilBERT Adaptation**: Leveraging pre-trained language models to predict output lengths from input prompts

### Evaluation Framework

Our comprehensive evaluation system measures both:

1. **Accuracy**: How well models predict appropriate sequence lengths and EOS positions
2. **Performance**: Assessing generation quality on standard language model benchmarks

The evaluation framework in `evaluation/` includes:
- `eval_eos.py`: Focused on EOS token prediction accuracy
- [REMOVE] `eval_test.py`: Comprehensive testing across various language understanding benchmarks
- `eval.py`: General evaluation pipeline for diffusion models

This work represents a significant step toward more efficient and practical diffusion language models for real-world applications where output length varies significantly based on context.

## TO-DO's
- [ ] Measure performance (eos prediction accuracy, benchmarks) of pre-trained DiffuGPT before and after fine-tuning on dataset of (text + eos + pad)
- [ ] Curriculum learning on optimzer steps (currently is on samples, +1 every time a new sample is yielded)
- [dave] - Implement Classification/PredictionDataset in `datamodule.py` to generate training data that predicts 1 for eos and 0 for non-eos tokens.
- see datamodule, same for regression.
- [dave] - Implement RegressionDataset in `datamodule.py` to handle the length of the sequence.
- the expected output is a tensor of shape (batch_size, 1)
- dataloader iterates over the dataset and returns a batch of sequences
- get item handles the logic of how to get the data from the dataset and returns X, y, msk
- structure of the file class regression with methods: init with prompts and answer, len, get item (which returns x, y, msk); second class: memmapdatamodule with 

### Sync 30/04/25
- [x] Measure eos accuracy (does it actually improve?) - then, create same dataset with different mask rationale: different training to force model to predict eos token at the first step of the diffusion process (goal: predicting eos one-shot)
    - otherwise, we mask and unmask tokens until model predicts where the eos token is
    - observation: if the model is able to predict some mask tokens, it is likely to predict the eos token as well
- [x] Define a method to evaluate our specific task, e.g., how to measure the performance of the model in predicting the eos token at the first step of the diffusion process (accuracy / metrics / as function of numbe of unmasked tokens etc).
- ~~[x] Complete `check_config_validity` in `utils.py`~~

### Experiments

- [ ] Baseline Experiment
	- [ ]	Train and evaluate DiffuGPT baseline to establish reference performance.

- [ ] EOS and PAD Prediction Experiments
	- [ ]	EOS-only training (Accuracy-focused): loss computed on <eos> tokens
	- [ ]	EOS+PAD joint training (Random data) (Accuracy-focused): 

- [ ] Curriculum Learning Experiments
	- [ ]	EOS+PAD training with Curriculum Learning (Accuracy and generalization): Begin training on sequences of short length (e.g., 20 tokens), gradually increasing length (20 → 40 → 60, etc.). 

- [ ] Loss Function Ablation Studies
	- [ ]	Loss-weighting for EOS/PAD tokens (Improving token prediction accuracy): Experiment with weighted cross-entropy losses specifically targeting EOS/PAD token accuracy.
	- [ ] Auxiliary loss (contrastive/regularization) (Promoting robustness): Introduce auxiliary contrastive or regularization terms focusing on EOS/PAD prediction.

- [ ] Quick RL-based Experiment????
    - [ ]	Reinforcement Learning for EOS accuracy optimization (Rapid improvement): reward explicitly based on EOS token prediction accuracy. Policy-gradient or PPO fine-tuning.

- [ ] Inference Performance Checks
	- [ ] Benchmark inference speed (tokens/sec, latency per sequence).
	- [ ] Compare EOS/PAD trained models vs baseline(both arm and diffu-gpt)

- [ ] Generalization & Robustness Testing
	- [ ]	Length generalization tests: Evaluate models trained on shorter sequences against significantly longer test sequences.
	- [ ]	Domain generalization tests???

### Sync 14/05/25

5 methods for evaluation
1. logit (baseline)
2. concat regression (vitto)
3. average regression (vitto)
4. classification (luca)
5. distilbert (vitto)

after training them, we can evaluate them on the test set.

#### TODO
- [ ] Implement the 5 methods for evaluation
- [ ] remove unnecessary code from the repo
- [ ] add table to the report - how good are the models at predicting the eos token?
- [ ] generate with 25 - 50 - 75% of the tokens masked : 5 

### Feedback from Professor (private)
- ambition is good, doability is the question
- concretize the chance of success - a series of questions that can be answered quickly at the beginning
- check how we compare the different models, what kind of benchmarks and metrics we want to use (throughput: tokens per second with minimal perplexity loss)
- be very explicit about research question, don’t fear to be overly specific, also be open about the limitations
- change formulations to see if changing head affects anything: robustness checks
- find sources that do not affect variance
- walk the reader through the resulting paper

## Usage

We provide a comprehensive toolkit for training, evaluating, and using diffusion language models. Follow these instructions to get started.

### Installation

Install the required packages:

```bash
$ python -m pip install -r requirements.txt
```

### Configuration

The `config.json` file is the central place for configuring all aspects of the system:
- Model architecture (dimensions, layers, etc.)
- Training parameters (learning rate, batch size, etc.)
- Generation settings (temperature, strategies, etc.)
- Evaluation metrics and benchmarks

You can create a local copy for experimentation:

```bash
$ cp config.json local_config.json  # This file is gitignored
```

### Data Preparation

For training, prepare data from the FineWeb dataset in two formats:

#### 1. Standard Sequential Format
```bash
$ cd diffusion-llms/data
$ python prepare.py 100  # Creates memmap with 100 documents
```

#### 2. Variable-Length with Padding
```bash
$ cd diffusion-llms/data
$ python prepare_var_len.py path/to/config.json --train 100 --test 10
```
This creates separate train/test memmaps with each document padded to the same length.

### Training

The repository supports multiple training approaches:

#### Standard Training
```bash
$ python train.py path/to/config.json
```

#### LLaDa Approach Training
```bash
$ python train_llada.py path/to/config.json
```

#### Lightning-based Training
```bash
$ python train_llada_pl.py path/to/config.json
```

#### Specialized Training for EOS Prediction

Multiple notebook-based approaches are provided in the `baseline/` directory:
- `train_DistilBERT_DGPT_clas.ipynb`: Classification-based EOS prediction
- `train_DistilBERT_DGPT_reg.ipynb`: Regression-based length prediction
- `train_DistilBERT_LLaDa_clas.ipynb`: Classification for LLaDa models

### Sampling and Generation

Generate text using either diffusion or autoregressive methods:

```bash
$ python sample.py path/to/config.json
```

For DiffuGPT-specific sampling:
```bash
$ python baseline/sample_DiffuGPT.py path/to/config.json
```

### Evaluation

The repository offers multiple evaluation approaches:

#### Standard Benchmarks
```bash
$ cd evaluation
$ python eval.py lambda 100  # Evaluate on lambda benchmark with 100 examples
```

#### EOS Prediction Evaluation
```bash
$ python evaluation/eval_eos.py path/to/config.json
```

#### Comprehensive Benchmark Suite [REMOVE]
```bash
$ python evaluation/eval_test.py --config path/to/config.json --tasks bbh gsm8k humaneval --samples 10
```
This evaluates the model on multiple standard benchmarks:
- BBH (Big-Bench Hard): Reasoning tasks
- GSM8K: Grade-school math problems
- Minerva/MATH: Scientific reasoning
- HumanEval: Code generation
- MBPP: Practical Python programming

Results are saved in a detailed JSON format for analysis.

### Config.json
```json
{
  "pipeline": "arm",                        // (str) The training/sampling pipeline to use (e.g., arm, regular, etc.)
  "init_from":"",                           // (str) Model initialization checkpoint (empty for from scratch, "gpt2" for pretrained, or path to custom checkpoint)
  "memmap_path": "./data/train_7K.bin",     // (str) Path to generated memmap.bin
  "padded_dataset": false,                  // (bool) Whether the dataset has fixed-length chunks with padding
  "pad_masked_perc": 0.15,                  // (float) Percentage of tokens among the masked ones which are pad
  "pad_annealing_steps": 100,               // (int) Number of steps for linear schedule from 0 to pad_masked_perc
  "context_length": 1024,                   // (int) Maximum sequence length for model input (if pad datset, the two should coincide)

  "n_embd": 768,                            // (int) Embedding dimension size for model
  "n_layer": 12,                            // (int) Number of transformer layers
  "n_head": 12,                             // (int) Number of attention heads
  "resume_training": false,                 // (bool) Whether to resume from previous training state (must provide .ckpt in "init_from")
  
  "attn_annealing_steps": 1000,             // (int) Number of steps for attention mask annealing
  "mask_id": 10541,                         // (int) Token ID used for masking (must be < vocab size)
  "eos_token_id": 50256,                    // (int) End-of-sequence token ID
  "pad_token_id": 50257,                    // (int) Padding token ID (must be vocab size +1)

  "n_epochs": 1,                            // (int) Number of training epochs
  "n_steps": 100,                           // (int) Number of training steps per epoch (training samples / effective batch size)
  "val_check_interval": 100,                // (int) Frequency of validation checks (steps)
  "val_test_perc": 0.05,                    // (float) Percentage of data to use for validation

  "batch_size": 8,                          // (int) Number of sequences per batch
  "accumulate_grad": 16,                    // (int) Number of batches for gradient accumulation
  "grad_clip": 1.0,                         // (float) Maximum gradient norm for gradient clipping

  "betas": [0.9,0.95],                      // (list) Adam optimizer beta parameters
  "weight_decay": 0.01,                     // (float) L2 regularization strength
  "max_lr": 6e-4,                           // (float) Maximum learning rate
  "warmup_pct": 0.1,                        // (float) Percentage of training steps (n_steps) for learning rate warmup
  "div_factor": 25.0,                       // (float) Factor to divide max_lr by to get initial learning rate
  "final_div_factor": 1e4,                  // (float) Factor to divide max_lr by to get final learning rate

  "enable_checkpointing": true,             // (bool) Whether to save model checkpoints
  "save_dir": "./checkpoints/",             // (str) Directory to save model checkpoints
  "wandb": true,                            // (bool) Whether to use Weights & Biases for experiment tracking
  "run_name": "gpt2-scratch-openwebtext",   // (str) Name of the experiment run for logging
  "project_name": "diffusion-llms",         // (str) Project name for experiment organization in logging

  "user_prompt": "Once upon a time",        // (str) Initial text prompt for generation
  "n_samples": 1,                           // (int) Number of text samples to generate
  "temperature": 1.0,                       // (float) Sampling temperature (higher = more random)
  "max_new_tokens": 10,                     // (int) Maximum number of new tokens to generate
  "top_k": null,                            // (int or null) Limit sampling to top k most likely tokens
  "diffusion_steps": 4,                     // (int) Number of diffusion steps in the generation process
  "repetition_penalty": 1.2,                // (float) Penalty for repeating tokens (higher = less repetition)
  "do_sample": false,                       // (bool) Whether to use sampling (true) or greedy decoding (false)
  "denoising_strategy": "random"            // (str) Strategy for denoising (e.g., random, deterministic, etc.)
}
```

## Misc
1. Use own branch during development, together we handle merges.
2. Duplicate `config.json`, rename to `local_config.json` (added to `.gitignore`) and use it to test locally.
3. Login (locally or on hpc) to wandb from the CLI by running the following:
```bash 
$ python
>>> import wandb
>>> wandb.login()
```
Re-installing some packages fixed an issue with FineWeb streaming:
```bash
$ python -m pip install --upgrade datasets huggingface-hub fsspec
```

## Repository Structure
```bash
├── diffusion_llms/ 
│   ├── baseline/           # EOS prediction baseline implementations
│       ├── datamodule.py   # Data loading utilities
│       ├── model_baseline.py  # DistilBERT-based models for length prediction
│       ├── sample_DiffuGPT.py # Generation script for DiffuGPT models
│       ├── train_DistilBERT_DGPT_clas.ipynb  # Classification-based EOS prediction
│       ├── train_DistilBERT_DGPT_reg.ipynb   # Regression-based length prediction 
│       ├── train_DistilBERT_LLaDa_clas.ipynb # LLaDa classification training
│       └── train_DistilBERT_LLaDa_reg.ipynb  # LLaDa regression training
│   ├── configs/            # Configuration files for different model variants
│       ├── diffugpt-base.json     
│       ├── diffugpt-eoseos.json   
│       ├── diffugpt-eoshead.json  
│       ├── diffugpt-eospad.json   
│       ├── eval_config.json      
│       └── finetune_config.json   
│   ├── data/               # Data preparation and processing
│       ├── prepare.py    
│       ├── prepare_var_len.py 
│       ├── prepare_llada.py  
│       ├── train.csv      
│       └── test.csv        
│   ├── dataloader/         
│       ├── llada_datamodule.py  
│       ├── llada_from_file.py   
│       └── llada_2.py     
│   ├── evaluation/         
│       ├── eval.py         # General evaluation framework
│       ├── eval_eos.py     
│       ├── eval_eospad.py  
│       ├── eval_test.py    # Comprehensive benchmark testing
│       └── utils.py    
│   ├── models/           
│       ├── gpt2_diffusion.py     # DiffuGPT implementation
│       ├── gpt2_arm.py           # Autoregressive GPT-2 implementation
│       ├── llada.py              # LLaDa model implementation
│       ├── llada_binary_head.py  # LLaDa with binary classification head
│       ├── diffugpt2_length_head.py  # DiffuGPT with length prediction
│       └── diffugpt2_reg_head.py  # DiffuGPT with regression head
│   ├── tokenizers/       
│       ├── custom_gpt_w_pad.py  # GPT-2 tokenizer with padding support
│       └── __init__.py
│   ├── __init__.py       
│   ├── attention_patch.py  # Patch for attention mechanism 
│   ├── config.json         # Default configuration file
│   ├── datamodule.py       # Data loading utilities
│   ├── model.py            # PyTorch Lightning wrapper for GPT-2
│   ├── sample.py           # Text generation script
│   ├── train.py           
│   ├── train_llada.py    
│   ├── train_llada_pl.py   
│   └── utils.py            
├── notebooks/             
│   ├── eos_logit_exploration.ipynb  
│   ├── eos_insights.py   
│   └── eos_curve.py      
├── outputs/                
│   ├── random_eos.png     
│   ├── trigger_eos.png    
│   └── multilingual_eos/   
├── LICENSE
├── README.md
└── requirements.txt 
```

## References

- **(LLaDa)** Nie et al. (2025): [_Large Language Diffusion Models_](https://arxiv.org/pdf/2502.09992)
- **(Block Diffusion)** Arriola et al. (ICLR 2025): [_Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models_](https://arxiv.org/pdf/2503.09573)
- **(DiffuGPT)** Gong et al. (ICLR 2025): [_Scaling Diffusion Language Models via Adaptation from Autoregressive Models_](https://arxiv.org/pdf/2410.17891)
- **(DiffusionBERT)** Gong et al. (2022): [_DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models_](https://arxiv.org/pdf/2211.15029)
- **(Scaling Laws)** Liang et al. (2024): [_Scaling Laws for Diffusion Transformers_](https://arxiv.org/pdf/2410.08184)
- **(Survey)** Zou et al. (2023): [_A Survey of Diffusion Models in Natural Language Processing_](https://arxiv.org/pdf/2305.14671)