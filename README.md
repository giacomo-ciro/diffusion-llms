# More Efficient Text Diffusion via Length Prediction

Davide Beltrame, Giacomo Cirò, Luca Gandolfi, Vittorio Rossi

## Abstract

Diffusion language models (DLMs) offer a promising alternative to autoregressive models (ARMs) for text generation, but their fixed-length decoding process leads to significant computational inefficiencies. In this work, we address this limitation by predicting an upper bound to the output sequence length before generation begins and reduce the context window to be processed accordingly. Our approach relies solely on the internal representations of the model and explores both zero-shot and embedding-based techniques. When considering a state-of-the-art DLM, LLaDa-8B, a token-level classifier built on top of the encoded token embeddings successfully predicts an upper bound to the sequence length 80% of the times and better avoids underestimation compared to a DistilBERT-based baseline. Our results show that output length prediction is an effective and lightweight strategy to improve DLM efficiency, enabling significant computational savings with minimal overhead.

## Overview

Language generation with Large Language Models (LLMs) has traditionally been dominated by autoregressive models (ARMs), which generate text one token at a time. While effective, their sequential nature limits inference speed. Diffusion Language Models (DLMs) have emerged as a promising alternative, offering potential for faster generation through a denoising process.

DLMs operate by iteratively unmasking a sequence of tokens, where initial tokens represent the prompt and remaining ones are placeholder mask tokens, revealed progressively. At each denoising step, the model predicts logits for the full masked sequence and only unmasks tokens when confident. The context length must be fixed at the start, and DLMs handle variable length output by appending special end-of-sentence (EoS) tokens after the end of the sentence. This approach is effective, but computationally inefficient: the entire context window must be processed during each forward call of the denoising process, regardless of output length.

In this work, we focus on LLaDa, Large Language Diffusion with mAsking, an 8-B-parameter DLM, and propose methods to predict an upper bound for the generated sequence length at the initial stage of the denoising process, restricting the effective context to this predicted window and reducing unnecessary computations.

### Core Contributions

Our key contributions involve developing techniques to predict output length efficiently:

1. **Zero-shot Length Prediction**: We analyze the model's internal signals by examining the logits corresponding to the EoS token in a zero-shot fashion.

2. **Embedding-based Methods**: We develop:
   - A regression model using average prompt embeddings
   - A token-wise classification approach to identify EoS tokens
   
3. **Comparative Analysis**: We compare our methods against DistilBERT-based baselines, evaluating them on their ability to provide accurate upper bounds that avoid underestimating sequence lengths.

### Implementation Approaches

We explore the following methods for output length prediction:

1. **Logit Quantile Heuristic**: A zero-shot approach analyzing token-wise logit distributions for the EoS token
2. **Embeddings-based Regression**: A neural network trained on prompt embeddings to predict output length
3. **Token-wise Classification**: A classifier trained to identify which tokens are likely to be EoS tokens
4. **DistilBERT Methods**: Classification and regression models using DistilBERT's representations

### Evaluation Framework

Our evaluation framework focuses on the quality of upper bounds provided by each method:

1. **Bound Correctness**: Percentage of test samples for which a correct upper bound is estimated
2. **Bound Tightness**: Average number of tokens from true end of sequence to estimated end
3. **Saved Tokens**: Average number of tokens saved from estimated end to context window end
4. **Root MSE**: Square root of mean squared error between predicted and true sequence length

We prioritize methods that tend to overestimate rather than underestimate sequence lengths, as overestimation is safer for generation quality.

### Experimental Results

Our experiments with LLaDa-8B show:

1. **Token-level Classification**: The classifier based on LLaDa embeddings correctly predicts upper bounds for over 80% of test cases, with an average bound looseness of ~127 tokens.

2. **Zero-shot Quantile Heuristics**: Show a clear trade-off - higher quantiles yield higher correctness (Q75 achieves 99.88% valid bounds) but looser estimates (~510 tokens).

3. **Regression Methods**: Achieve the lowest RMSE but produce valid bounds for fewer samples (46-57%), as they're trained to minimize error rather than ensure upper bounds.

4. **Comparison with DistilBERT**: While DistilBERT exhibited lower average error in some cases, it frequently underestimated length, which is problematic for generation. LLaDa methods tended to safely overestimate.

5. **Efficiency Gains**: Our best methods save approximately 760-870 tokens on a 1024-token context window, representing substantial computational savings with minimal overhead.

### Limitations & Future Work

Our work has several limitations that suggest directions for future research:

1. **Computational Constraints**: Limited computational resources restricted our ability to explore more complex models or larger datasets.

2. **Model Requirements**: Our approach works best with DLMs that have embeddings pre-trained for variable-length output generation.

3. **Pretraining Challenges**: We began exploring ways to adapt existing diffusion language models to support variable-length generation (e.g., with DiffuGPT), but this remains an open challenge.

4. **Zero-shot Prediction Depth**: We did not test zero-shot EoS prediction at different stages of the denoising process, which might improve accuracy.

5. **Output Quality Analysis**: Our assumption that models maintain performance under varying upper bounds needs further verification.

6. **Dataset Limitations**: We used clean, multilingual, conversational prompts. A more diverse and length-balanced dataset could yield more generalizable insights.

7. **Multilingual Capabilities**: We only superficially explored the model's multilingual capabilities and its sensitivity to prompt phrasing.

### Conclusion

Our experiments addressed a significant limitation of DLMs by investigating how to upper bound the output sequence length. The core challenge is balancing the tightness of the predicted bound with the risk of underestimation, which can lead to premature truncation of generated text.

We demonstrated that upper bound prediction can be successfully approached as a classification or regression problem using a DLM's internal representations. A classifier relying on the DLM's internal representation strikes the best balance between bound tightness and accuracy, even compared to methods using sentence-level DistilBERT embeddings.

This work shows that predicting output sequence length is a viable strategy for enhancing the efficiency of DLMs like LLaDa, with potential for zero-shot and specialized solutions to address computational challenges in large-scale generative models.


## Usage

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