# Diffusion LLMs

The authors of the DiffuGPT paper adapted an Autoregressive Language Model (ARM), namely GPT-2, to obtain a Diffusion Language Model (DLM) while leveraging the pre-trained weights with almost no loss in performance. 

However, they did not address the issue of fixed output length which comes with discrete diffusion. In fact, DiffuGPT is only capable of generating output whose length is the full diffusion context size, which is not always the best choice and can also hinder performance in some cases. For example, in the case of a yes/no question, the model is forced to generate a full 512 tokens output, even if the answer is only 2 tokens long.

In this project, we aim to further improve DiffuGPT, making it capable of variable length generation.

First, we test an approach similar to the one proposed in the Llada paper: we continue pre-training with a custom dataset of [text + eos + pad], to give DiffuGPT the ability to generate a pad token, allowing it to limit the length of its ouput.

Then, we test how confident our model is in predicting the EoS token at the first step of the diffusion generation, which is fundamental to avoid computing all the subsequent pad tokens.

Finally, we propose a method to improve the model capacity of predicting the EoS token: fine-tuning on [text + mask + EoS + mask] examples to improve the model's accuracy in predicting the eos token at the first diffusion step by only looking at the provided context.

## TO-DO's
- [ ] Measure performance (eos prediction accuracy, benchmarks) of pre-trained DiffuGPT before and after fine-tuning on dataset of (text + eos + pad)
- [ ] Curriculum learning on optimzer steps (currently is on samples, +1 every time a new sample is yielded)

### Sync 30/04/25
- [ ] Measure eos accuracy (does it actually improve?) - then, create same dataset with different mask rationale: different training to force model to predict eos token at the first step of the diffusion process (goal: predicting eos one-shot)
    - otherwise, we mask and unmask tokens until model predicts where the eos token is
    - observation: if the model is able to predict some mask tokens, it is likely to predict the eos token as well
- [dave - in progress] Define a method to evaluate our specific task, e.g., how to measure the performance of the model in predicting the eos token at the first step of the diffusion process (accuracy / metrics / as function of numbe of unmasked tokens etc).
- [ ] Complete `check_config_validity` in `utils.py`

### Feedback from Professor
- ambition is good, doability is the question
- concretize the chance of success - a series of questions that can be answered quickly at the beginning
- check how we compare the different models, what kind of benchmarks and metrics we want to use (throughput: tokens per second with minimal perplexity loss)
- be very explicit about research question, don’t fear to be overly specific, also be open about the limitations
- change formulations to see if changing head affects anything: robustness checks
- find sources that do not affect variance
- walk the reader through the resulting paper

## Usage

Install the necessary packages using `pip`:

```bash
$ python -m pip install -r requirements.txt
```

The `config.json` is the unique place where to specify hyper-parameters for all the tasks to perform. It handles the model instantiation logic, training and sampling procedure.

### Training
To train, you first have to generate the training data from FineWeb dataset and save it to a `np.memmap` object:
```bash
$ cd diffusion-llms/data
```
To create a unique memmap array of tokens, corresponding to the specified number of documents in the dataset, separated by the eos tokens:
```bash
$ python prepare.py 100
```
Instead, to create a unique memmap array of tokens, corresponding to documents in the dataset, where each document is of the same length (pad token id is appended). In the `config.json` file you can specify the length of each document `context_length` and the `pad_token_id` to use for padding shorter documents. The attributes are useed to specify how man documents to use for training and testing. Two different memmap arrays are created with the train and test documents.
```bash
$ python prepare_var_len.py path/to/config.json --train 100 -test 10
```
Once the data is ready, we specify the path in the `config.json` together with the other hyper-params used for training and start training:
```bash
$ cd ..
$ python train.py path/to/config.json
```
Training can be conducted starting from:
- Auto-regressive Model (scratch or GPT-2 checkpoint)
- Diffusion Model (scratch or diffuGPT checkpoint)
And using the pipeline:
- Auto-regressive (predict next tokne, causal attention mask)
- Diffusion (predict masked tokens, full-attention mask)
When training with diffusion, you can specify the attention annealing schedule. When using a padded dataset, you can introduce a pad annealing schedule (to gradually mask pad tokens).

### Sampling
We can sample using diffusion or autoregressive strategy from any model. We specify a combination of `pipeline` (diffusion or arm) and `init_from` keys in the `config.json`. Then we add the `user_prompt` and the generation arguments (top k, temperature, denoising strategy etc.). Then run:
``` bash 
$ cd diffusion-llms
$ python sample.py path/to/config.json
```
Notice that we can also generate using discrete diffusion from a arm gpt2, and the results will likely be bad.

### Evaluation
We can evaluate a diffusion model on standard benchmarks:
```bash
$ cd evaluation
$ python eval.py lambda 100
```
Where we specify the benchmark to evaluate on and the number of documents in the benchmark to test.

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
├── diffusion-llms/ 
│   ├── checkpoints/        # Checkpoint files for model weights
│   ├── data/openwebtext_local/
│       ├── prepare.py      # Script for tokenizing and preparing dataset
│       ├── prepare_var_len.py 
│       ├── train_1M.txt
│       └── etc.
│   ├── evaluation/    
│       ├── eval.py
│       ├── test_performances.py
│       └── etc.            # Duplicate scripts to be removed (after check)
│   ├── attention_patch.py
│   ├── config.json         # Default configuration file
│   ├── datamodule.py       # Data loading utilities using PyTorch Lightning
│   ├── gpt2.py             # Core GPT-2 model architecture
│   ├── main.ipynb          # only for testing, ignore
│   ├── model.py            # PyTorch Lightning wrapper for GPT-2
│   ├── sample.py           # Text generation script
│   ├── train.py            # Main training script
│   └── utils.py            
├── papers/
├── .gitignore
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
