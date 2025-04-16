# diffusion-llms
Making LLMs inference faster with diffusion.

# Proposed Research
DiffuGPT managed to adapt GPT-2, effectively converting a Autoregressive Language Model (ARM) to a Diffusion Language Model (DLM). However, they did not address the output length, making DiffuGPT only capable of generating output with fixed length. In this project, we aim to further train DiffuGPT, making it capable of variable length generation.

First, we an approach similar to the one proposed in the Llada paper (training with custom dataset of answer + prompt + EoS + pad tokens).

Then, we test how confident our model is in predicting the EoS token at the first step of the diffusion generation (to avoid computing all the pad token and save computations).

Finally, we propose methods to improve the model capacity of predicting the EoS token (e.g., by fine-tuning on prompt + msk + EoS + msk) and evaluate the first-step EoS prediction ability.

## Roadmap
- [ ] train GPT2 without and with diffusion (+mask, +pad (`max_length(oww)-token)`)
- [ ] check what training data DiffuGPT uses 
- [ ] should we train one-shot or add pad token to the training data?
- [ ] choose as baseline DiffuGPT naive approach: choose token wrt EoS pad token (pass prompt with mask and pad)
- [ ] instruction tuning
- [ ] can we adapt a diffusion adapted model to variable length generation? 
- [ ] check how DiffuGPT tackles dynamic length inference (there is no pad token we need to check if there is an EoS token)
- [ ] confirm if in LLaDa they mask with a proportion t ~ U(0,1)

### Extra
- [ ] explore how many tokens you can infer on parallel - if there is a gain on inference time wrt to diffusion w/o padding and wrt to ARM

### Feedback from Professor
- ambition is good, doability is the question
- concretize the chance of success - a series of questions that can be answered quickly at the beginning
- control how we compare the different models, what kind of benchmarks and metrics we want to use (throughput: tokens per second with minimal perplexity loss)
- be very explicit about research question, don’t fear to be overly specific, also be open about the limitations
- change formulations to see if changing head affects anything: robustness checks
- find sources that do not affect variance
- walk the reader through the resulting paper

## TODOs
- [x] ~~Setup WandB project and logging~~
- [x] ~~Update README with wandB instructions~~
- [x] ~~Compute lr decay steps automatically (e.g. 0.6 of total steps)~~
- [ ] fix self.log warning in sample.py 
- [ ] Adapt gpt2 for diffusion, obtain DiffuGPT_ours
- [ ] Implement dynamic length inference (at first step, look for the token with highest < pad > probability and return it to set an upper bound, then proceed with diffusion sampling as in the other papers)
- [ ] Test dynamic length inference on DiffuGPT, DiffuLAMA, DiffuGPT_ours, LlaDa
- [ ] Setup init from local weights
- [ ] Update README with instructions for running on HPC
- [x] ~~Implement attention mask annealing~~
- [x] ~~Adapt datamodule to diffusion pipeline~~
- [ ] Check the implementation of discrete diffusion training works as expected

## Overview
### Research Question
(tl;dr) We explored previous research trying to overcome the issue with fixed-length outputs in diffusion models compromising between diffusion and auto-regression. We propose a variable length diffusion generation that is fully diffusion.

### Discrete Diffusion Framework
Training step for diffusion model:  
    1. Input sequence $X\in\mathbb{R}^n$ of integers  
    2. Sample $t \sim U(0,1)$  
    3. Mask each token in the input sequence with probability $t$  
    4. Pass to the model to get logits for all  
    5. Measure cross entropy loss only on masked tokens  
### Notes
When running on hpc, gpt2 small with batch_size = 8 is the largest it can be (using both 1080 gpus...)

Sample response from `checkpoints/ymd_250405_HMS_21_15_27/epoch_0_ce_1.86.ckpt` to the prompt `Hello, what's your name?`:
```
D-par can come to make push is creatures movie The increases in the my partner for the New Leaf - 28, casual fans of those minor surgery on 1994 philosophy by the hand. From the other countries where they outlined an independent son, a lot more�/11.

But there“Lots of different types of the inclusion of course, something terrible, apologetic - but by publishers action
```
#### IDEA FOR NLP
Compare attention weights between auto-regressive and diffusion models.

> from NLP slides `01_intro`: Graded on data set size, correctness of implementations, annotation
quality, performance, originality, and ambition

## Rules
### Branches
Everybody **must** use its own branch during development, together we handle merges.

Create a branch locally:
```
git branch <branch_name>
```
Move to that branch and work on it
```
git checkout <branch_name>
```
Push local branch to remote branch (create remote branch if not existing)
```
git push -u origin <branch-name>
```
Check what remote branch your current branch is tracking
```
git branch -vv
```
Check list of available branches
```
git branch -a
```
### Configuration file
Please, duplicate `config.json`, rename to `local_config.json` (added to `.gitignore`) and modify this to test locally.

## Usage

### WandB login
To use wandb, you must login (locally or on hpc). From the CLI, run the following (and follow the prompts):
```
$ python
>>> import wandb
>>> wandb.login()
```

### Train a model
Specify in the `config.json` file the parameters of the training. The key `init_from` is used to specify the starting point. If one of `['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']`, then it downloads the weights from huggingface and instantiate a pre-trained model. Any other value backs off to init from scratch. Then start the training:
```
cd diffusion-llms
python train.py path/to/config.json
```

### Sample from model
In the `config.json` file using the GPT-2 implementation from [Andrej Karpathy](https://github.com/karpathy/nanoGPT).

Download the weights of gpt-2 from huggingface, instantiate a `GPT()` model class from `model.py`, load `configurator.py` and sample from the model. Specifying the prompt and the number of answers to generate.
``` 
$ cd diffusion-llms
$ python sample.py path/to/config
```

#### Using a saved checkpoint from training
Modify `sample.py` to add:
```python
from lightning.pytorch import LightningModule
model = GPT2.load_from_checkpoint("path/to/checkpoint.ckpt", config_path=CONFIG_PATH)
```

The configuration file should specify:
- `user_prompt`: The text to use as a starting point
- `n_samples`: How many text samples to generate

## Repository Structure
```bash
├── diffusion-llms/ 
│   ├── checkpoints/        # Checkpoint files for model weights
│   ├── data/openwebtext/ 
│       ├── prepare.py      # Script for tokenizing and preparing dataset
│       ├── train_1M.bin
│       ├── train_1M.txt
│       └── etc.
│   ├── attention_patch.py
│   ├── config.json         # Default configuration file
│   ├── configurator.py     # Configuration utilities
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

## Virtual Environment (conda)
To ensure dependencies are cleanly managed and consistent, create and activate a conda environment:

```bash
conda deactivate # if another environment is currently active
conda create --name dl-nlp python=3.12
conda activate dl-nlp
```

## Requirements
Install the necessary packages using `pip`:

```bash
python -m pip install -r requirements.txt
```

> Using `python -m pip` ensures that packages are installed in the environment linked to your current Python interpreter, avoiding issues with multiple Python installations.

## References

- **(LLaDa)** Nie et al. (2025): [_Large Language Diffusion Models_](https://arxiv.org/pdf/2502.09992)
- **(Block Diffusion)** Arriola et al. (ICLR 2025): [_Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models_](https://arxiv.org/pdf/2503.09573)
- **(Adapt-Diff)** Gong et al. (ICLR 2025): [_Scaling Diffusion Language Models via Adaptation from Autoregressive Models_](https://arxiv.org/pdf/2410.17891)
- **(DiffusionBERT)** Gong et al. (2022): [_DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models_](https://arxiv.org/pdf/2211.15029)
- **(Scaling Laws)** Liang et al. (2024): [_Scaling Laws for Diffusion Transformers_](https://arxiv.org/pdf/2410.08184)
- **(Survey)** Zou et al. (2023): [_A Survey of Diffusion Models in Natural Language Processing_](https://arxiv.org/pdf/2305.14671)
