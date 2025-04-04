# diffusion-llms
Making LLMs inference faster with diffusion.

## TODOs
- [ ] Setup WandB project and logging
- [ ] Adapt gpt2 for diffusion, obtain DiffuGPT_ours
- [ ] Implement dynamic length inference (at first step, look for the token with highest < pad > probability and return it to set an upper bound, then proceed with diffusion sampling as in the other papers)
- [ ] Test dynamic length inference on DiffuGPT, DiffuLAMA, DiffuGPT_ours, LlaDa
- [ ] Setup checkpointing (save weights and init from local weights)
## Overview
### Research Question
We explored previous research trying to overcome the issue with fixed-length outputs in diffusion models compromising between diffusion and auto-regression. We propose a variable length diffusion generation that is fully diffusion.

#### IDEA FOR NLP
Compare attention weights between auto-regressive and diffusion models.

### Technical Soundness (Experimental Strategy)

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
### Sample from model
In the `config.json` file.sing the GPT-2 implementation from [Andrej Karpathy](https://github.com/karpathy/nanoGPT).

Download the weights of gpt-2 from huggingface, instantiate a `GPT()` model class from `model.py`, load `configurator.py` and sample from the model. Specifying the prompt and the number of answers to generate.
``` 
$ cd diffusion-llms
$ python sample.py path/to/config
```

### Train a model
Specify in the `config.json` file the parameters of the training. The key `init_from` is used to specify the starting point. If one of `['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']`, then it downloads the weights from huggingface and instantiate a pre-trained model. Any other value backs off to init from scratch. Then start the training:
```
cd diffusion-llms
python train.py path/to/config.json
```

### WandB login
To use wandb, you must login (locally or on hpc). From the CLI, run the following (and follow the prompts):
```
$ python
>>> import wandb
>>> wandb.login()
```

## Repository Structure
```zsh
├── diffusion-llms/ 
│   ├── __pycache__/
│   ├── config.json
│   ├── configurator.py 
│   ├── gpt2.py 
│   ├── main.py 
│   ├── model.py 
│   └── sample.py 
├── .gitignore
├── LICENSE
└── README.md
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
python -m pip install torch tiktoken lightning transformers
```

These are required for:
- `torch`: deep learning (model, training, tensor ops)
- `tiktoken`: tokenizer used by OpenAI models
- `lightning`: PyTorch Lightning for training loop abstraction
- `transformers`: Hugging Face Transformers library for pre-trained models and tokenizers

> Using `python -m pip` ensures that packages are installed in the environment linked to your current Python interpreter, avoiding issues with multiple Python installations.

## References
- [Large Language Diffusion Models](https://arxiv.org/pdf/2502.09992)
- [BLOCK DIFFUSION: INTERPOLATING BETWEEN AUTOREGRESSIVE AND DIFFUSION LANGUAGE MODELS](https://arxiv.org/pdf/2503.09573)
- [SCALING DIFFUSION LANGUAGE MODELS VIA ADAPTATION FROM AUTOREGRESSIVE MODELS](https://arxiv.org/pdf/2410.17891)
- [DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models](https://arxiv.org/pdf/2211.15029)
- [SCALING LAWS FOR DIFFUSION TRANSFORMERS](https://arxiv.org/pdf/2410.08184)
