# diffusion-llms
Making LLMs inference faster with diffusion.

## TODOs
- [ ] Adapt gpt2 for diffusion, obtain DiffuGPT_ours
- [ ] Implement dynamic length inference (at first step, look for the token with highest < pad > probability and return it to set an upper bound, then proceed with diffusion sampling as in the other papers)
- [ ] Test dynamic length inference on DiffuGPT, DiffuLAMA, DiffuGPT_ours, LlaDa
## Overview
### Research Question
We explored previous research trying to overcome the issue with fixed-length outputs in diffusion models compromising between diffusion and auto-regression. We propose a variable length diffusion generation that is fully diffusion.

#### IDEA FOR NLP
Compare attention weights between auto-regressive and diffusion models.

### Technical Soundness (Experimental Strategy)

## Rules
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

## Sample from GPT-2
Using the GPT-2 implementation from [Andrej Karpathy](https://github.com/karpathy/nanoGPT).

Download the weights of gpt-2 from huggingface, instantiate a `GPT()` model class from `model.py`, load `configurator.py` and sample from the model.
```
cd diffusion-llms
python sample.py --init_from=gpt2 --start="What is the answer to life?" --num_samples=1 --max_new_tokens=100 --device=cpu
```

In `sample.py` the following line is used to override the declared variable inside the script:
```
exec(open('configurator.py').read())
```
What it does is it executes the content of the file `configurator.py`, which reads the command linle args and overrides the previously declared variables in `globals()`. Effectively, we are in file A, and runnning the content of file B which modifies the global variables previously declared in file A. (TODO replace all this with a `config.json` file)

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
