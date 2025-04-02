# diffusion-llms
Making LLMs inference faster with diffusion.

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

# Jack e' piu forte