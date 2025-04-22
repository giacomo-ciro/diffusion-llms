# Diffusion LLMs

The authors of the DiffuGPT paper adapted an Autoregressive Language Model (ARM), namely GPT-2, to obtain a Diffusion Language Model (DLM) while leveraging the pre-trained weights with almost no loss in performance. 

However, they did not address the issue of fixed output length which comes with discrete diffusion. In fact, DiffuGPT is only capable of generating output whose length is the full diffusion context size, which is not always the best choice and can also hinder performance in some cases (suppose we want a yes/no answer to a prompt, but the model is forced to generate a 512 tokens output). 

In this project, we aim to further adapt DiffuGPT, making it capable of variable length generation.

First, we test an approach similar to the one proposed in the Llada paper: we continue pre-training with a custom dataset of [text + eos + pad], to give DiffuGPT the ability to generate a pad token, allowing it to limit the length of its ouput.

Then, we test how confident our model is in predicting the EoS token at the first step of the diffusion generation, which is fundamental to avoid computing all the subsequent pad tokens and save computations.

Finally, we propose a method to improve the model capacity of predicting the EoS token: fine-tuning on [text + mask + EoS + mask] examples to improve the model's accuracy in predicting the eos token at the first diffusion step by only looking at the provided context.

## TO-DO's
- [ ] Check correctness of `prepare_var_len.py` script (generate dataset of sequences of text + eos + pad. The length of the sequences is the same thanks to the pad token, but the underlying content has variable length)
- [ ] Check correctness of the implmenetation in `model.py` (correct masking, correct loss computation, correct shifting etc)
- [ ] Further pre-train DiffuGPT on the var len dataset
- [ ] Create `eval.py` script to measure performance on benchmarks (the ones used in DiffuGPT paper, can copy from evaluation script on their github)
- [ ] Train of dataset of text + mask + eos + mask (to improve eos accuracy)
- [ ] Measure eos accuracy (does it actually improve?)

### Feedback from Professor
- ambition is good, doability is the question
- concretize the chance of success - a series of questions that can be answered quickly at the beginning
- control how we compare the different models, what kind of benchmarks and metrics we want to use (throughput: tokens per second with minimal perplexity loss)
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
To train, you first have to generate the training data and save it to a `np.memmap` object:
```bash
$ cd diffusion-llms/data
$ python prepare.py 100
```
Where we are specifying the number of documents to use from the FineWebDataset. This creates a unique memmap array of tokens, correspodning to the documents in the dataset separated by the eos tokens. We are nor ready to train. We specify the hyper-parameters in the `config.json` file and start training:

```bash
$ cd diffusion-llms
$ python train.py path/to/config.json
```

### Sampling
We can sample using diffusion or autoregressive strategy from any model. We specify a combination of `pipeline` (diffusion or arm) and `init_from` keys in the `config.json`. Then we add the `user_prompt` and the generation arguments (top k, temperature, number of samples, diffusion steps). Then run:
``` bash 
$ cd diffusion-llms
$ python sample.py path/to/config.json
```
Notice that we can also generate using discrete diffusion from a arm gpt2, and the results will likely be bad.

## Misc
1. Use own branch during development, together we handle merges.
2. Duplicate `config.json`, rename to `local_config.json` (added to `.gitignore`) and use it to test locally.
3. Login (locally or on hpc) to wandb from the CLI by running the following:
```bash 
$ python
>>> import wandb
>>> wandb.login()
```

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

## References

- **(LLaDa)** Nie et al. (2025): [_Large Language Diffusion Models_](https://arxiv.org/pdf/2502.09992)
- **(Block Diffusion)** Arriola et al. (ICLR 2025): [_Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models_](https://arxiv.org/pdf/2503.09573)
- **(DiffuGPT)** Gong et al. (ICLR 2025): [_Scaling Diffusion Language Models via Adaptation from Autoregressive Models_](https://arxiv.org/pdf/2410.17891)
- **(DiffusionBERT)** Gong et al. (2022): [_DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models_](https://arxiv.org/pdf/2211.15029)
- **(Scaling Laws)** Liang et al. (2024): [_Scaling Laws for Diffusion Transformers_](https://arxiv.org/pdf/2410.08184)
- **(Survey)** Zou et al. (2023): [_A Survey of Diffusion Models in Natural Language Processing_](https://arxiv.org/pdf/2305.14671)
