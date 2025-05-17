"""
Test the inference speed of a model.
Usage:
    python test_performances.py -i <number_of_iterations> -l <length_of_each_sequence> 
    
"""


import torch
import time
import argparse
from diffusion_llms.utils import get_device
import pandas as pd 
from transformers import AutoTokenizer, AutoModel
from diffusion_llms.LLaDA.generate import generate




def speed_test_distilbert_reg(n_sequences, verbose=False):
    """Test the inference speed of a model."""
    from diffusion_llms.baseline.model_baseline import DistilBertRegressor
    device = get_device()
    ### TODO: create model and tokenizer ###################################

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)    


    ########################################################################
    dataset = pd.read_csv(".diffusion_llms/data/train.csv")
    prompts = dataset["user_prompt"]
    # Create DistilBERT regressor to predict the length of the answer
    length_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    length_predictor_model = DistilBertRegressor()
    length_predictor_model.load_state_dict(torch.load("diffusion_llms/baseline/checkpoints/DistilBERT_DGPT_reg.pth"))
    length_predictor_model.to(device)
    length_predictor_model.eval()
    start = time.time()

    n_tokens_generated = 0
    for i in range(n_sequences):
        prompt = prompts[i]
        # Get the length of the answer
        with torch.no_grad():
            input_enc = length_tokenizer(prompt, return_tensors="pt", padding='max_length', truncation=True, max_length=128).to(device)
            pred, _ = length_predictor_model(input_enc['input_ids'], input_enc['attention_mask'])
            max_new_tokens = int(pred.item())
            print("MAX_NEW_TOKENS", max_new_tokens)

        ###### GENERATION OF THE ANSWER ##########################

        # Add special tokens for the Instruct model. The Base model does not require the following two lines.
        m = [{"role": "user", "content": prompt}, ]
        prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        gen_length = max_new_tokens + input_ids.shape[1]
        # let's do for multiple of 32
        block_length = 32
        gen_length = (gen_length // block_length + 1) * block_length
        max_new_tokens = gen_length - input_ids.shape[1]
        steps = gen_length

        out = generate(model, input_ids, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., cfg_scale=0., remasking='low_confidence')
        if verbose:
            print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])
            # could be interesting to see the position of the eos token

        ############################################################

        n_tokens_generated += max_new_tokens


        
    end = time.time()
    print(f"Generated {n_sequences} responses, {n_tokens_generated} total tokens, in {end - start:.4f} seconds")
    print(f"Average time per prompt: {(end - start) / n_sequences:.4f} sec")
    print(f"Average time per 1000 tokens generated: {1000* (end - start) / (n_tokens_generated):.4f} sec")
    print(f"Average speed: {n_tokens_generated / (end - start):.4f} tokens/sec")






def main():
    # From the command line we can specify the config.file
    parsers = argparse.ArgumentParser()
    parsers.add_argument(
        "-t", "--type", type=str, default="dostillbert_regressor",
    )
    parsers.add_argument("-i", "--iters", type=int, default=100)
    parsers.add_argument("-l", "--length", type=int, default=512)
    args = parsers.parse_args()

    CONFIG_PATH = args.config
    SEQUENCE_LENGTH = args.length
    MAX_ITER = args.iters
    print(
        f"SEQUENCE_LENGTH = {SEQUENCE_LENGTH}\n"
        F"MAX_ITER = {MAX_ITER}"
    )
        
    if type == "distilbert_regressor":
        speed_test_distilbert_reg(n_sequences=MAX_ITER)
    # speed_test_gpt2(max_length=SEQUENCE_LENGTH, n_sequences=MAX_ITER, parallel=False)


if __name__ == "__main__":
    main()
