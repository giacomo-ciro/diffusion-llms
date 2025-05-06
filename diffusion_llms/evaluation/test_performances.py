"""
Test the inference speed of a model.
Usage:
    python test_performances.py -i <number_of_iterations> -l <length_of_each_sequence> 
    
    (the interface is to be completed)"""


import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time, sys, os
import argparse
from eval import get_device
import pandas as pd 
sys.path.append("../")
from model import GPT2
import json
import tiktoken




def speed_test_gpt2(max_length, n_sequences, parallel = False):

    device = get_device()
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # nb, il modello deve essere compilato
    model = model.to(device)
    model.eval()

    dataset = pd.read_csv("./evaluation/train_prompts.csv")
    prompts = dataset["user_prompt"]
    if not parallel:
        start = time.time()
        n_tokens_generated = 0
        for i in range(n_sequences):
            prompt = prompts[i]
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_attention_mask=True,
            ).to(device)
            max_new_tokens = max_length - inputs["input_ids"].shape[1]
            n_tokens_generated += max_new_tokens
            with torch.no_grad():
                # with torch.amp.autocast(device_type=device):
                    # Generate text
                output = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    num_return_sequences=1,
                    do_sample=False,
                    repetition_penalty=1.2,
                    temperature=1.0,
                    top_k=50,
                        
                        )
            
        end = time.time()
    else:
        prompts = prompts[:n_sequences].to_list()
        start = time.time()
        inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_attention_mask=True,
            ).to(device)
        max_new_tokens = max_length
        n_tokens_generated = max_new_tokens * n_sequences
        with torch.no_grad():
                # with torch.amp.autocast(device_type=device):
                    # Generate text
            output = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_return_sequences=n_sequences,
                do_sample=True,
                repetition_penalty=1.2,
                temperature=1.0,
                top_k=50,
                    )

        end = time.time()
    print(f"Generated {n_sequences} responses, {n_tokens_generated} total tokens, in {end - start:.4f} seconds")
    print(f"Average time per prompt: {(end - start) / n_sequences:.4f} sec")
    print(f"Average time per 1000 tokens generated: {1000* (end - start) / (n_tokens_generated):.4f} sec")
    print(f"Average speed: {n_tokens_generated / (end - start):.4f} tokens/sec")



def test_DiffuGPT_inference(
    length , n_sequences , input_text, device
):
    pass


def test_our_inference(
    length , n_sequences , input_text, device
):
    pass




def speed_test(model, tokenizer, config, max_length, n_sequences, device):
    """Test the inference speed of a model."""
    dataset = pd.read_csv("./evaluation/train_prompts.csv")
    prompts = dataset["user_prompt"]
    start = time.time()
    n_tokens_generated = 0
    for i in range(n_sequences):
        prompt = prompts[i]
        input_ids = (
                torch.tensor([50256] + tokenizer.encode(" ".join(prompt.split()[:-1])))
                .unsqueeze(0)
                .to(device)
            )
        max_new_tokens = max_length - input_ids.shape[1]
        n_tokens_generated += max_new_tokens
        model.generate(
                pipeline = config["pipeline"],
                input_ids = input_ids,
                max_new_tokens = max_new_tokens,
                temperature = config["temperature"],
                top_k = config["top_k"],
                denoising_strategy = config["denoising_strategy"],    # "random" / "entropy"
                diffusion_steps = config["diffusion_steps"],
                do_sample = config["do_sample"],        
                repetition_penalty = config["repetition_penalty"],               
            )
        
    end = time.time()
    print(f"Generated {n_sequences} responses, {n_tokens_generated} total tokens, in {end - start:.4f} seconds")
    print(f"Average time per prompt: {(end - start) / n_sequences:.4f} sec")
    print(f"Average time per 1000 tokens generated: {1000* (end - start) / (n_tokens_generated):.4f} sec")
    print(f"Average speed: {n_tokens_generated / (end - start):.4f} tokens/sec")












# if __name__ == "__main__":

#     # # Parse command-line arguments
#     # parser = argparse.ArgumentParser(description="Test Model inference performance.")
#     # parser.add_argument("model_name", type=str, help="Name of the model to test. (gpt2 OR diffugpt OR our)")
#     # parser.add_argument("length", type=int, help="Length of the generated text.")
#     # parser.add_argument("n_sequences", type=int, help="Number of sequences to generate.")
#     # args = parser.parse_args()
#     # if args.model_name not in ["gpt2", "diffugpt", "our"]:
#     #     raise ValueError("Invalid model name. Choose from 'gpt2', 'diffugpt', or 'our'.")
    
#     # device = get_device()
#     # input_text = "Once upon a time, in a land far, far away,"

#     # if args.model_name == "gpt2":
#     #     test_function = test_GPT2_inference
#     # elif args.model_name == "diffugpt":
#     #     test_function = test_DiffuGPT_inference
#     # else:
#     #     test_function = test_our_inference

        
#     # print(f"Testing model: {args.model_name} on device: {device}")
#     # test_function(
#     #         length=args.length,
#     #         n_sequences=args.n_sequences,
#     #         input_text= input_text,
#     #         device=device,)
#     # print("Test completed.")
#     model = 
#     speed_test()



def main():
    # From the command line we can specify the config.file
    parsers = argparse.ArgumentParser()
    parsers.add_argument("config", type=str, help="path/to/config.json")
    parsers.add_argument("-i", "--iters", type=int, default=float("inf"))
    parsers.add_argument("-l", "--length", type=int, default=512)
    args = parsers.parse_args()

    CONFIG_PATH = args.config
    SEQUENCE_LENGTH = args.length
    MAX_ITER = args.iters
    print(
        f"CONFIG_PATH = {CONFIG_PATH}\n"
        f"SEQUENCE_LENGTH = {SEQUENCE_LENGTH}\n"
        F"MAX_ITER = {MAX_ITER}"
    )
        
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)


        
    # Tokenize
    tokenizer = tiktoken.get_encoding("gpt2")

    # Load model
    device = get_device()
    
    # Instantiate a model (new or pretrained)
    if os.path.exists(config["init_from"]):
        model = GPT2.from_pretrained(config["init_from"])
    else:
        model = GPT2(CONFIG_PATH)
    
    # Compile and set to evaluation mode
    # model = torch.compile(model).to(device)
    model = model.to(device)
    model.eval()
    speed_test(model, tokenizer, config, SEQUENCE_LENGTH, MAX_ITER, device)
    # speed_test_gpt2(max_length=SEQUENCE_LENGTH, n_sequences=MAX_ITER, parallel=False)


if __name__ == "__main__":
    main()
