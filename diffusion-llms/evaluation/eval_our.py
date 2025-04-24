import sys
import json

import torch
import tiktoken
from model import GPT2

def eval_Lambada(model, tokenizer, config):
    total_cnt = 0
    cor = 0
    mask_token = tokenizer.decode([config["mask_id"]])
    temperature = config.get("temperature")
    top_k = config.get("top_k")
    with open('evaluation/lambada_test_plain_text.txt', 'r', encoding='utf-8') as file:
        for line in file:
            total_cnt += 1
            if total_cnt % 1000 == 0:
                print('total cnt:', total_cnt)
            line = line.strip()
            x0 = torch.tensor(
                [50256] + tokenizer.encode(line)
                ).unsqueeze(0).to('cuda')          # x0 is all the line
            prefix = torch.tensor(
                [50256] + tokenizer.encode(' '.join(line.split()[:-1]))).unsqueeze(0).to('cuda') # prefix is all the line except the last word
            masked_nums = x0.shape[1] - prefix.shape[1]
            # args.diffusion_steps = masked_nums
            # args.logits_temp = 1.0
            # res = generate_samples(model, args, tokenizer, inputs, eval=True)
            # manca di mettere il diffusion steps
            print("masked nums: ", masked_nums)
            xs = model.generate(
                prefix, 
                max_new_tokens=masked_nums,
                temperature=temperature,
                top_k=top_k,
                )
            x = xs[-1][0]
            pred = tokenizer.decode(
                x.tolist()[1:]
                ).replace(mask_token, "<mask>")
            if pred.strip() == line.strip():
                cor += 1
            # print(pred.strip())
            # print(line.strip())
            # probabilmente c'è un errore nel loro codice, perchè conta la corretta 2 volte
            # if pred.strip() == line.split()[-1].strip():
            #     cor += 1

            # print_output(x, tokenizer, mask_token)
            # print(line)
            # if total_cnt > 3:
            #     break
    print('acc:', cor/total_cnt)


def print_output(x, tokenizer, mask_token):
    print("-"*89)
    out = tokenizer.decode(
        x.tolist()[1:]
        ).replace(mask_token, "<mask>")
    print(out)
    print("-"*89)


def main():
    # From the command line we can specify the config.file
    if len(sys.argv) == 2:
        CONFIG_PATH = sys.argv[1]
    else:
        print("No path/to/config.json provided, defaulting to \'./config.json\'")
        CONFIG_PATH = './config.json'

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    # Tokenize
    tokenizer= tiktoken.get_encoding("gpt2")

    # Mask token
    mask_token = tokenizer.decode([config["mask_id"]])

    # Get prompt
    input_ids = torch.tensor(
        [50256] + tokenizer.encode(config["user_prompt"])
    ).unsqueeze(0)

    # Load model
    model = GPT2(CONFIG_PATH).to('cuda')
    # model = torch.compile(model).to('cuda')

        
    # # List of tensors of shape (B, seq_len)
    # xs = model.generate(
    #     input_ids, 
    #     max_new_tokens=config.get("max_new_tokens", 128),
    #     temperature=config.get("temperature"),
    #     top_k=config.get("top_k"),
    # )

    # # Illustrate the diffusion process
    # print_output(xs[-1][0], tokenizer, mask_token)

    eval_Lambada(model, tokenizer, config)



if __name__ == "__main__":
    main()
