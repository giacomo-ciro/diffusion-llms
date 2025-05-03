"""
This script evaluates the performance of a GPT-2 model on various tasks such 
as Lambada, HellaSwag, Winogrande, PIQA, Social IQA, and infilling tasks. 

It uses the Hugging Face datasets library to load the datasets and evaluates 
the model's performance using accuracy and F1 score metrics. 

The script also allows for the evaluation of the model on a specific task by 
specifying the task name as a command line argument.

HOW TO USE:
Run the script from the command line with the following command:
   $ python eval.py <path/to/config.json> <EVALUATION_TYPE> [max_iter]
   where <path/to/config.json> is the path to the configuration file 
   <EVALUATION_TYPE> can be one of the following:
   - lambada
   - hellaswag
   - wino
   - piqa
   - siqa
   - infilling
   - trivia
   and [max_iter] is an optional argument to limit the number of iterations for evaluation.
"""

import sys
import json
import csv
import os
import argparse

import evaluate
import torch
import tiktoken
from f1 import compute_f1, normalize_answer

import re
import numpy as np
import time

# To load model from parent directory
sys.path.append("../")
from model import GPT2

def get_device():
    """Get the device to use for PyTorch operations (CPU or cuda or MPS)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def eval_Lambada(model, tokenizer, config, max_iter=np.inf):
    print("Evaluating lambada...")
    print("Setting diffusion_steps = masked_num")
    start_time = time.time()
    device = get_device()
    total_cnt = 0
    cor = 0
    mask_token = tokenizer.decode([config["mask_id"]])

    with open("evaluation/lambada_test_plain_text.txt", "r", encoding="utf-8") as file:
        for line in file:
            total_cnt += 1
            line = line.strip()
            x0 = (
                torch.tensor([50256] + tokenizer.encode(line)).unsqueeze(0).to(device)
            )  # x0 is all the line
            prefix = (
                torch.tensor([50256] + tokenizer.encode(" ".join(line.split()[:-1])))
                .unsqueeze(0)
                .to(device)
            )  # prefix is all the line except the last word
            masked_nums = x0.shape[1] - prefix.shape[1]
            xs = model.generate(
                pipeline = config["pipeline"],
                input_ids = prefix,
                max_new_tokens = masked_nums,
                temperature = config["temperature"],
                top_k = config["top_k"],
                denoising_strategy = config["denoising_strategy"],    # "random" / "entropy"
                diffusion_steps = masked_nums                       # NOTE: in diffugpt they set the diffusion step it in this way
            )
            x = xs[-1][0]
            pred = tokenizer.decode(x.tolist()[1:]).replace(mask_token, "<mask>")
            if pred.strip() == line.strip():
                cor += 1

            # probabilmente c'è un errore nel loro codice, perchè conta la corretta 2 volte
            # if pred.strip() == line.split()[-1].strip():
            #     cor += 1

            print(f"step {total_cnt} done. Accuracy: {cor/total_cnt:.4f}")
            if total_cnt > max_iter:
                break
    
    ans = cor / total_cnt
    print("acc:", ans)
    print(f"speed: {(total_cnt - 1) / (time.time() - start_time):.3f} step/sec")
    return ans


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def eval_hellaswag(model, tokenizer, config, max_iter=np.inf):
    assert(config.get("pipeline") == "diffusion")           # Implemented only for diffusion pipeline
    assert(not config.get("use_eos_head"))          # Must be False 
    from datasets import load_dataset
    print("Evaluating HellaSwag...")
    start_time = time.time()

    device = get_device()

    ds = load_dataset("Rowan/hellaswag", split="validation")

    total_cnt = 0
    cor = 0

    for doc in ds:
        total_cnt += 1
        if total_cnt % 1000 == 0:
            print("total cnt:", total_cnt)
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()

        query = preprocess(doc["activity_label"] + ": " + ctx)
        choices = [preprocess(ending) for ending in doc["endings"]]
        gold = int(doc["label"])
        score_list = []
        prefix = [50256] + tokenizer.encode(query)

        for choice in choices:

            x0 = prefix + tokenizer.encode(choice)
            input_ids = torch.tensor(x0).unsqueeze(0).to(device)  # x0 is all the line
            src_mask = torch.Tensor(
                [[1] * len(prefix) + [0] * (len(x0) - len(prefix))]
            ).to(device)
            score = model.eval_forward(input_ids, src_mask)
            score_list.append(score)
        pred = np.argmin(np.array(score_list))

        if pred == gold:
            cor += 1
        print(f"step {total_cnt} done. Accuracy: {cor/total_cnt:.4f}")
        if total_cnt >= max_iter:
            break
    
    ans = cor / total_cnt
    print("acc:", ans)
    print(f"speed: {(total_cnt - 1) / (time.time() - start_time):.3f} step/sec")
    return ans

def eval_wino(model, tokenizer, config, max_iter=np.inf):
    assert(config.get("pipeline") == "diffusion")           # Implemented only for diffusion pipeline
    assert(not config.get("use_eos_head"))          # Must be False 
    from datasets import load_dataset
    print("Evaluating Winogrande...")
    start_time = time.time()

    device = get_device()

    ds = load_dataset(
        "allenai/winogrande",
        "winogrande_xl",
        split="validation",
        trust_remote_code=True,
    )

    total_cnt = 0
    cor = 0

    for doc in ds:
        total_cnt += 1
        idx = doc["sentence"].index("_")
        options = [doc["option1"], doc["option2"]]
        answer_to_num = {"1": 0, "2": 1}
        gold = answer_to_num[doc["answer"]]
        score_list = []

        for opt in options:
            target_str = opt
            suffix_str = doc["sentence"][idx + 1 :].strip()
            target_id = tokenizer.encode(target_str)
            suffix_id = tokenizer.encode(suffix_str)
            prefix_str = doc["sentence"][:idx]
            prefix_id = [50256] + tokenizer.encode(prefix_str)

            x0 = prefix_id + target_id + suffix_id
            # src_mask = torch.Tensor([[1]*len(prefix_id)+[0]*(len(x0)-len(prefix_id))]).to(device)      # in diffugpt they do it in this way, but it doesn't make much sense
            src_mask = torch.Tensor(
                [[1] * len(prefix_id) + [0] * len(target_id) + [1] * len(suffix_id)]
            ).to(device)

            input_ids = torch.tensor(x0).unsqueeze(0).to(device)  # x0 is all the line
            score = model.eval_forward(input_ids, src_mask)
            score_list.append(score)
        pred = np.argmin(np.array(score_list))

        if pred == gold:
            cor += 1
        print(f"step {total_cnt} done. Accuracy: {cor/total_cnt:.4f}")

        if total_cnt >= max_iter:
            break
    ans = cor / total_cnt
    print("acc:", ans)
    print(f"speed: {(total_cnt - 1) / (time.time() - start_time):.3f} step/sec")
    return ans

def eval_piqa(model, tokenizer, config, max_iter=np.inf):
    assert(config.get("pipeline") == "diffusion")           # Implemented only for diffusion pipeline
    assert(not config.get("use_eos_head"))          # Must be False 

    from datasets import load_dataset
    print("Evaluating PIQA...")
    start_time = time.time()

    device = get_device()

    ds = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
    total_cnt = 0
    cor = 0

    for doc in ds:
        total_cnt += 1

        query = f"Question: {doc['goal']}\nAnswer: "
        choices = [doc["sol1"], doc["sol2"]]
        gold = doc["label"]

        score_list = []
        prefix = [50256] + tokenizer.encode(query)

        for choice in choices:

            x0 = prefix + tokenizer.encode(" " + choice)
            src_mask = torch.Tensor(
                [[1] * len(prefix) + [0] * (len(x0) - len(prefix))]
            ).to(device)
            input_ids = torch.tensor(x0).unsqueeze(0).to(device)
            score = model.eval_forward(input_ids, src_mask)
            score_list.append(score)
        pred = np.argmin(np.array(score_list))

        if pred == gold:
            cor += 1
        # print(total_cnt, cor/total_cnt)
        print(f"step {total_cnt} done. Accuracy: {cor/total_cnt:.4f}")
        if total_cnt >= max_iter:
            break
    ans = cor / total_cnt
    print("acc:", ans)
    print(f"speed: {(total_cnt - 1) / (time.time() - start_time):.3f} step/sec")
    return ans

def eval_siqa(model, tokenizer, config, max_iter=np.inf):
    assert(config.get("pipeline") == "diffusion")           # Implemented only for diffusion pipeline
    assert(not config.get("use_eos_head"))          # Must be False 
    print("Evaluating Social IQA...")
    start_time = time.time()
    from datasets import load_dataset

    device = get_device()

    ds = load_dataset("allenai/social_i_qa", split="validation", trust_remote_code=True)
    total_cnt = 0
    cor = 0

    for doc in ds:
        total_cnt += 1

        query = f"Question: {doc['context']} {doc['question']}\nAnswer: "
        choices = [doc["answerA"], doc["answerB"], doc["answerC"]]
        gold = int(doc["label"]) - 1

        score_list = []
        prefix = [50256] + tokenizer.encode(query)

        for choice in choices:

            x0 = prefix + tokenizer.encode(" " + choice)
            src_mask = torch.Tensor(
                [[1] * len(prefix) + [0] * (len(x0) - len(prefix))]
            ).to(device)
            input_ids = torch.tensor(x0).unsqueeze(0).to(device)
            score = model.eval_forward(input_ids, src_mask)
            score_list.append(score)
        pred = np.argmin(np.array(score_list))

        if pred == gold:
            cor += 1
        # print(total_cnt, cor/total_cnt)
        print(f"step {total_cnt} done. Accuracy: {cor/total_cnt:.4f}")
        if total_cnt >= max_iter:
            break
    
    ans = cor / total_cnt
    print("acc:", ans)
    print(f"speed: {(total_cnt - 1) / (time.time() - start_time):.3f} step/sec")
    return ans

def eval_infilling(model, tokenizer, config, max_iter=np.inf):
    """It does infilling on the true sentence in the Story Cloze Test, generating the middle sentence given the first
    and last sentences and computes the rouge score between the generated and the true middle sentence.
    """
    assert(config.get("pipeline") == "diffusion")           # Implemented only for diffusion pipeline
    print("Evaluating infilling on Story Cloze Test...")
    start_time = time.time()
    device = get_device()
    assert config.get("pipeline") == "diffusion"
    problems = []
    with open("evaluation/cloze_test_val__spring2016.csv") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            sents = row[1:-3] + [row[-3] if row[-1] == "1" else row[-2]]
            # sents = [s if i == 0 else " " + s for i, s in enumerate(sents)]
            problems.append(sents)

    samples = []
    total_cnt = 0
    gens = []
    refs = []

    for stories in problems:
        total_cnt += 1
        prompt = stories[0] + " " + stories[1]
        suffix = stories[3] + " " + stories[4]
        middle = stories[2]

        prefix = [50256] + tokenizer.encode(prompt)  # , add_special_tokens=False)
        suff = tokenizer.encode(suffix)  # , add_special_tokens=False)
        x0 = prefix + tokenizer.encode(middle) + suff  # , add_special_tokens=False)
        src_mask = torch.Tensor(
            [
                [1] * len(prefix)
                + [0] * (len(x0) - len(prefix) - len(suff))
                + [1] * len(suff)
            ]
        ).to(device)
        input_ids = torch.tensor(x0).unsqueeze(0).to(device)
        res = model.generate_infilling(
            input_ids,
            src_mask,
            temperature=config.get("temperature"),
            top_k=config.get("top_k"),
        )
        pred = tokenizer.decode(
            res.tolist()[0][len(prefix) - 1 : len(x0) - len(suff) - 1]
        )

        samples.append(dict(pred=pred, label=middle, prefix=prompt, suffix=suffix))
        gens.append(pred)
        refs.append(middle)

        print(f"step {total_cnt} done")
        if total_cnt == 1000:
            break
        if total_cnt >= max_iter:
            break

    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=gens, references=refs)
    for key in results.keys():
        results[key] *= 100
    results["rougeAvg"] = (
        results["rouge1"] + results["rouge2"] + results["rougeL"]
    ) / 3
    print(
        f"rouge1={results['rouge1']:.2f}, rouge2={results['rouge2']:.2f}, rougeL={results['rougeL']:.2f}, rougeAvg={results['rougeAvg']:.2f}"
    )
    print(f"speed: {(total_cnt - 1) / (time.time() - start_time):.3f} step/sec")

    with open(f'ROCInfill_medium_t{config.get("diffusion_steps")}_tmp.jsonl', "w") as f:
        for json_obj in samples:
            f.write(json.dumps(json_obj) + "\n")
    return results["rouge1"], results["rouge2"], results["rougeL"]

def humaneval_infill(model, tokenizer):
    # not clear
    pass


def eval_triva(model, tokenizer, config, max_iter=np.inf):
    from datasets import load_dataset
    print("Evaluating TriviaQA...")
    start_time = time.time()

    device = get_device()
    # ds = load_dataset("mandarjoshi/trivia_qa", "rc", split='validation')
    ds = load_dataset("rajpurkar/squad", split="validation")
    gens = []
    refs = []
    total_cnt = 0
    cor = 0

    for doc in ds:
        total_cnt += 1
        query = f"{doc['context']}\nQuesion: {doc['question']}?\nAnswer: "
        labels = doc["answers"]["text"]
        encoded_labels = [
            tokenizer.encode(lab) for lab in labels
        ]  # , add_special_tokens=False)
        long_gold = max(encoded_labels, key=len)

        prompt_id = [50256] + tokenizer.encode(query)
        full = long_gold
        tokens = len(full)

        x0 = prompt_id + [0] * (tokens)
        # src_mask = torch.Tensor([[1]*len(input_ids)+[0]*(tokens)]).to('cuda')
        # args.diffusion_steps = tokens      COMMENTED FOR NOW, NOT SURE WHY IT IS THERE

        input_ids = torch.tensor(x0).unsqueeze(0).to(device)

        # si potrebbe usare anche generate_infilling con src_mask
        res = model.generate(
            input_ids,
            max_new_tokens=tokens,
            temperature=config.get("temperature"),
            top_k=config.get("top_k"),
            pipeline = config.get("pipeline"),
            diffusion_steps = tokens,    # as in diffugpt
            denoising_strategy = config.get("denoising_strategy")
        )[-1]
        pred = tokenizer.decode(res.tolist()[0][len(prompt_id) - 1 :])

        for lab in labels:
            if normalize_answer(lab) in normalize_answer(pred.strip()):
                cor += 1
                break

        gens.append(pred)
        refs.append(labels)
        print(f"step {total_cnt} done")
        if total_cnt >= max_iter:
            break
        if total_cnt == 2000:
            break

    print(gens)
    print(refs)
    ans = cor / total_cnt
    print("em acc:", ans)
    print(compute_f1(gens, refs))
    print(f"speed: {(total_cnt - 1) / (time.time() - start_time):.3f} step/sec")
    return ans

def print_output(x, tokenizer, mask_token):
    print("-" * 89)
    out = tokenizer.decode(x.tolist()[1:]).replace(mask_token, "<mask>")
    print(out)
    print("-" * 89)


def main():
    # From the command line we can specify the config.file
    parsers = argparse.ArgumentParser()
    parsers.add_argument("config", type=str, help="path/to/config.json")
    parsers.add_argument("-i", "--iters", type=int, default=float("inf"))
    parsers.add_argument("-t", "--test", nargs='+', type=str, required=True, help='Benchmarks to test the model on (all to test all).')
    args = parsers.parse_args()

    CONFIG_PATH = args.config
    EVALUATION_TYPE = args.test
    MAX_ITER = args.iters
    print(
        f"CONFIG_PATH = {CONFIG_PATH}\n"
        f"EVALUATION_TYPE = {EVALUATION_TYPE}\n"
        F"MAX_ITER = {MAX_ITER}"
    )
        
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    for i in EVALUATION_TYPE:
        if i == "all":
            # TODO: fix the rest
            EVALUATION_TYPE = ["lambada", "hellaswag", "wino", "piqa","siqa", "trivia"]
            break
        if i not in {
            "lambada",
            "hellaswag",
            "wino",
            "piqa",
            "siqa",
            "infilling",
            "trivia",
        }:
            print(f"Unknown evaluation type: {EVALUATION_TYPE}")
            print("Available evaluation types: lambada, hellaswag, wino, piqa, siqa, infilling, trivia")
            sys.exit(1)

    if config["pipeline"] != "diffusion":
        print("Evaluation implemented only for pipeline = diffusion.")
        sys.exit(1)
        
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

    # Run the required eval
    payload = {}
    for eval_type in EVALUATION_TYPE:
        if eval_type == "lambada":
            ans = eval_Lambada(model, tokenizer, config, MAX_ITER)
        elif eval_type == "hellaswag":
            ans = eval_hellaswag(model, tokenizer, config, MAX_ITER)
        elif eval_type == "wino":
            ans = eval_wino(model, tokenizer, config, MAX_ITER)
        elif eval_type == "piqa":
            ans = eval_piqa(model, tokenizer, config, MAX_ITER)
        elif eval_type == "siqa":
            ans = eval_siqa(model, tokenizer, config, MAX_ITER)
        elif eval_type == "infilling":
            ans = eval_infilling(model, tokenizer, config, MAX_ITER)
        elif eval_type == "trivia":
            ans = eval_triva(model, tokenizer, config, MAX_ITER)
        payload[eval_type] = ans
    
    # Save args
    payload["model"] = config["init_from"]
    payload["pipeline"] = config["pipeline"]
    payload["iters"] = MAX_ITER
    
    # When more than 1 evaluation save
    if len(EVALUATION_TYPE) > 1:
        with open(f"{int(time.time())}.json", "w") as f:
            json.dump(payload, f)

if __name__ == "__main__":
    main()
