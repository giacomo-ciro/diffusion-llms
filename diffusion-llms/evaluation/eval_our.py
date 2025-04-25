import sys
import json

import torch
import tiktoken
from model import GPT2
from f1 import compute_f1, normalize_answer


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



import re
import numpy as np

def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def eval_hellaswag(model, tokenizer, max_iter = np.inf):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split='validation')

    total_cnt = 0
    cor = 0

    for doc in ds:
        total_cnt += 1
        if total_cnt % 1000 == 0:
            print('total cnt:', total_cnt)
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()

        query = preprocess(doc["activity_label"] + ": " + ctx)
        # print("query: ", query) 
        choices = [preprocess(ending) for ending in doc["endings"]]
        # print("choices: ", choices)
        gold = int(doc["label"])
        # print(gold)

        score_list = []
        prefix = [50256] + tokenizer.encode(query)

        for choice in choices:

            x0 = prefix + tokenizer.encode(choice)
            input_ids = torch.tensor(
                x0).unsqueeze(0).to('cuda')          # x0 is all the line
            src_mask = torch.Tensor([[1]*len(prefix)+[0]*(len(x0)-len(prefix))]).to('cuda')
            score = model.eval_forward(input_ids, src_mask)
            # import pdb; pdb.set_trace();
            score_list.append(score)
        pred = np.argmax(np.array(score_list))

        if pred == gold:
            cor += 1
        print(f"step {total_cnt} done. Accuracy: {cor/total_cnt:.4f}")
        if total_cnt >= max_iter:
            break

        
    print('acc:', cor/total_cnt)  



def eval_wino(model, tokenizer, max_iter = np.inf):
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split='validation', trust_remote_code=True)

    total_cnt = 0
    cor = 0

    for doc in ds:
        total_cnt += 1
        
        idx = doc["sentence"].index("_")
        
        options = [doc["option1"], doc["option2"]]

        answer_to_num = {"1": 0, "2": 1}
        gold = answer_to_num[doc["answer"]]

        score_list = []
        # print("options: ", options)
        # print("suffix: ", doc["sentence"][idx+1:].strip())
        # print("prefix: ", doc["sentence"][:idx])
        
        for opt in options:
            target_str = opt 
            suffix_str = doc["sentence"][idx+1:].strip()
            target_id = tokenizer.encode(target_str) # , add_special_tokens=False)
            suffix_id = tokenizer.encode(suffix_str)    #, add_special_tokens=False)
            prefix_str = doc["sentence"][:idx]
            prefix_id = [50256] + tokenizer.encode(prefix_str) #, add_special_tokens=False)

            x0 = prefix_id + target_id + suffix_id
            # src_mask = torch.Tensor([[1]*len(prefix_id)+[0]*(len(x0)-len(prefix_id))]).to("cuda")      # da capire perchè loro lo fanno così
            src_mask = torch.Tensor([[1]*len(prefix_id)+[0]*len(target_id) + [1]* len(suffix_id)]).to("cuda")      # da capire perchè lo fanno così

            input_ids = torch.tensor(
                x0).unsqueeze(0).to('cuda')          # x0 is all the line
            score = model.eval_forward(input_ids, src_mask)
            # import pdb; pdb.set_trace();
            score_list.append(score)
        pred = np.argmax(np.array(score_list))

        if pred == gold:
            cor += 1
        print(f"step {total_cnt} done. Accuracy: {cor/total_cnt:.4f}")

        if total_cnt >= max_iter:
            break
        
    print('acc:', cor/total_cnt)  


def eval_piqa(model, tokenizer, max_iter = np.inf):
    from datasets import load_dataset
    ds = load_dataset("ybisk/piqa", split='validation', trust_remote_code=True)
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
            src_mask = torch.Tensor([[1]*len(prefix)+[0]*(len(x0)-len(prefix))]).to('cuda')
            input_ids = torch.tensor(
                x0).unsqueeze(0).to('cuda')
            score = model.eval_forward(input_ids, src_mask)
            # import pdb; pdb.set_trace();
            score_list.append(score)
        pred = np.argmax(np.array(score_list))

        if pred == gold:
            cor += 1
        # print(total_cnt, cor/total_cnt)
        print(f"step {total_cnt} done. Accuracy: {cor/total_cnt:.4f}")
        if total_cnt >= max_iter:
            break
        
    print('acc:', cor/total_cnt)  



def eval_siqa(model, tokenizer, max_iter = np.inf):
    from datasets import load_dataset
    ds = load_dataset("allenai/social_i_qa", split='validation', trust_remote_code=True)
    total_cnt = 0
    cor = 0

    for doc in ds:
        total_cnt += 1
        
        query = f"Question: {doc['context']} {doc['question']}\nAnswer: "
        choices = [doc['answerA'], doc['answerB'], doc['answerC']]
        gold = int(doc["label"]) - 1

        score_list = []
        prefix = [50256] + tokenizer.encode(query)

        for choice in choices:

            x0 = prefix + tokenizer.encode(" " + choice)
            src_mask = torch.Tensor([[1]*len(prefix)+[0]*(len(x0)-len(prefix))]).to('cuda')
            input_ids = torch.tensor(
                x0).unsqueeze(0).to('cuda')
            score = model.eval_forward(input_ids, src_mask)
            # import pdb; pdb.set_trace();
            score_list.append(score)
        pred = np.argmax(np.array(score_list))

        if pred == gold:
            cor += 1
        # print(total_cnt, cor/total_cnt)
        print(f"step {total_cnt} done. Accuracy: {cor/total_cnt:.4f}")
        if total_cnt >= max_iter:
            break
        
    print('acc:', cor/total_cnt)  





import csv, json
import evaluate

def eval_infilling(model, tokenizer, config, max_iter = np.inf):
    """It does infilling on the true sentence in the Story Cloze Test, generating the middle sentence given the first
     and last sentences and computes the rouge score between the generated and the true middle sentence."""
    assert config.get("pipeline") == "diffusion"
    problems = []
    with open(f"evaluation/cloze_test_val__spring2016.csv") as f:
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
        # import pdb; pdb.set_trace();
        prompt = stories[0] + " " + stories[1]
        suffix = stories[3] + " " + stories[4]
        middle = stories[2]
        
        prefix = [50256] + tokenizer.encode(prompt) #, add_special_tokens=False)
        suff = tokenizer.encode(suffix) #, add_special_tokens=False)
        x0 = prefix + tokenizer.encode(middle) + suff       #, add_special_tokens=False)
        src_mask = torch.Tensor([[1]*len(prefix)+[0]*(len(x0)-len(prefix)-len(suff))+[1]*len(suff)]).to('cuda')   
        input_ids = torch.tensor(
            x0).unsqueeze(0).to('cuda')         
        res = model.generate_infilling(
            input_ids,src_mask, temperature=config.get("temperature"), top_k=config.get("top_k"))
        pred = tokenizer.decode(res.tolist()[0][len(prefix)-1:len(x0)-len(suff)-1])
    
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
    results["rougeAvg"] = (results["rouge1"] + results["rouge2"] + results["rougeL"]) / 3
    print(f"rouge1={results['rouge1']:.2f}, rouge2={results['rouge2']:.2f}, rougeL={results['rougeL']:.2f}, rougeAvg={results['rougeAvg']:.2f}")


    with open(f'ROCInfill_medium_t{config.get("diffusion_steps")}_tmp.jsonl', 'w') as f:
        for json_obj in samples:
            f.write(json.dumps(json_obj) + '\n')



def humaneval_infill(model, tokenizer):
    # non ho idea di cosa faccia
    pass


def eval_triva(model, tokenizer, config, max_iter = np.inf):
    from datasets import load_dataset
    # ds = load_dataset("mandarjoshi/trivia_qa", "rc", split='validation')
    ds = load_dataset("rajpurkar/squad", split='validation')
    gens = []
    refs = []
    total_cnt = 0
    cor = 0

    for doc in ds:
        total_cnt += 1
        # import pdb; pdb.set_trace();
        query = f"{doc['context']}\nQuesion: {doc['question']}?\nAnswer: "
        labels = doc["answers"]["text"]
        encoded_labels = [tokenizer.encode(l) for l in labels]      #, add_special_tokens=False)
        long_gold = max(encoded_labels, key=len)

        prompt_id = [50256] + tokenizer.encode(query)
        full = long_gold
        tokens = len(full)
        
        x0 = prompt_id + [0]*(tokens)
        # src_mask = torch.Tensor([[1]*len(input_ids)+[0]*(tokens)]).to('cuda')
        #args.diffusion_steps = tokens      COMMENTED FOR NOW, NOT SURE WHY IT IS THERE

        input_ids = torch.tensor(
            x0).unsqueeze(0).to('cuda')   

        # si potrebbe usare anche generate_infilling con src_mask    
        res = model.generate(input_ids, max_new_tokens=tokens, temperature=config.get("temperature"), top_k=config.get("top_k"))[-1]
        pred = tokenizer.decode(res.tolist()[0][len(prompt_id)-1:])
        
        for l in labels:
            if normalize_answer(l) in normalize_answer(pred.strip()):
                cor += 1
                break
                
        gens.append(pred)
        refs.append(labels)


        if total_cnt >= max_iter:
            break
        if total_cnt == 2000:
            break

        

    print(gens)
    print(refs)
    print('em acc:', cor/total_cnt)
    print(compute_f1(gens, refs))    


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

    # # Mask token
    # mask_token = tokenizer.decode([config["mask_id"]])

    # # Get prompt
    # input_ids = torch.tensor(
    #     [50256] + tokenizer.encode(config["user_prompt"])
    # ).unsqueeze(0)

    # Load model
    model = GPT2(CONFIG_PATH)
    model = torch.compile(model).to('cuda')

        
    # # List of tensors of shape (B, seq_len)
    # xs = model.generate(
    #     input_ids, 
    #     max_new_tokens=config.get("max_new_tokens", 128),
    #     temperature=config.get("temperature"),
    #     top_k=config.get("top_k"),
    # )

    # # Illustrate the diffusion process
    # print_output(xs[-1][0], tokenizer, mask_token)

################## TO BE UNCOMMENTED ##################
    # eval_Lambada(model, tokenizer, config)
    # eval_hellaswag(model, tokenizer, 1)
    # eval_wino(model, tokenizer, 100)
    # eval_piqa(model, tokenizer,100)
    # eval_siqa(model, tokenizer,100)
    # eval_infilling(model, tokenizer, config)
    # eval_triva(model, tokenizer, config, 100)



if __name__ == "__main__":
    main()
