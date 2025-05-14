"""
Reads the predictions stored in this folder and generates the result table.

Prediction are named:

logit_25
logit_50
logit_75
regression_avg
regression_concat
classification_token
bert_class
bert_regr

.npy

Each file contains the predictions done by each model, in the form
of a numpy array where each entry corresponds to the predicted length
of the test sentence.

The ground truth are obtained by directly reading the test.csv file and measuring the length with LLaDa tokenizer.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

def main():

    # dir1/dir2/this_file.py
    # I want to get
    # dir1/data/test.csv
    path_to_test = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "test.csv"
    )

    # Read in the prompts
    df = pd.read_csv(path_to_test)
    
    # Init the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct"
    )

    # Get GT lengths
    # for response in df.model_response:
    #     print(type(response))
    #     print(tokenizer(response))
    #     exit()
    gt = [
        len(tokenizer(response)["input_ids"])
        for response in df.model_response.dropna()
    ]
    print(f"Evaluating {len(gt)} test instances...")

    # Get list of pred files
    # all the .npy's in this folder
    paths_to_pred = [fp for fp in os.listdir(".") if fp.endswith(".npy")]
    
    # Open the files and store in a dict
    preds = {p.split(".")[0]:np.load(p) for p in paths_to_pred}
    
    # Sanity check
    for model in preds.keys():
        assert len(preds[model]) == len(gt)
    
    # Store results
    ans = {
        p:{
            "above":[],
            "below":[],
            "exact":[]
        } for p in preds.keys()
    }

    # Loop through test instances
    for i in tqdm(range(len(gt))):
        
        # Get true
        true = gt[i]

        # Evaluate each pred
        for model in preds.keys():
            
            # Pred for this model
            pred = preds[model][i]
            
            # Model predict exact length
            if pred == true:
                # print(f"Wow! {model} predicted the exact length ({pred}) of the {i}-th test instance!")
                k = "exact"
            # Model predicted more than true length (good)
            elif pred > true:
                k = "above"
            # Model predicted less than true length (bad)
            elif pred < true:
                k = "below"

            # Store the tuple in the correct place
            ans[model][k].append(
                    (pred-true).item()
                )
    
    # Get aggregate metrics
    ans_df = pd.DataFrame(
        columns=[
            "Model",
            "Below",
            "Above",
            "Exact",
            "Below_avg"
            "Above_avg"
        ]
    )

    for i, model in enumerate(preds.keys()):
        
        # The model name
        ans_df.loc[i, "Model"] = model
        
        # How many above, below, exact predicted (with avg)
        for k in ["above", "below", "exact"]:
            ans_df.loc[i, k.capitalize()] = len(ans[model][k]) 
            if k != "exact":
                ans_df.loc[i, f"{k.capitalize()}_avg"] = np.mean(ans[model][k]).item()

    # Print the result
    print(ans_df.to_string())

    # Save it
    ans_df.to_csv("eval.csv")

if __name__=="__main__":
    main()
