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

def csv_to_latex_table(input_file, output_file="eval.tex"):
    # Read the CSV file
    df = pd.read_csv(input_file, index_col=0)
    
    # Format the index (model names): replace underscores with spaces and capitalize
    df.index = df.index.map(lambda x: x.replace('_', ' ').title())
    
    # Format the LaTeX table with improved appearance
    latex_code = df.sort_index().to_latex(
        index=True,
        float_format=lambda x: f"{x:,.2f}",  # Format with comma for thousands
        caption="Comparison of Model Performance Metrics",
        label="tab:model_performance",
        position="htbp",
        escape=False,
        na_rep="-",
    )
    
    # Enhance the LaTeX table with better formatting
    latex_preamble = r"""\begin{table*}[htbp]
\centering
\caption{\textbf{Length Prediction Approaches.} Comparison of different prediction mechanisms used for length estimation and sequence termination.}
\label{tab:model_performance}
\begin{tabularx}{\linewidth}{l|YYYYYYY}
\toprule
"""
    
    # Create the header row with better column names with all titles in bold
    header = r"\textbf{Model} & \textbf{\% Below} & \textbf{\% Above} & \textbf{\% Exact} & \textbf{\# Tokens Below (avg.)} & \textbf{\# Tokens Above (Avg.)} & \textbf{MSE} \\"
    
    # Get the data rows from the original LaTeX output
    lines = latex_code.split('\n')
    data_rows = []
    capture = False
    
    for line in lines:
        if r'\midrule' in line:
            capture = True
            continue
        if r'\bottomrule' in line:
            capture = False
            continue
        if capture and line.strip():
            data_rows.append(line)
    
    # Assemble the enhanced LaTeX table
    enhanced_latex = (
        latex_preamble + 
        header + r" \midrule" + "\n" + 
        "\n".join(data_rows) + "\n" + 
        r"\bottomrule" + "\n" + 
        r"\end{tabularx}" + "\n" + 
        r"\end{table*}"
    )
    
    with open(output_file, 'w') as f:
        f.write(enhanced_latex)
    print(f"LaTeX table saved to {output_file}")

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
    gt = [
        len(tokenizer(response)["input_ids"])
        for response in df.model_response.dropna()
    ]
    n_test_samples = len(gt)
    print(f"Evaluating {n_test_samples} test instances...")

    # Get list of pred files
    # all the .npy's in this folder
    paths_to_pred = [fp for fp in os.listdir(".") if fp.endswith(".npy")]
    
    # Open the files and store in a dict
    preds = {p.split(".")[0]:np.load(p) for p in paths_to_pred}

    # Add baselines
    preds["random"] = np.random.randint(
        low = 0,
        high = np.max(gt),
        size=(len(gt),)
    )
    mean_ = np.mean(gt).round().item()
    preds[f"mean_{mean_}"] = np.full(
        shape=(len(gt),),
        fill_value=mean_
    )
    min_ = np.min(gt).item()
    preds[f"min_{min_}"] = np.full(
        shape=(len(gt),),
        fill_value=min_
    )
    max_ = np.max(gt).item()
    preds[f"max_{max_}"] = np.full(
        shape=(len(gt),),
        fill_value=max_
    )
    
    # Sanity check
    for model in preds.keys():
        print(model)
        assert preds[model].shape == (len(gt),)
    
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
            "Below_avg",
            "Above_avg"
        ]
    )

    for i, model in enumerate(preds.keys()):
        
        # The model name
        ans_df.loc[i, "Model"] = model
        
        # How many above, below, exact predicted (with avg)
        for k in ["above", "below", "exact"]:
            ans_df.loc[i, k.capitalize()] = len(ans[model][k]) / n_test_samples * 100
            if k != "exact":
                ans_df.loc[i, f"{k.capitalize()}_avg"] = np.mean(ans[model][k]).item()

        # Compute mse
        ans_df.loc[i, "MSE"] = np.mean(
            np.power(
                ans[model]["above"] + ans[model]["below"] + ans[model]["exact"],
                2
            )
        ).item()

    # Print the result
    print(ans_df.to_string())

    # Save it
    ans_df.to_csv("eval.csv", index=False)

    # Read the csv back and save to latex
    csv_to_latex_table("eval.csv")

if __name__=="__main__":
    main()
