from diffusion_llms.train_llada_pl import LladaBackbone
from diffusion_llms.dataloader.llada_dataloader import DataModule
from diffusion_llms.input_helper import get_config
from transformers import AutoTokenizer
import torch 
import numpy as np
import os
@torch.no_grad()
def step_zero(
    model,
    masked_prompt: torch.Tensor, # shape (B, L)
    *,
    mask_id: int = 126336,       # the id you use for [MASK]
    eos_token_id: int = 2,       # model‑specific <eos>
    percentiles: list = [0.90],    # e.g. 0.90 → top‑10 % highest probs
    use_probs: bool = True       # set False if you really want raw logits
) -> torch.LongTensor:
    """
    Predict the first position in *each* sequence where the model already thinks
    an <eos> is highly likely.

    The percentile is computed **only over currently‑masked positions** so the
    prompt tokens cannot pollute the statistic.

    Returns
    -------
    first_pos : LongTensor  shape (batch,)
        One index per batch element.  If no position reaches the threshold the
        function returns the *first* still‑masked position so generation will
        at least start there instead of at 0.
    """
    logits = model(masked_prompt).logits              # (B, L, V)
    eos_token_id = getattr(model.config, "eos_token_id", eos_token_id)

    # (B, L) log‑probabilities (or logits) of generating <eos>
    if use_probs:
        eos_scores = torch.softmax(logits.float(), dim=-1)[..., eos_token_id]
    else:
        eos_scores = logits[..., eos_token_id].float()

    batch_size, seq_len = eos_scores.shape
    first_pos = torch.zeros(batch_size, dtype=torch.long, device=masked_prompt.device)

    thresholds = []
    for b in range(batch_size):
        mask_positions = (masked_prompt[b] == mask_id)     # which tokens are still masked?
        if not mask_positions.any():
            # nothing left to predict → return last token
            first_pos[b] = seq_len - 1
            continue

        masked_scores = eos_scores[b][mask_positions]

        # score threshold for this sample
        thresh = torch.quantile(masked_scores, percentiles)

        thresholds.append(thresh)

    return thresholds

def main():
    args = get_config()    
    print("Configuration loaded successfully.")
    print("args:", args)
    # Create config for data module
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")
    # Create data module
    data_module = DataModule(
        args, 
        tokenizer=tokenizer,
        num_workers=args["num_workers"]
    )
    data_module.setup()
    model = LladaBackbone()
    positions = []
    for glob_idx, batch in enumerate(data_module.test_dataloader()):
        input_ids = batch["input_ids"]
        thresholds = step_zero(model, input_ids, eos_token_id=tokenizer.eos_token_id,
                                percentiles=[0.25, 0.50, 0.75]) #dict
        positions.extend(thresholds)  # list of dict

        if glob_idx % 50 == 0:
            print(f"Processed {glob_idx} batches, current positions: {positions}")
            
    # Save positions to file in output_dir using numpy
    positions = np.array(positions)
    output_path = os.path.join(args["output_dir"], "zero_shot.npy")
    np.save(output_path, positions)
            
        
        

if __name__ == "__main__":
    main()