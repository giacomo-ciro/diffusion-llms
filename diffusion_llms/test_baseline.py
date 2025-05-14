from diffusion_llms.train_llada_pl import LladaBackbone
from diffusion_llms.dataloader.llada_dataloader import DataModule
from diffusion_llms.input_helper import get_config
from transformers import AutoTokenizer, AutoModel
import torch 
import numpy as np
import os
from tqdm import tqdm

@torch.no_grad()
def step_zero(
    model,
    masked_prompt: torch.Tensor,  # shape (B, L)
    masked_prompt: torch.Tensor,  # shape (B, L)
    *,
    mask_id: int = 126336,       # the id you use for [MASK]
    eos_token_id: int = 126081,       # model‑specific <eos>
    percentiles: list = [0.90],    # e.g. 0.90 → top‑10 % highest probs
    use_probs: bool = True       # set False if you really want raw logits
) -> torch.LongTensor:
    mask_id: int = 126336,        # the id you use for [MASK]
    eos_token_id: int = 12081,        # model-specific <eos>
    percentiles: list = [0.5, 0.75, 0.90],  # e.g. [50%, 75%, 90%]
    use_probs: bool = True        # set False if you really want raw logits
) -> list[list[torch.Tensor]]:
    """
    For each element in the batch, compute the quantile thresholds
    over the currently-masked positions for each requested percentile.
    
    For each element in the batch, compute the quantile thresholds
    over the currently-masked positions for each requested percentile.
    
    Returns
    -------
    thresholds : List of length B, each entry is a List of length len(percentiles)
                 containing the threshold tensor for that percentile.
    """
    # Get (B, L, V)
    logits = model(masked_prompt).logits
    thresholds : List of length B, each entry is a List of length len(percentiles)
                 containing the threshold tensor for that percentile.
    """
    # Get (B, L, V)
    logits = model(masked_prompt).logits
    eos_token_id = getattr(model.config, "eos_token_id", eos_token_id)

    # (B, L) scores for <eos>
    # (B, L) scores for <eos>
    if use_probs:
        eos_scores = torch.softmax(logits.float(), dim=-1)[..., eos_token_id]
    else:
        eos_scores = logits[..., eos_token_id].float()

    batch_size, seq_len = eos_scores.shape
    all_thresholds: list[list[torch.Tensor]] = []
    all_thresholds: list[list[torch.Tensor]] = []

    for b in range(batch_size):
        # which positions are still masked?
        mask_positions = masked_prompt[b] == mask_id

        # which positions are still masked?
        mask_positions = masked_prompt[b] == mask_id

        if not mask_positions.any():
            # no masks left → append zeros or some sentinel
            all_thresholds.append([torch.tensor(0.0, device=eos_scores.device)
                                   for _ in percentiles])
            # no masks left → append zeros or some sentinel
            all_thresholds.append([torch.tensor(0.0, device=eos_scores.device)
                                   for _ in percentiles])
            continue

        # scores only at masked positions
        # scores only at masked positions
        masked_scores = eos_scores[b][mask_positions]

        # for each percentile, compute and collect threshold
        thresholds_b: list[torch.Tensor] = []
        for p in percentiles:
            thresh = torch.quantile(masked_scores, p)
            thresholds_b.append(thresh)
        all_thresholds.append(thresholds_b)
        # for each percentile, compute and collect threshold
        thresholds_b: list[torch.Tensor] = []
        for p in percentiles:
            thresh = torch.quantile(masked_scores, p)
            thresholds_b.append(thresh)
        all_thresholds.append(thresholds_b)

    return all_thresholds
    return all_thresholds

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
    model = AutoModel.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    model.to("cuda")
    model.eval()
    
    positions = []
    for glob_idx, batch in enumerate(tqdm(data_module.test_dataloader(), desc="Batches")):
        input_ids = batch["input_ids"].to("cuda")  # shape (B, L)
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