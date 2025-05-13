import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
import time

def step_zero(model, masked_prompt, eos_token_id=2, percentile=0.9):
    """
    Initializes the generation process by using the model and the masked prompt to compute
    the expected length of the sentence at the first step.

    Returns:
        The position (index) where the eos probability first meets or exceeds the threshold for each batch element.
    """
    # Generate logits for the masked prompt
    logits = model(masked_prompt).logits  # shape: (batch, seq_len, vocab_size)

    # Get eos token id from model config if available, else use a default (e.g., 2 or 50256)
    eos_token_id = getattr(model.config, 'eos_token_id', eos_token_id)
    
    # Get probabilities for the eos token at each position
    eos_logits = logits[..., eos_token_id]  # shape: (batch, seq_len)

    # Calculate the threshold for the eos token probabilities
    threshold = torch.quantile(eos_logits, percentile, dsim=-1, keepdim=True)  # shape: (batch, 1)

    # Find the first position where eos_logits >= threshold for each batch
    meets_threshold = eos_logits >= threshold  # shape: (batch, seq_len)
    # Use argmax on the boolean mask; if never met, returns 0

    first_pos = meets_threshold.float().cumsum(dim=-1).eq(1).float().argmax(dim=-1)

    return first_pos


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate_step_zero_based(model, prompt, gen_length=128, temperature=0., cfg_scale=0.,
                              remasking='low_confidence', mask_id=126336, percentile=0.9, eos_token_id=2):
    """
    Generate using step-zero estimate to run the entire process in one-shot.
    """
    # Prepare input tensor
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)

    # Step zero to determine how many steps to run
    first_eos_pos = step_zero(model, x, eos_token_id=eos_token_id, percentile=percentile)
    steps = torch.clamp(first_eos_pos, min=1).item()

    for i in range(steps):
        mask_index = (x == mask_id)
        if cfg_scale > 0.:
            un_x = x.clone()
            un_x[prompt_index] = mask_id
            x_ = torch.cat([x, un_x], dim=0)
            logits = model(x_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x).logits

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)

        if remasking == 'low_confidence':
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        elif remasking == 'random':
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            raise NotImplementedError(remasking)

        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        for j in range(confidence.shape[0]):
            k = (mask_index[j].sum() // steps) + 1
            _, select_index = torch.topk(confidence[j], k=min(k, confidence.shape[1]))
            transfer_index[j, select_index] = True
        x[transfer_index] = x0[transfer_index]

    return x

@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, method='step_zero',
             percentile=0.9
             ):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    if method == 'step_zero':
        return generate_step_zero_based(model, prompt, gen_length=gen_length, temperature=temperature,
                                        cfg_scale=cfg_scale, remasking=remasking, mask_id=mask_id,
                                        percentile=percentile)

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def main():
    device = 'mps'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt_text = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt_text)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    # Generate with step_zero method
    start_time = time.time()
    out_step_zero = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence', method='step_zero')
    elapsed_step_zero = time.time() - start_time

    print("Step Zero Generation:")
    print(tokenizer.batch_decode(out_step_zero[:, input_ids.shape[1]:], skip_special_tokens=True)[0])
    print(f"Step Zero Generation took {elapsed_step_zero:.2f} seconds\n")

    # Generate with default method
    start_time = time.time()
    out_default = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence', method='default')
    elapsed_default = time.time() - start_time

    print("Default Generation:")
    print(tokenizer.batch_decode(out_default[:, input_ids.shape[1]:], skip_special_tokens=True)[0])
    print(f"Default Generation took {elapsed_default:.2f} seconds")


if __name__ == '__main__':
    main()
