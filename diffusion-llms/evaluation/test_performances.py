import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time
import argparse


def test_GPT2_inference(
    length , n_sequences , input_text, device
):

    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # nb, il modello deve essere compilato
    model = model.to(device)
    model.eval()

    # Tokenize input text
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True,
    ).to(device)

    # Warm-up (optional, to stabilize performance)
    print("Warming up...")
    for _ in range(5):
        with torch.no_grad():
            _ = model(**inputs)

    # Measure inference time
    print("Measuring inference performance...")
    start_time = time.time()
    with torch.no_grad():
        with torch.amp.autocast(device_type=device):
            # Generate text
            output = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=length,
                num_return_sequences=n_sequences,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                
                )
    end_time = time.time()
    print("\n")
    print(tokenizer.decode(output[0], skip_special_tokens=False))
    

    # Calculate and print results
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.6f} seconds")

    generated_tokens = (output.shape[-1] - inputs["input_ids"].shape[-1]) * n_sequences
    print(f"Generated {generated_tokens} tokens in {inference_time:.4f} seconds")
    print(f"Speed: {generated_tokens / (inference_time):.4f} tokens/sec")
    return inference_time



def test_DiffuGPT_inference(
    length , n_sequences , input_text, device
):
    pass


def test_our_inference(
    length , n_sequences , input_text, device
):
    pass


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test Model inference performance.")
    parser.add_argument("model_name", type=str, help="Name of the model to test. (gpt2 OR diffugpt OR our)")
    parser.add_argument("length", type=int, help="Length of the generated text.")
    parser.add_argument("n_sequences", type=int, help="Number of sequences to generate.")
    args = parser.parse_args()
    if args.model_name not in ["gpt2", "diffugpt", "our"]:
        raise ValueError("Invalid model name. Choose from 'gpt2', 'diffugpt', or 'our'.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_text = "Once upon a time, in a land far, far away,"

    if args.model_name == "gpt2":
        test_function = test_GPT2_inference
    elif args.model_name == "diffugpt":
        test_function = test_DiffuGPT_inference
    else:
        test_function = test_our_inference

        
    print(f"Testing model: {args.model_name} on device: {device}")
    test_function(
            length=args.length,
            n_sequences=args.n_sequences,
            input_text= input_text,
            device=device,)
    print("Test completed.")
