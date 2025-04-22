import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time
import argparse

def test_GPT2_inference(length = 50, n_sequences = 1, input_text="The quick brown fox jumps over the lazy dog."):
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = "cuda"
    else:
        raise RuntimeError("CUDA is not available. Please use a device='cpu' or ensure CUDA is properly installed.")
        

    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    device = torch.device(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # nb, il modello deve essere compilato
    model = model.to(device)
    model.eval()

    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    print(inputs["input_ids"].shape)

    # Warm-up (optional, to stabilize performance)
    print("Warming up...")
    for _ in range(5):
        with torch.no_grad():
            _ = model(**inputs)

    # Measure inference time
    print("Measuring inference performance...")
    start_time = time.time()
    with torch.no_grad():
        output = model.generate(inputs["input_ids"], max_length=length, num_return_sequences=1)
        # output = model.generate(
        #     input_ids= inputs["input_ids"],              # L'input
        #     max_length = length,                    # La lunghezza massima della sequenza
        #     num_return_sequences= n_sequences,           # Il numero di sequenze da generare
        #     no_repeat_ngram_size=2,           # Evita ripetizioni di n-grammi
        #     top_p=0.9,                        # Nucleus sampling: prendi il top-p% più probabili
        #     top_k=50,                         # Top-k sampling: prendi il top-k più probabili
        #     temperature=0.7,                  # Regola la "creatività" del modello
        #     do_sample=True,                   # Abilita il campionamento
        # )
        print(tokenizer.decode(output[0], skip_special_tokens=False))
    end_time = time.time()

    # Calculate and print results
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.6f} seconds")
    return inference_time

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Model Name (gpt2 OR diffugpt OR our).")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model Name (gpt2 OR diffugpt OR our).")
    args = parser.parse_args()

    # Example usage
    model_name = args.model_name
    if model_name not in ["gpt2", "diffugpt", "our"]:
        raise ValueError("Invalid model name. Choose from 'gpt2', 'diffugpt', or 'our'.")
    if model_name == "gpt2":
        pass
    elif model_name == "diffugpt":
        assert(False) # diffugpt model is not implemented yet.
    elif model_name == "our":
        assert(False) # our model is not implemented yet.
    input_text = "Once upon a time, in a land far, far away,"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Testing model: {model_name} on device: {device}")
    test_GPT2_inference(length=50, n_sequences=1, input_text="Once upon a time, in a land far, far away,")
