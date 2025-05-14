#!/usr/bin/env python3
"""
This script tests the generation capabilities of the model on various benchmarks:
- BBH (Big-Bench Hard)
- GSM8K (Grade School Math)
- Minerva (Scientific tasks via MATH dataset)
- HumanEval (Code Generation)
- MBPP (Mostly Basic Python Problems)

Usage:
python eval_test.py --config path/to/config.json [--tasks task1 task2 ...] [--samples N] [--output results.json] [--debug]

Example:
python eval_test.py --config ../configs/eval_config.json --tasks bbh gsm8k --samples 10 --debug
"""

import sys
import os
import json
import argparse
import time
from tqdm import tqdm
from typing import List, Dict, Any, Union, Optional
import numpy as np

import torch
import tiktoken
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import model
try:
    from diffusion_llms.model import GPT2
except ImportError:
    try:
        from model import GPT2
    except ImportError:
        print("Error: Cannot import GPT2 model. Check your import paths.")
        sys.exit(1)

# Constants
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 40

def get_device():
    """Get the device to use for PyTorch operations (CPU or CUDA or MPS)."""
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

def load_model_and_tokenizer(config_path, checkpoint_path=None):
    """Load the model and tokenizer based on config file."""
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Loading model from configuration: {config_path}")
    
    # Get tiktoken encoder
    tokenizer_tiktoken = tiktoken.get_encoding("gpt2")
    
    # For HuggingFace interface
    tokenizer_hf = AutoTokenizer.from_pretrained("gpt2")
    tokenizer_hf.pad_token = tokenizer_hf.eos_token
    
    # Get device
    device = get_device()
    
    # Instantiate model
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = GPT2.load_from_checkpoint(checkpoint_path)
    elif os.path.exists(config["init_from"]):
        print(f"Loading pre-trained model from: {config['init_from']}")
        model = GPT2.from_pretrained(config["init_from"])
    else:
        print(f"Initializing model from config")
        model = GPT2(config_path)
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    return model, tokenizer_tiktoken, tokenizer_hf, config, device

def generate_text(model, tokenizer, prompt, config, max_new_tokens=None, temperature=None):
    """Generate text based on a prompt."""
    device = next(model.parameters()).device
    
    # Tokenize the prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
    
    # Set generation parameters
    max_new = max_new_tokens if max_new_tokens is not None else DEFAULT_MAX_NEW_TOKENS
    temp = temperature if temperature is not None else config.get("temperature", DEFAULT_TEMPERATURE)
    top_k = config.get("top_k", DEFAULT_TOP_K)
    pipeline = config.get("pipeline", "diffusion")
    
    # Additional parameters
    if pipeline == "diffusion":
        diffusion_steps = config.get("diffusion_steps", max_new)
        denoising_strategy = config.get("denoising_strategy", "entropy")
    else:  # arm pipeline
        do_sample = True
        repetition_penalty = 1.0
    
    # Generate text
    start_time = time.time()
    try:
        if pipeline == "diffusion":
            # Use the main generate method with diffusion pipeline
            outputs = model.generate(
                input_ids=input_ids,
                pipeline="diffusion",
                max_new_tokens=max_new,
                temperature=temp,
                top_k=top_k,
                denoising_strategy=denoising_strategy,
                diffusion_steps=diffusion_steps,
                var_len=False  # This parameter is required by the model
            )
            # Take the last output from diffusion steps
            output_ids = outputs[-1][0]
        else:  # arm pipeline
            # Use the main generate method with arm pipeline
            outputs = model.generate(
                input_ids=input_ids,
                pipeline="arm",
                max_new_tokens=max_new,
                temperature=temp,
                top_k=top_k,
                do_sample=True,
                repetition_penalty=1.0
            )
            output_ids = outputs[0][0]  # First batch, first sequence
            output_ids = outputs[0][0]
        
        # Decode the generated text
        full_text = tokenizer.decode(output_ids.tolist())
        
        # Extract only the newly generated text
        prompt_text = tokenizer.decode(input_ids[0].tolist())
        generated_text = full_text[len(prompt_text):]
        
        return {
            'prompt': prompt_text,
            'generated': generated_text,
            'full_text': full_text,
            'time_taken': time.time() - start_time
        }
    
    except Exception as e:
        print(f"Error during generation: {e}")
        return {
            'prompt': prompt,
            'generated': f"ERROR: {str(e)}",
            'full_text': prompt + f" ERROR: {str(e)}",
            'time_taken': time.time() - start_time
        }

def evaluate_bbh(model, tokenizer_tiktoken, config, num_samples=10, debug=False):
    """Evaluate on Big-Bench Hard tasks."""
    results = {}
    
    # List of BBH tasks to evaluate
    bbh_tasks = [
        "navigate", "causal_judgment", "date_understanding", "tracking_shuffled_objects",
        "disambiguation_qa", "geometric_shapes", "logical_deduction", "hyperbaton",
        "movie_recommendation", "formal_fallacies", "temporal_sequences", "sports_understanding",
        "reasoning_about_colored_objects", "penguins_in_a_table", "word_sorting",
        "web_of_lies", "ruin_names", "salient_translation_error_detection", "snarks"
    ]
    
    # Check if dataset can be accessed, otherwise skip evaluation
    try:
        _ = load_dataset("lukaemon/bbh", bbh_tasks[0], trust_remote_code=True, split="test")
    except Exception as e:
        print(f"Error: Cannot access BBH dataset: {e}")
        print("Skipping BBH evaluation")
        return {"error": f"Cannot access BBH dataset: {e}"}
    
    # Use subset for quicker test if debug mode
    if debug:
        bbh_tasks = bbh_tasks[:3]
        num_samples = min(num_samples, 3)
    
    overall_stats = {"correct": 0, "total": 0}
    
    for task in bbh_tasks:
        print(f"\nEvaluating on BBH task: {task}")
        
        try:
            # Load dataset
            ds = load_dataset("lukaemon/bbh", task, trust_remote_code=True)
            samples = ds["test"][:num_samples]
            
            correct = 0
            total = 0
            task_results = []
            
            for i, sample in enumerate(tqdm(samples, desc=f"Task: {task}")):
                input_text = sample["input"]
                target = sample["target"]
                
                # Generate answer
                result = generate_text(model, tokenizer_tiktoken, input_text, config)
                generated = result["generated"].strip()
                
                # Check if the answer is correct (exact match)
                # This is a simplistic evaluation - could be improved with more task-specific metrics
                is_correct = generated == target
                
                task_results.append({
                    "prompt": input_text,
                    "generated": generated,
                    "target": target,
                    "is_correct": is_correct
                })
                
                if is_correct:
                    correct += 1
                total += 1
                
                if debug:
                    print(f"\nExample {i+1}:")
                    print(f"Prompt: {input_text}")
                    print(f"Target: {target}")
                    print(f"Generated: {generated}")
                    print(f"Correct: {is_correct}")
            
            accuracy = correct / total if total > 0 else 0
            results[task] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "examples": task_results
            }
            
            print(f"Task {task} accuracy: {accuracy:.4f} ({correct}/{total})")
            
            # Update overall stats
            overall_stats["correct"] += correct
            overall_stats["total"] += total
            
        except Exception as e:
            print(f"Error evaluating task {task}: {e}")
            results[task] = {"error": str(e)}
    
    # Calculate overall accuracy
    overall_accuracy = overall_stats["correct"] / overall_stats["total"] if overall_stats["total"] > 0 else 0
    results["overall"] = {
        "accuracy": overall_accuracy,
        "correct": overall_stats["correct"],
        "total": overall_stats["total"]
    }
    
    print(f"\nOverall BBH accuracy: {overall_accuracy:.4f} ({overall_stats['correct']}/{overall_stats['total']})")
    return results

def evaluate_gsm8k(model, tokenizer_tiktoken, config, num_samples=10, debug=False):
    """Evaluate on GSM8K math problems."""
    print("\nEvaluating on GSM8K")
    
    # Check if dataset can be accessed
    try:
        # Load dataset
        ds = load_dataset("gsm8k", "main")
        # GSM8K dataset has integer indices, ensure we're using them properly
        test_indices = list(range(min(num_samples, len(ds["test"]))))
        samples = [ds["test"][i] for i in test_indices]
        
        correct = 0
        total = 0
        results = []
        
        for i, sample in enumerate(tqdm(samples, desc="GSM8K")):
            question = sample["question"]
            # Extract the correct answer from the answer field (which contains the reasoning + answer)
            reference_answer = sample["answer"].split("####")[-1].strip()
            
            # Generate answer
            prompt = f"Problem: {question}\nSolve this step-by-step:"
            result = generate_text(model, tokenizer_tiktoken, prompt, config, max_new_tokens=512)
            generated = result["generated"].strip()
            
            # Extract the predicted answer
            # This is a simple heuristic and might need to be improved for better extraction
            predicted_answer = None
            
            # Try to find the answer after "####" or "The answer is" patterns
            if "####" in generated:
                predicted_answer = generated.split("####")[-1].strip()
            elif "The answer is" in generated:
                predicted_answer = generated.split("The answer is")[-1].strip()
                predicted_answer = predicted_answer.split(".")[0].strip()
            else:
                # As a fallback, take the last sentence with a number
                sentences = generated.split(".")
                for sentence in reversed(sentences):
                    if any(c.isdigit() for c in sentence):
                        predicted_answer = sentence.strip()
                        break
            
            # Check if the predicted answer contains the reference answer
            # This is a simple evaluation - could be improved
            if predicted_answer:
                # Normalize answers to just compare the numbers
                ref_num = ''.join(c for c in reference_answer if c.isdigit() or c == '.')
                pred_num = ''.join(c for c in predicted_answer if c.isdigit() or c == '.')
                
                is_correct = ref_num in pred_num
            else:
                is_correct = False
            
            results.append({
                "question": question,
                "generated": generated,
                "reference_answer": reference_answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct
            })
            
            if is_correct:
                correct += 1
            total += 1
            
            if debug:
                print(f"\nExample {i+1}:")
                print(f"Question: {question}")
                print(f"Reference answer: {reference_answer}")
                print(f"Generated: {generated}")
                print(f"Predicted answer: {predicted_answer}")
                print(f"Correct: {is_correct}")
        
        accuracy = correct / total if total > 0 else 0
        print(f"GSM8K accuracy: {accuracy:.4f} ({correct}/{total})")
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "examples": results
        }
    
    except Exception as e:
        print(f"Error evaluating GSM8K: {e}")
        return {"error": str(e)}

def evaluate_minerva(model, tokenizer_tiktoken, config, num_samples=10, debug=False):
    """Evaluate on Minerva scientific tasks using MATH dataset as a proxy."""
    print("\nEvaluating on MATH dataset (proxy for Minerva)")
    
    # Check if dataset can be accessed
    try:
        # Load dataset
        ds = load_dataset("hendrycks/math")
        
        # If debug mode, use fewer categories
        categories = list(ds.keys()) if not debug else ["algebra"]
        
        # List to store results
        overall_stats = {"correct": 0, "total": 0}
        results = {}
        
        for category in categories:
            print(f"Evaluating category: {category}")
            samples = ds[category][:num_samples]
            
            correct = 0
            total = 0
            category_results = []
            
            for i, sample in enumerate(tqdm(samples, desc=f"Category: {category}")):
                problem = sample["problem"]
                solution = sample["solution"]
                
                # Generate answer
                prompt = f"Problem: {problem}\nSolve this step-by-step:"
                result = generate_text(model, tokenizer_tiktoken, prompt, config, max_new_tokens=512)
                generated = result["generated"].strip()
                
                # Very basic evaluation - check if the final answer is in the generated text
                # Extracting the correct answer from the solution is complex
                # For real evaluation, we would need a more sophisticated method
                
                # For now, let's check if any numbers in the solution appear in the generated text
                # Extract all numerical values from both texts
                def extract_numbers(text):
                    import re
                    return re.findall(r'\d+(?:\.\d+)?', text)
                
                solution_numbers = extract_numbers(solution)
                generated_numbers = extract_numbers(generated)
                
                # Check if the last number in the solution appears in the generated text
                is_correct = False
                if solution_numbers and generated_numbers:
                    last_solution_number = solution_numbers[-1]
                    is_correct = last_solution_number in generated_numbers
                
                category_results.append({
                    "problem": problem,
                    "solution": solution,
                    "generated": generated,
                    "is_correct": is_correct
                })
                
                if is_correct:
                    correct += 1
                total += 1
                
                if debug and i < 2:  # Show only first 2 examples in debug mode
                    print(f"\nExample {i+1}:")
                    print(f"Problem: {problem[:100]}... (truncated)")
                    print(f"Solution: {solution[:100]}... (truncated)")
                    print(f"Generated: {generated[:100]}... (truncated)")
                    print(f"Correct: {is_correct}")
            
            accuracy = correct / total if total > 0 else 0
            results[category] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "examples": category_results
            }
            
            print(f"Category {category} accuracy: {accuracy:.4f} ({correct}/{total})")
            
            # Update overall stats
            overall_stats["correct"] += correct
            overall_stats["total"] += total
        
        # Calculate overall accuracy
        overall_accuracy = overall_stats["correct"] / overall_stats["total"] if overall_stats["total"] > 0 else 0
        results["overall"] = {
            "accuracy": overall_accuracy,
            "correct": overall_stats["correct"],
            "total": overall_stats["total"]
        }
        
        print(f"\nOverall MATH accuracy: {overall_accuracy:.4f} ({overall_stats['correct']}/{overall_stats['total']})")
        return results
    
    except Exception as e:
        print(f"Error evaluating MATH dataset: {e}")
        return {"error": str(e)}

def evaluate_humaneval(model, tokenizer_tiktoken, tokenizer_hf, config, num_samples=None, debug=False):
    """Evaluate on HumanEval code generation tasks."""
    print("\nEvaluating on HumanEval")
    
    # Check if dataset and metrics can be accessed
    try:
        # Check if evaluate is available
        try:
            meteor = evaluate.load("meteor")
        except Exception as e:
            print(f"Warning: Cannot load METEOR metric: {e}")
            print("Will skip METEOR score calculation")
            meteor = None
            
        # Load dataset
        ds = load_dataset("openai_humaneval")
        samples = ds["test"]
        
        # Use all samples unless specified otherwise
        if num_samples is not None:
            samples = samples[:num_samples]
        
        results = []
        successful_executions = 0
        total = 0
        meteor_scores = []
        
        for i, sample in enumerate(tqdm(samples, desc="HumanEval")):
            task_id = sample["task_id"]
            prompt = sample["prompt"]
            canonical_solution = sample["canonical_solution"]
            
            # Generate code
            result = generate_text(model, tokenizer_tiktoken, prompt, config, max_new_tokens=512)
            generated = result["generated"].strip()
            
            # Extract the generated function
            # This is a simple approach and might need improvement
            # Ideally, we'd use Python's AST to properly extract the function
            
            # Check if the code executes without errors
            full_code = prompt + generated
            is_executable = False
            
            try:
                # Try to evaluate the code in a safe environment
                # Using exec here, but in production would use a proper sandbox
                namespace = {}
                exec(full_code, namespace)
                is_executable = True
                successful_executions += 1
            except Exception as e:
                if debug:
                    print(f"Execution error for {task_id}: {e}")
            
            # Calculate METEOR score if available (not ideal for code but gives some indication)
            meteor_score = 0.0
            if meteor:
                try:
                    references = [canonical_solution]
                    predictions = [generated]
                    
                    meteor_result = meteor.compute(predictions=predictions, references=references)
                    meteor_score = meteor_result["meteor"]
                    meteor_scores.append(meteor_score)
                except Exception as e:
                    if debug:
                        print(f"Error calculating METEOR for {task_id}: {e}")
            
            results.append({
                "task_id": task_id,
                "prompt": prompt,
                "generated": generated,
                "canonical_solution": canonical_solution,
                "is_executable": is_executable,
                "meteor_score": meteor_score
            })
            
            total += 1
            
            if debug and i < 3:  # Show only first 3 examples in debug mode
                print(f"\nExample {i+1} ({task_id}):")
                print(f"Prompt: {prompt[:100]}... (truncated)")
                print(f"Generated: {generated[:100]}... (truncated)")
                print(f"Executable: {is_executable}")
                print(f"METEOR score: {meteor_score:.4f}")
        
        # Calculate metrics
        pass_rate = successful_executions / total if total > 0 else 0
        avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
        
        print(f"HumanEval pass rate: {pass_rate:.4f} ({successful_executions}/{total})")
        print(f"Average METEOR score: {avg_meteor:.4f}")
        
        return {
            "pass_rate": pass_rate,
            "successful_executions": successful_executions,
            "total": total,
            "avg_meteor": avg_meteor,
            "examples": results
        }
    
    except Exception as e:
        print(f"Error evaluating HumanEval: {e}")
        return {"error": str(e)}

def evaluate_mbpp(model, tokenizer_tiktoken, config, num_samples=10, debug=False):
    """Evaluate on Mostly Basic Python Problems."""
    print("\nEvaluating on MBPP")
    
    # Check if dataset can be accessed
    try:
        # Load dataset
        ds = load_dataset("mbpp")
        samples = ds["test"][:num_samples]
        
        results = []
        successful_executions = 0
        total = 0
        
        for i, sample in enumerate(tqdm(samples, desc="MBPP")):
            task_id = sample["task_id"]
            text = sample["text"]  # The problem statement
            code = sample["code"]  # The reference solution
            test_list = sample["test_list"]  # Test cases
            
            # Generate code
            prompt = f"# Python function\n# {text}\n\ndef"
            result = generate_text(model, tokenizer_tiktoken, prompt, config, max_new_tokens=512)
            generated = "def" + result["generated"].strip()
            
            # Check if the code executes and passes test cases
            is_executable = False
            passes_tests = False
            
            try:
                # Create test environment
                namespace = {}
                
                # Execute the generated code
                exec(generated, namespace)
                is_executable = True
                
                # Execute test cases
                all_tests_pass = True
                for test_case in test_list:
                    try:
                        exec(test_case, namespace)
                    except AssertionError:
                        all_tests_pass = False
                        break
                    except Exception:
                        all_tests_pass = False
                        break
                
                passes_tests = all_tests_pass
                
                if passes_tests:
                    successful_executions += 1
            
            except Exception as e:
                if debug:
                    print(f"Execution error for task {task_id}: {e}")
            
            results.append({
                "task_id": task_id,
                "text": text,
                "generated": generated,
                "reference": code,
                "is_executable": is_executable,
                "passes_tests": passes_tests,
                "test_list": test_list
            })
            
            total += 1
            
            if debug and i < 3:  # Show only first 3 examples in debug mode
                print(f"\nExample {i+1} ({task_id}):")
                print(f"Problem: {text}")
                print(f"Generated: {generated[:150]}... (truncated)")
                print(f"Executable: {is_executable}")
                print(f"Passes tests: {passes_tests}")
        
        # Calculate metrics
        pass_rate = successful_executions / total if total > 0 else 0
        
        print(f"MBPP pass rate: {pass_rate:.4f} ({successful_executions}/{total})")
        
        return {
            "pass_rate": pass_rate,
            "successful_executions": successful_executions,
            "total": total,
            "examples": results
        }
    
    except Exception as e:
        print(f"Error evaluating MBPP: {e}")
        return {"error": str(e)}

def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate model on various benchmarks")
    parser.add_argument("--config", type=str, required=True, help="Path to model config file")
    parser.add_argument("--tasks", nargs="+", default=["bbh", "gsm8k", "minerva", "humaneval", "mbpp"],
                        help="List of tasks to evaluate on")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to evaluate per task")
    parser.add_argument("--output", type=str, default=None, help="Output file to save results")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--max-new-tokens", type=int, default=None, 
                        help="Maximum number of new tokens to generate (default: from config or 256)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Temperature for generation (default: from config or 0.7)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to specific model checkpoint (overrides config's init_from)")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer_tiktoken, tokenizer_hf, config, device = load_model_and_tokenizer(args.config, args.checkpoint)
    
    # Dictionary to store all results
    all_results = {}
    
    # Update config with command-line parameters if provided
    if args.max_new_tokens:
        config["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        config["temperature"] = args.temperature
    
    # Evaluate on selected tasks
    for task in args.tasks:
        task = task.lower()
        
        if task == "bbh":
            results = evaluate_bbh(model, tokenizer_tiktoken, config, args.samples, args.debug)
            all_results["bbh"] = results
        
        elif task == "gsm8k":
            results = evaluate_gsm8k(model, tokenizer_tiktoken, config, args.samples, args.debug)
            all_results["gsm8k"] = results
        
        elif task == "minerva":
            results = evaluate_minerva(model, tokenizer_tiktoken, config, args.samples, args.debug)
            all_results["minerva"] = results
        
        elif task == "humaneval":
            results = evaluate_humaneval(model, tokenizer_tiktoken, tokenizer_hf, config, args.samples, args.debug)
            all_results["humaneval"] = results
        
        elif task == "mbpp":
            results = evaluate_mbpp(model, tokenizer_tiktoken, config, args.samples, args.debug)
            all_results["mbpp"] = results
        
        else:
            print(f"Unknown task: {task}")
    
    # Save results if output file is specified
    if args.output:
        output_path = args.output
        # If no extension, add .json
        if not output_path.endswith('.json'):
            output_path += '.json'
        
        # Add timestamp to filename if it doesn't have a timestamp
        if not any(c.isdigit() for c in output_path):
            timestamp = int(time.time())
            output_path = output_path.replace('.json', f'_{timestamp}.json')
        
        with open(output_path, 'w') as f:
            json.dump({
                "results": all_results,
                "config": config,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "samples_per_task": args.samples
            }, f, indent=2)
        
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()