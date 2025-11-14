#!/usr/bin/env python3
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Quality test script for HuggingFace models using AutoRound evaluation.

This script evaluates quantized or dequantized models on various tasks
using the AutoRound evaluation framework.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round.eval.evaluation import simple_evaluate_user_model

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def disable_chat_template(tokenizer):
    """Disable chat template for evaluation (use raw text, not chat format)."""
    tokenizer.chat_template = None
    
    # Also disable apply_chat_template if it exists
    if hasattr(tokenizer, 'apply_chat_template'):
        original_apply_chat_template = tokenizer.apply_chat_template
        
        def no_template(messages, **kwargs):
            # Return the last user message as plain text
            if isinstance(messages, list) and len(messages) > 0:
                last_msg = messages[-1]
                if isinstance(last_msg, dict) and 'content' in last_msg:
                    return last_msg['content']
            return str(messages)
        
        tokenizer.apply_chat_template = no_template


def load_model(model_path, device_map="cpu", dtype="auto", trust_remote_code=True):
    """Load a model from the specified path."""
    print(f"Loading model from {model_path}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    
    # Move model to MPS for faster inference (if available and not using device_map)
    if device_map == "cpu" and torch.backends.mps.is_available():
        print("Moving model to MPS for faster inference...")
        model = model.to("mps")
    
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    return model


def load_tokenizer(model_path, trust_remote_code=True):
    """Load tokenizer from the specified path."""
    print(f"Loading tokenizer from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    
    # Disable chat template for evaluation
    disable_chat_template(tokenizer)
    print("Chat template disabled for evaluation")
    
    return tokenizer


def evaluate_model(model, tokenizer, tasks, batch_size=8, max_batch_size=32, 
                   limit=None, eval_model_dtype="auto", device="cpu"):
    """Evaluate the model on specified tasks."""
    # Convert comma-separated string to list if needed
    if isinstance(tasks, str):
        tasks_list = [task.strip() for task in tasks.split(",") if task.strip()]
    else:
        tasks_list = tasks
    
    print(f"\nRunning evaluation on tasks: {', '.join(tasks_list)}")
    print("=" * 60)
    
    current_batch_size = batch_size
    current_max_batch_size = max_batch_size
    
    # Try evaluation, reduce batch size or fall back to CPU if MPS fails
    max_retries = 3
    for attempt in range(max_retries):
        try:
            results = simple_evaluate_user_model(
                model,
                tokenizer=tokenizer,
                batch_size=current_batch_size,
                tasks=tasks_list,  # Pass as list, not string
                limit=limit,
                max_batch_size=current_max_batch_size,
                eval_model_dtype=eval_model_dtype,
            )
            return results
        except RuntimeError as e:
            error_msg = str(e)
            # Check if it's an MPS-related error
            if "MPS" in error_msg or "mps" in error_msg.lower() or "INT_MAX" in error_msg:
                if attempt < max_retries - 1:
                    # Try reducing batch size first
                    if current_batch_size > 1:
                        current_batch_size = max(current_batch_size // 2, 1)
                        current_max_batch_size = max(current_max_batch_size // 2, 1)
                        print(f"\n⚠ MPS Error detected: {error_msg}")
                        print(f"Reducing batch size to {current_batch_size} (max: {current_max_batch_size}) and retrying...")
                        continue
                    else:
                        # Batch size already at minimum, fall back to CPU
                        print(f"\n⚠ MPS Error detected: {error_msg}")
                        print("Batch size already at minimum. Falling back to CPU evaluation...")
                        
                        # Move model to CPU
                        if next(model.parameters()).device.type == "mps":
                            print("Moving model from MPS to CPU...")
                            model = model.to("cpu")
                        
                        # Reset batch sizes for CPU
                        current_batch_size = batch_size
                        current_max_batch_size = max_batch_size
                        
                        # Retry with CPU
                        results = simple_evaluate_user_model(
                            model,
                            tokenizer=tokenizer,
                            batch_size=current_batch_size,
                            tasks=tasks_list,  # Pass as list, not string
                            limit=limit,
                            max_batch_size=current_max_batch_size,
                            eval_model_dtype=eval_model_dtype,
                        )
                        return results
                else:
                    # Last attempt failed, fall back to CPU
                    print(f"\n⚠ MPS Error detected after {max_retries} attempts: {error_msg}")
                    print("Falling back to CPU evaluation...")
                    
                    # Move model to CPU
                    if next(model.parameters()).device.type == "mps":
                        print("Moving model from MPS to CPU...")
                        model = model.to("cpu")
                    
                    # Reset batch sizes for CPU
                    current_batch_size = batch_size
                    current_max_batch_size = max_batch_size
                    
                    # Retry with CPU
                    results = simple_evaluate_user_model(
                        model,
                        tokenizer=tokenizer,
                        batch_size=current_batch_size,
                        tasks=tasks_list,  # Pass as list, not string
                        limit=limit,
                        max_batch_size=current_max_batch_size,
                        eval_model_dtype=eval_model_dtype,
                    )
                    return results
            else:
                # Re-raise if it's not an MPS error
                raise
    
    # Should not reach here, but just in case
    raise RuntimeError("Failed to evaluate model after multiple attempts")


def print_results(results, tasks):
    """Print evaluation results in a formatted way."""
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    # Parse task list
    task_list = [task.strip() for task in tasks.split(",")]
    
    # Store results for summary table
    summary_data = []
    
    if "results" in results:
        # Print detailed results for each task
        for task in task_list:
            if task in results["results"]:
                task_results = results["results"][task]
                print(f"\n{task.upper()} Results:")
                print("-" * 60)
                
                # Try to find accuracy metric
                accuracy = None
                accuracy_key = None
                for key in task_results:
                    if "acc" in key.lower():
                        accuracy = task_results[key]
                        accuracy_key = key
                        print(f"  Accuracy ({accuracy_key}): {accuracy}")
                        break
                
                # Print all metrics
                print(f"  All metrics:")
                for key, value in task_results.items():
                    print(f"    {key}: {value}")
                
                # Store for summary
                summary_data.append({
                    "task": task.upper(),
                    "accuracy": accuracy if accuracy is not None else "N/A",
                    "accuracy_key": accuracy_key if accuracy_key else "N/A",
                    "all_metrics": task_results
                })
            else:
                print(f"\n{task.upper()}: No results found")
                summary_data.append({
                    "task": task.upper(),
                    "accuracy": "No results",
                    "accuracy_key": "N/A",
                    "all_metrics": {}
                })
    else:
        print("Results:", results)
        return
    
    # Print summary table at the end
    print("\n" + "=" * 60)
    print("Summary of All Results")
    print("=" * 60)
    
    if summary_data:
        # Print table header
        print(f"\n{'Task':<20} {'Accuracy':<15} {'Metric Key':<20}")
        print("-" * 60)
        
        # Print each task result
        for data in summary_data:
            accuracy_str = str(data["accuracy"])
            if isinstance(data["accuracy"], (int, float)):
                accuracy_str = f"{data['accuracy']:.4f}"
            
            print(f"{data['task']:<20} {accuracy_str:<15} {data['accuracy_key']:<20}")


def main():
    """Main function to parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Quality test script for HuggingFace models using AutoRound evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on boolq task
  python eval_hf_model.py --model /path/to/model --tasks boolq
  
  # Evaluate on multiple tasks
  python eval_hf_model.py --model /path/to/model --tasks boolq,hellaswag,piqa
  
  # Use MPS device with custom batch size
  python eval_hf_model.py --model /path/to/model --tasks boolq --device mps --batch_size 16
  
  # Limit number of examples
  python eval_hf_model.py --model /path/to/model --tasks boolq --limit 100
        """,
    )
    
    parser.add_argument(
        "--model",
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory or HuggingFace model identifier",
    )
    
    parser.add_argument(
        "--tasks",
        "--task",
        type=str,
        default="boolq",
        help="Comma-separated list of evaluation tasks (default: boolq). "
             "Examples: boolq, hellaswag, piqa, mmlu",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Device to use for evaluation (default: cpu). "
             "Note: MPS may have compatibility issues (e.g., 'MPSGaph does not support tensor dims larger than INT_MAX'). "
             "If MPS fails, the script will automatically reduce batch size or fall back to CPU. "
             "For MPS, consider using smaller batch sizes (e.g., --batch_size 4).",
    )
    
    parser.add_argument(
        "--device_map",
        type=str,
        default="cpu",
        help="Device map for model loading (default: cpu). "
             "Options: cpu, auto, mps:0, cuda:0, etc.",
    )
    
    parser.add_argument(
        "--dtype",
        "--torch_dtype",
        type=str,
        default="auto",
        help="Model dtype (default: auto). Options: auto, float16, bfloat16, float32",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8)",
    )
    
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=32,
        help="Maximum batch size for evaluation (default: 32)",
    )
    
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. "
             "Integer: exact number (e.g., 100). "
             "Float 0-1: fraction of total examples (e.g., 0.1 for 10%%)",
    )
    
    parser.add_argument(
        "--eval_model_dtype",
        type=str,
        default="auto",
        help="Model dtype for evaluation (default: auto)",
    )
    
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading model",
    )
    
    parser.add_argument(
        "--no_trust_remote_code",
        action="store_true",
        help="Do not trust remote code (default behavior)",
    )
    
    args = parser.parse_args()
    
    # Determine trust_remote_code
    trust_remote_code = args.trust_remote_code and not args.no_trust_remote_code
    
    # Determine device_map
    device_map = args.device_map
    if device_map == "cpu" and args.device == "mps" and torch.backends.mps.is_available():
        # Will move to MPS after loading
        pass
    elif args.device == "mps":
        device_map = "cpu"  # Load to CPU first, then move to MPS
    
    # Load model
    model = load_model(
        args.model,
        device_map=device_map,
        dtype=args.dtype,
        trust_remote_code=trust_remote_code,
    )
    
    # Load tokenizer
    tokenizer = load_tokenizer(
        args.model,
        trust_remote_code=trust_remote_code,
    )
    
    # Run evaluation
    results = evaluate_model(
        model,
        tokenizer,
        args.tasks,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        limit=args.limit,
        eval_model_dtype=args.eval_model_dtype,
        device=args.device,
    )
    
    # Print results
    print_results(results, args.tasks)
    
    # Print comparison notes for known tasks
    task_list = [task.strip().lower() for task in args.tasks.split(",")]
    
    if "boolq" in task_list:
        print("\n" + "=" * 60)
        print("Comparison Notes (BoolQ):")
        print("=" * 60)
        if "results" in results and "boolq" in results["results"]:
            accuracy = None
            for key in results["results"]["boolq"]:
                if "acc" in key.lower():
                    accuracy = results["results"]["boolq"][key]
                    break
            
            print(f"- Current model accuracy: {accuracy}")
            print("- Original Qwen3-0.6B FP16:            ~62-65% (expected)")
            print("- Quantized GPTQ on CUDA:               59.14%")
            if accuracy:
                print(f"- Current model:                      {accuracy}")
            print("\nExpected: Dequantized accuracy should match quantized (59.14%)")
            print("Degradation from quantization: ~3-6 percentage points is normal")


if __name__ == "__main__":
    main()

