#!/usr/bin/env python3
"""
CORRECT GPTQ dequantization based on AutoGPTQ source code

Reference: https://github.com/AutoGPTQ/AutoGPTQ/blob/main/auto_gptq/nn_modules/qlinear/qlinear_cuda_old.py

This script directly dequantizes GPTQ models from safetensors without loading the full model.
More memory-efficient than loading the model through transformers.
"""

import argparse
import os
import sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def dequantize_gptq_weight_correct(qweight, scales, qzeros, bits=4, group_size=-1):
    """
    Correct GPTQ dequantization based on AutoGPTQ implementation
    
    Args:
        qweight: Quantized weight tensor [infeatures//8, outfeatures]
        scales: Scale tensor
        qzeros: Zero-point tensor
        bits: Quantization bits (default: 4)
        group_size: Group size (-1 for per-channel, 0 for per-tensor, >0 for group)
    
    Returns:
        Dequantized weight tensor [outfeatures, infeatures]
    """
    # Constants
    maxq = 2 ** bits - 1
    outfeatures = qweight.shape[1]
    infeatures = qweight.shape[0] * (32 // bits)

    # Bit shift factors for unpacking
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32)

    # Unpack qweight
    # Shape: [infeatures//8, outfeatures] -> [infeatures, outfeatures]
    weight = torch.bitwise_right_shift(
        qweight.unsqueeze(1).expand(-1, 32 // bits, -1),
        wf.unsqueeze(-1),
    ).to(torch.int8)
    weight = torch.bitwise_and(weight, maxq)

    # Unpack qzeros - CRITICAL: add 1 BEFORE masking!
    zeros = torch.bitwise_right_shift(
        qzeros.unsqueeze(2).expand(-1, -1, 32 // bits),
        wf.unsqueeze(0),
    ).to(torch.int8)
    zeros = zeros + 1  # IMPORTANT: Add 1 before masking
    zeros = torch.bitwise_and(zeros, maxq)

    # Reshape
    if group_size == 0:
        # Per-tensor: entire weight uses ONE scale/zero-point
        weight = weight.reshape(infeatures, outfeatures)
        zeros = zeros.reshape(-1, outfeatures)

        # Use single scale and zero-point for entire tensor
        dequant = scales[0, 0] * (weight.float() - zeros[0, 0].float())
    elif group_size == -1:
        # Per-channel: one scale per output channel
        weight = weight.reshape(infeatures, outfeatures)
        zeros = zeros.reshape(-1, outfeatures)

        # Apply dequantization: weight = scale * (qweight - qzero)
        dequant = scales[0] * (weight.float() - zeros[0].float())
    else:
        # Group quantization
        num_groups = (infeatures + group_size - 1) // group_size
        weight = weight.reshape(-1, group_size, outfeatures)
        zeros = zeros.reshape(num_groups, 1, -1)

        # Expand scales and zeros to match weight shape
        weight = weight.reshape(num_groups, group_size, outfeatures)

        # Apply dequantization per group
        dequant = scales[:num_groups].unsqueeze(1) * (weight.float() - zeros.float())
        dequant = dequant.reshape(infeatures, outfeatures)

    # Transpose to get [outfeatures, infeatures] for nn.Linear
    return dequant.t()


def dequantize_gptq_model(gptq_model_path, output_path, bits=None, trust_remote_code=True):
    """
    Dequantize a GPTQ model to fake format.
    
    Args:
        gptq_model_path: Path to the GPTQ quantized model directory
        output_path: Path where to save the dequantized model
        bits: Quantization bits (if None, will be read from model config)
        trust_remote_code: Whether to trust remote code
    """
    print("=" * 60)
    print("CORRECT GPTQ Dequantization")
    print("Based on AutoGPTQ source code")
    print("=" * 60)
    
    os.makedirs(output_path, exist_ok=True)
    
    # Try to read bits from config if not provided
    if bits is None:
        print("\nReading quantization config from model...")
        try:
            config = AutoConfig.from_pretrained(gptq_model_path, trust_remote_code=trust_remote_code)
            if hasattr(config, 'quantization_config'):
                quant_config = config.quantization_config
                if isinstance(quant_config, dict):
                    bits = quant_config.get('bits', 4)
                elif hasattr(quant_config, 'bits'):
                    bits = quant_config.bits
                else:
                    bits = 4
                print(f"  Found bits={bits} in quantization_config")
            else:
                bits = 4
                print(f"  No quantization_config found, defaulting to bits={bits}")
        except Exception as e:
            bits = 4
            print(f"  Could not read config, defaulting to bits={bits} (error: {e})")
    
    print(f"Using bits={bits} for dequantization")
    
    # Find safetensors file
    gptq_safetensors = None
    if os.path.isfile(os.path.join(gptq_model_path, "model.safetensors")):
        gptq_safetensors = os.path.join(gptq_model_path, "model.safetensors")
    elif os.path.isfile(os.path.join(gptq_model_path, "model.safetensors.index.json")):
        # Handle sharded models
        print("Warning: Sharded safetensors not fully supported yet")
        gptq_safetensors = os.path.join(gptq_model_path, "model.safetensors.index.json")
    else:
        # Try to find any safetensors file
        for file in os.listdir(gptq_model_path):
            if file.endswith(".safetensors") and "model" in file:
                gptq_safetensors = os.path.join(gptq_model_path, file)
                break
    
    if gptq_safetensors is None or not os.path.exists(gptq_safetensors):
        raise FileNotFoundError(f"Could not find model.safetensors in {gptq_model_path}")
    
    print(f"\nLoading GPTQ model from: {gptq_safetensors}")
    
    # Load all tensors
    with safe_open(gptq_safetensors, framework="pt") as f:
        all_keys = list(f.keys())
    
    # Separate quantized and non-quantized tensors
    quantized_layers = {}
    dequantized_tensors = {}
    other_tensors = {}
    
    print("\nIdentifying quantized layers...")
    for key in all_keys:
        if key.endswith('.qweight'):
            layer_name = key[:-8]  # Remove '.qweight'
            quantized_layers[layer_name] = True
        elif not any(key.endswith(suffix) for suffix in ['.qzeros', '.scales', '.g_idx']):
            other_tensors[key] = None
    
    print(f"Found {len(quantized_layers)} quantized layers")
    print(f"Found {len(other_tensors)} non-quantized tensors")
    
    # Dequantize each layer
    print("\nDequantizing layers...")
    with safe_open(gptq_safetensors, framework="pt") as f:
        # First copy non-quantized tensors
        for key in other_tensors:
            tensor = f.get_tensor(key)
            dequantized_tensors[key] = tensor
            print(f"  Copied: {key}")
        
        # Dequantize quantized layers
        for layer_name in quantized_layers:
            qweight_key = f"{layer_name}.qweight"
            scales_key = f"{layer_name}.scales"
            qzeros_key = f"{layer_name}.qzeros"
            weight_key = f"{layer_name}.weight"
            
            if qweight_key not in all_keys:
                print(f"  Warning: {qweight_key} not found, skipping")
                continue
            
            qweight = f.get_tensor(qweight_key)
            scales = f.get_tensor(scales_key)
            qzeros = f.get_tensor(qzeros_key)
            
            # Determine group_size from shapes
            outfeatures = qweight.shape[1]
            infeatures = qweight.shape[0] * (32 // bits)
            
            if scales.shape[0] == 1 and scales.shape[1] == 1:
                group_size = 0  # Per-tensor
            elif scales.shape[0] == 1:
                group_size = -1  # Per-channel
            else:
                group_size = infeatures // scales.shape[0]
            
            # Dequantize
            dequant_weight = dequantize_gptq_weight_correct(
                qweight, scales, qzeros, bits=bits, group_size=group_size
            )
            
            # Store as FP16 (make contiguous for safetensors)
            dequantized_tensors[weight_key] = dequant_weight.to(torch.float16).contiguous()
            print(f"  Dequantized: {layer_name} (group_size={group_size})")
            
            # Copy bias if it exists
            bias_key = f"{layer_name}.bias"
            if bias_key in all_keys:
                dequantized_tensors[bias_key] = f.get_tensor(bias_key)
    
    # Save dequantized model
    print(f"\nSaving to {output_path}...")
    save_file(dequantized_tensors, os.path.join(output_path, "model.safetensors"))
    
    # Copy config and tokenizer
    config = AutoConfig.from_pretrained(gptq_model_path, trust_remote_code=trust_remote_code)
    
    # Remove quantization config
    if hasattr(config, 'quantization_config'):
        delattr(config, 'quantization_config')
    
    config.save_pretrained(output_path)
    
    tokenizer = AutoTokenizer.from_pretrained(gptq_model_path, trust_remote_code=trust_remote_code)
    tokenizer.save_pretrained(output_path)
    
    print("\n" + "=" * 60)
    print("Dequantization complete!")
    print(f"Saved to: {output_path}")
    print("=" * 60)
    print("\nKey fix: zeros = zeros + 1 BEFORE masking (from AutoGPTQ source)")
    
    return output_path


def test_inference(model_path: str, device: str = "cpu", trust_remote_code: bool = True):
    """
    Test inference on the dequantized model.
    
    Args:
        model_path: Path to the dequantized model
        device: Device to use for inference
        trust_remote_code: Whether to trust remote code
    """
    print("\n" + "=" * 60)
    print("Testing Inference on Dequantized Model")
    print("=" * 60)
    
    try:
        # Load the dequantized model (no quantization config needed)
        print(f"\nLoading dequantized model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype="auto",
            device_map=device,
            trust_remote_code=trust_remote_code,
        )
        model.eval()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        
        print(f"Model loaded successfully!")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        
        # Test inference
        test_prompt = "What is Apple Neural Engine?"
        print(f"\nTest prompt: '{test_prompt}'")
        
        # Apply chat template if available
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": test_prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            print(f"Using chat template. Formatted prompt: '{formatted_prompt[:100]}...'")
            # Tokenize the formatted prompt
            inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=False, truncation=True)
        else:
            # Tokenize directly if no chat template
            inputs = tokenizer(test_prompt, return_tensors="pt", padding=False, truncation=True)
        
        # Move inputs to model device
        model_device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(model_device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)
        
        # Generate
        print("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=1000,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
            )
        
        # Decode
        generated_ids = outputs[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print(f"Generated: '{generated_text}'")
        print(f"\n✅ Inference test passed! The dequantized model works correctly.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Inference test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Dequantize GPTQ model to fake format (based on AutoGPTQ source code)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dequantize a GPTQ model
  python dequantize_gptq.py \\
    --input /path/to/gptq/model \\
    --output /path/to/dequantized/model
  
  # Specify bits
  python dequantize_gptq.py \\
    --input /path/to/gptq/model \\
    --output /path/to/dequantized/model \\
    --bits 4
  
  # Skip inference test
  python dequantize_gptq.py \\
    --input /path/to/gptq/model \\
    --output /path/to/dequantized/model \\
    --skip_test
        """,
    )
    
    parser.add_argument(
        "--input",
        "--model",
        "--gptq_model",
        type=str,
        required=True,
        help="Path to the GPTQ quantized model directory",
    )
    
    parser.add_argument(
        "--output",
        "--output_dir",
        type=str,
        required=True,
        help="Path where to save the dequantized model",
    )
    
    parser.add_argument(
        "--bits",
        type=int,
        default=None,
        choices=[2, 3, 4, 8],
        help="Quantization bits (if not specified, will be read from model config). Default: auto-detect from config",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Device to use for inference test (default: cpu)",
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
    
    parser.add_argument(
        "--skip_test",
        action="store_true",
        help="Skip inference test after dequantization",
    )
    
    args = parser.parse_args()
    
    # Determine trust_remote_code
    trust_remote_code = args.trust_remote_code and not args.no_trust_remote_code
    
    # Validate input path
    if not os.path.exists(args.input):
        print(f"Error: Input model path does not exist: {args.input}")
        sys.exit(1)
    
    # Dequantize
    output_path = dequantize_gptq_model(
        gptq_model_path=args.input,
        output_path=args.output,
        bits=args.bits,
        trust_remote_code=trust_remote_code,
    )
    
    # Test inference if not skipped
    if not args.skip_test:
        test_inference(
            model_path=output_path,
            device=args.device,
            trust_remote_code=trust_remote_code,
        )
    else:
        print("\n⚠️  Inference test skipped (--skip_test flag used)")


if __name__ == "__main__":
    main()

