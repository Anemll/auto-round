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
Verify that a model saved with --format fake actually has quantized weights
by checking quantization error patterns.

This script can be used to test a model you just exported.
"""

import torch
from transformers import AutoModelForCausalLM
import os
import argparse
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def verify_quantization(model_path, grouped_channels=1, verbose=True):
    """
    Verify quantization in a fake-format model.

    Args:
        model_path: Path to the fake-format model directory
        grouped_channels: Number of channels grouped together (for grouped_channels > 1)
        verbose: Whether to print detailed information

    Returns:
        bool: True if quantization appears valid, False otherwise
    """
    # Validate path exists
    if not os.path.exists(model_path):
        print(f"Error: Model path does not exist: {model_path}")
        return False

    if verbose:
        print("=" * 60)
        print("Verifying Quantization in Fake Format Model")
        print("=" * 60)

    # Load the fake format model
    if verbose:
        print(f"\nLoading model from: {model_path}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype="auto",  # Use dtype instead of deprecated torch_dtype
            device_map="cpu",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

    # Get a weight tensor from a quantized layer
    # Skip embedding layers as they're typically not quantized
    # Prefer attention/MLP layers which are usually quantized
    layer = None
    layer_name = None
    
    # First, try to find attention or MLP layers (these are typically quantized)
    preferred_patterns = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "mlp", "attn"]
    
    for name, module in model.named_modules():
        if hasattr(module, "weight") and len(module.weight.shape) == 2:
            # Skip embedding layers
            if "embed" in name.lower():
                continue
            # Prefer attention/MLP layers
            if any(pattern in name.lower() for pattern in preferred_patterns):
                layer = module
                layer_name = name
                break
    
    # If no preferred layer found, try any non-embedding layer
    if layer is None:
        for name, module in model.named_modules():
            if hasattr(module, "weight") and len(module.weight.shape) == 2:
                # Skip embedding layers
                if "embed" in name.lower():
                    continue
                layer = module
                layer_name = name
                break

    if layer is None:
        print("Error: Could not find a linear layer with weights (excluding embedding layers)")
        return False

    weight = layer.weight.data  # Shape: [out_features, in_features]

    if verbose:
        print(f"\nLayer: {layer_name}")
        print(f"Weight shape: {weight.shape}")
        print(f"Weight dtype: {weight.dtype}")

    # Check if values look quantized (have quantization error patterns)
    if verbose:
        print("\n" + "=" * 60)
        print("Quantization Pattern Analysis")
        print("=" * 60)

    # Sample a small portion
    sample = weight[:10, :100].flatten()

    if verbose:
        print(f"\nSample values (first 20):")
        print(sample[:20].tolist())

        # Check for quantization "steps" - quantized weights often cluster
        # around certain values due to the quantization grid
        print(f"\nValue distribution analysis:")
        print(f"  Min: {sample.min().item():.6f}")
        print(f"  Max: {sample.max().item():.6f}")
        print(f"  Mean: {sample.mean().item():.6f}")
        print(f"  Std: {sample.std().item():.6f}")

        # Count unique values in sample
        unique_vals = torch.unique(sample)
        print(f"\nUnique values in sample ({sample.numel()} values): {unique_vals.numel()}")

    # The key insight: If this was truly quantized to 4-bit, then when you
    # reverse-engineer the quantization, you should find that the values
    # can be explained by: value = scale * (quantized_int - zero_point)
    # where quantized_int is in [0, 15]

    if verbose:
        print("\n" + "=" * 60)
        print("Quantization Grid Analysis")
        print("=" * 60)

    # For per-channel quantization (group_size=-1), each output channel
    # has its own scale and zero_point. Let's check one channel:
    channel_0 = weight[0, :]  # First output channel

    if verbose:
        print(f"\nFirst output channel (shape: {channel_0.shape}):")
        print(f"  Min: {channel_0.min().item():.6f}")
        print(f"  Max: {channel_0.max().item():.6f}")
        print(f"  Mean: {channel_0.mean().item():.6f}")
        print(f"  Unique values: {torch.unique(channel_0).numel()}")

    # If quantized to 4-bit, this channel should have at most 16 distinct values
    # (though in practice, due to rounding and scale, you might see fewer)
    unique_channel_0 = torch.unique(channel_0)
    unique_count_channel_0 = unique_channel_0.numel()

    if verbose:
        print(f"\nUnique values in channel 0: {unique_count_channel_0}")
        if unique_count_channel_0 <= 16:
            print("  ✓ This channel appears quantized (≤16 distinct values)")
        else:
            print(f"  ⚠ This channel has {unique_count_channel_0} distinct values")
            print("     (May be due to per-channel quantization with different scales)")

    # Check multiple channels
    if verbose:
        print(f"\nChecking multiple channels:")

    quantized_channels = 0
    checked_channels = min(10, weight.shape[0])
    results = []

    for i in range(checked_channels):
        channel = weight[i, :]
        unique_count = torch.unique(channel).numel()
        is_quantized = unique_count <= 16
        if is_quantized:
            quantized_channels += 1

        results.append({
            "channel": i,
            "unique_count": unique_count,
            "is_quantized": is_quantized,
            "min": channel.min().item(),
            "max": channel.max().item(),
        })

        if verbose:
            status = "✓" if is_quantized else ""
            print(f"  Channel {i}: {unique_count} unique values", end="")
            if is_quantized:
                print(f" {status}")
            else:
                print(f" (range: [{channel.min():.4f}, {channel.max():.4f}])")

    # Test grouped_channels if specified
    if grouped_channels > 1:
        if verbose:
            print("\n" + "=" * 60)
            print(f"Grouped Channels Analysis (grouped_channels={grouped_channels})")
            print("=" * 60)

        # For grouped_channels > 1, we need to check groups of channels together
        # The quantization is applied to groups of channels, so we should check
        # if groups of channels have similar quantization patterns

        num_channels = weight.shape[0]
        num_groups = (num_channels + grouped_channels - 1) // grouped_channels

        if verbose:
            print(f"\nTotal channels: {num_channels}")
            print(f"Group size: {grouped_channels}")
            print(f"Number of groups: {num_groups}")

        quantized_groups = 0
        for group_idx in range(min(5, num_groups)):
            start_channel = group_idx * grouped_channels
            end_channel = min(start_channel + grouped_channels, num_channels)
            group_channels = weight[start_channel:end_channel, :]

            # Flatten the group to check unique values
            group_flat = group_channels.flatten()
            unique_count = torch.unique(group_flat).numel()
            is_quantized = unique_count <= 16

            if is_quantized:
                quantized_groups += 1

            if verbose:
                status = "✓" if is_quantized else ""
                print(
                    f"  Group {group_idx} (channels {start_channel}-{end_channel-1}): "
                    f"{unique_count} unique values", end=""
                )
                if is_quantized:
                    print(f" {status}")
                else:
                    print(f" (range: [{group_flat.min():.4f}, {group_flat.max():.4f}])")

        if verbose:
            print(f"\nQuantized groups: {quantized_groups}/{min(5, num_groups)}")

        # For grouped_channels, we expect at least some groups to be quantized
        if quantized_groups == 0:
            print(
                f"\n⚠ Warning: No groups show quantization patterns. "
                f"This might indicate the model was not quantized with grouped_channels={grouped_channels}"
            )
            return False

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)

    # At least some channels should show quantization patterns
    if quantized_channels == 0:
        print(
            f"\n❌ FAIL: None of the checked {checked_channels} channels show quantization patterns "
            f"(all have >16 unique values). This suggests the model may not be quantized."
        )
        return False

    if verbose:
        print(f"\n✓ PASS: {quantized_channels}/{checked_channels} channels show quantization patterns")
        print(
            f"\nThe dequantized weights you see have many different float values because:\n"
            f"\n"
            f"1. Quantization happens at the INTEGER level (16 distinct integers: 0-15)\n"
            f"2. Dequantization converts these integers to floats: float = scale * (int - zero_point)\n"
            f"3. Each channel/group has different scale and zero_point values\n"
            f"4. This produces many different float values, even though the underlying\n"
            f"   quantized integers are limited to 16 values\n"
            f"\n"
            f"The fact that you see continuous-looking float values is CORRECT and expected\n"
            f"for a fake-format model. The quantization error is embedded in these float\n"
            f"values - they are approximations of the original weights.\n"
        )

    return True


def main():
    """Main function to parse arguments and run verification."""
    parser = argparse.ArgumentParser(
        description="Verify quantization in a fake-format model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python verify_fake_quantization.py /path/to/model
  python verify_fake_quantization.py /private/tmp/fake/Qwen3-0.6B-w4a16
  python verify_fake_quantization.py /path/to/model --grouped-channels 8
        """,
    )
    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        default="/private/tmp/fake/Qwen3-0.6B-w4a16",
        help="Path to the fake-format model directory (default: /private/tmp/fake/Qwen3-0.6B-w4a16)",
    )
    parser.add_argument(
        "--grouped-channels",
        type=int,
        default=1,
        help="Number of channels grouped together during quantization (default: 1). "
        "Use this if you quantized with --grouped_channels > 1",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output, only show pass/fail",
    )

    args = parser.parse_args()

    success = verify_quantization(
        args.model_path, grouped_channels=args.grouped_channels, verbose=not args.quiet
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

