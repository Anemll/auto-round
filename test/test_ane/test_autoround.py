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

import unittest
import os
import shutil
import torch
from transformers import AutoModelForCausalLM

from auto_round import AutoRound
from test.test_ane._test_helpers import model_infer


class TestAutoRoundANE(unittest.TestCase):
    """Test AutoRound on Apple Neural Engine (ANE)."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        cls.model_name = "facebook/opt-125m"
        cls.save_dir = "./tmp_autoround_ane_test"
        if os.path.exists(cls.save_dir):
            shutil.rmtree(cls.save_dir)
        os.makedirs(cls.save_dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        if os.path.exists(cls.save_dir):
            shutil.rmtree(cls.save_dir)

    def test_basic_quantization_ane(self):
        """Test basic quantization with ANE device."""
        # Check if MPS (Metal Performance Shaders) is available for ANE
        if not torch.backends.mps.is_available():
            self.skipTest("MPS (ANE) is not available on this system")

        autoround = AutoRound(
            model=self.model_name,
            device_map="mps:0",
            scheme="W4A16",
            dataset="NeelNanda/pile-10k",
            iters=10,
            seqlen=128,
            nsamples=32,
            batch_size=1,
        )

        model, folders = autoround.quantize_and_save(self.save_dir, format="fake")
        self.assertIsNotNone(model)
        self.assertIsNotNone(folders)

    def test_quantization_with_mps_device(self):
        """Test quantization using MPS device (Apple Silicon)."""
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is not available on this system")

        autoround = AutoRound(
            model=self.model_name,
            device_map="mps:0",
            scheme="W4A16",
            dataset="NeelNanda/pile-10k",
            iters=5,
            seqlen=64,
            nsamples=16,
            batch_size=1,
            low_gpu_mem_usage=True,
        )

        model, folders = autoround.quantize_and_save(self.save_dir, format="fake")
        self.assertIsNotNone(model)

        # Test inference
        tokenizer = autoround.tokenizer
        output = model_infer(model, tokenizer)
        self.assertIsNotNone(output)

    def test_verify_fake_quantization(self):
        """
        Verify that a model saved with --format fake actually has quantized weights
        by checking quantization error patterns.
        """
        if not torch.backends.mps.is_available():
            self.skipTest("MPS (ANE) is not available on this system")

        # Create a quantized model with fake format
        autoround = AutoRound(
            model=self.model_name,
            device_map="mps:0",
            scheme="W4A16",
            dataset="NeelNanda/pile-10k",
            iters=5,
            seqlen=64,
            nsamples=16,
            batch_size=1,
        )

        model, folders = autoround.quantize_and_save(self.save_dir, format="fake")
        self.assertIsNotNone(model)
        self.assertIsNotNone(folders)

        # Get the saved model path
        fake_model_path = folders[0] if folders else self.save_dir

        # Validate path exists
        self.assertTrue(os.path.exists(fake_model_path), f"Model path does not exist: {fake_model_path}")

        # Load the fake format model from disk
        loaded_model = AutoModelForCausalLM.from_pretrained(
            fake_model_path,
            torch_dtype="auto",
            device_map="cpu",
            trust_remote_code=True,
        )

        # Get a weight tensor from a quantized layer
        # Try to find a quantized linear layer
        layer = None
        for name, module in loaded_model.named_modules():
            if hasattr(module, "weight") and len(module.weight.shape) == 2:
                layer = module
                break

        self.assertIsNotNone(layer, "Could not find a linear layer with weights")

        weight = layer.weight.data  # Shape: [out_features, in_features]

        # Check if values look quantized (have quantization error patterns)
        # Sample a small portion for analysis
        sample = weight[:10, :100].flatten()

        # Count unique values in sample
        unique_vals = torch.unique(sample)
        unique_count = unique_vals.numel()

        # For quantized weights, we expect to see quantization patterns
        # The key insight: If this was truly quantized to 4-bit, then when you
        # reverse-engineer the quantization, you should find that the values
        # can be explained by: value = scale * (quantized_int - zero_point)
        # where quantized_int is in [0, 15]

        # For per-channel quantization (group_size=-1), each output channel
        # has its own scale and zero_point. Let's check one channel:
        channel_0 = weight[0, :]  # First output channel
        unique_channel_0 = torch.unique(channel_0)
        unique_channel_count = unique_channel_0.numel()

        # If quantized to 4-bit, this channel should have at most 16 distinct values
        # (though in practice, due to rounding and scale, you might see fewer)
        self.assertLessEqual(
            unique_channel_count,
            16,
            f"Channel 0 has {unique_channel_count} unique values, "
            f"expected ≤16 for 4-bit quantization. "
            f"Range: [{channel_0.min():.6f}, {channel_0.max():.6f}]",
        )

        # Check multiple channels to verify quantization pattern
        quantized_channels = 0
        checked_channels = min(5, weight.shape[0])

        for i in range(checked_channels):
            channel = weight[i, :]
            unique_count = torch.unique(channel).numel()
            if unique_count <= 16:
                quantized_channels += 1

        # At least some channels should show quantization patterns
        self.assertGreater(
            quantized_channels,
            0,
            f"None of the checked {checked_channels} channels show quantization patterns "
            f"(all have >16 unique values). This suggests the model may not be quantized.",
        )

        # Additional verification: Check that the weights are not the same as original
        # (quantization should introduce some error/approximation)
        # We can't directly compare to original here, but we can verify the values
        # have reasonable ranges and patterns consistent with quantization

        # Verify weight statistics are reasonable
        self.assertGreater(weight.abs().mean().item(), 0, "Weights should not be all zeros")
        self.assertLess(weight.abs().max().item(), 100, "Weights should not have extreme values")


if __name__ == "__main__":
    unittest.main()

