# AutoRound ANE (Apple Neural Engine) Tests

This directory contains tests and utilities for verifying AutoRound quantization on Apple Silicon devices using MPS (Metal Performance Shaders).

## Contents

- `test_autoround.py` - Unit tests for AutoRound on ANE/MPS
- `verify_fake_quantization.py` - Standalone script to verify fake-format quantization
- `eval_hf_model.py` - Quality test script for HuggingFace models using AutoRound evaluation
- `_test_helpers.py` - Helper functions for testing
- `conftest.py` - Pytest configuration

## How to Quantize with Grouped Channels

### What are Grouped Channels?

Grouped channels is a quantization technique where multiple output channels share the same quantization scale and zero-point, rather than each channel having its own. This can reduce memory overhead for quantization parameters.

- **Standard per-channel quantization** (`grouped_channels=1`): Each output channel has its own scale/zero-point
- **Grouped channel quantization** (`grouped_channels=8`): Every 8 channels share the same scale/zero-point

### Command Line Usage

To quantize a model with grouped channels, use the `--grouped_channels` parameter:

#### Example 1: Basic Quantization with Grouped Channels

```bash
auto-round \
  --model Qwen/Qwen3-0.6B \
  --bits 4 \
  --iters 200 \
  --output_dir ./output \
  --format fake \
  --group_size -1 \
  --grouped_channels 8 \
  --nsamples 512 \
  --scale_dtype fp16 \
  --model_dtype fp16 \
  --enable_torch_compile \
  --data_type int \
  --seqlen 1024 \
  --device_map mps:0
```

#### Example 2: Complete Workflow with Grouped Channels

```bash
# Step 1: Quantize with grouped_channels=8
auto-round \
  --model Qwen/Qwen3-0.6B \
  --bits 4 \
  --iters 200 \
  --output_dir /Users/anemll/Models/PTQ/new/qwen3-0.6b-4bit-gc8-200 \
  --format fake \
  --group_size -1 \
  --grouped_channels 8 \
  --nsamples 512 \
  --scale_dtype fp16 \
  --model_dtype fp16 \
  --enable_torch_compile \
  --data_type int \
  --seqlen 1024 \
  --device_map mps:0

# Step 2: Verify the quantization (use --grouped-channels 8 to match quantization)
python test/test_ane/verify_fake_quantization.py \
  /Users/anemll/Models/PTQ/new/qwen3-0.6b-4bit-gc8-200/Qwen3-0.6B-w4a16 \
  --grouped-channels 8

# Step 3: Evaluate the quantized model
python test/test_ane/eval_hf_model.py \
  --model /Users/anemll/Models/PTQ/new/qwen3-0.6b-4bit-gc8-200/Qwen3-0.6B-w4a16 \
  --tasks boolq \
  --device cpu \
  --batch_size 8
```

#### Example 3: Different Grouped Channels Values

```bash
# Group every 4 channels together
auto-round \
  --model Qwen/Qwen3-0.6B \
  --bits 4 \
  --iters 200 \
  --output_dir ./output-gc4 \
  --format fake \
  --group_size -1 \
  --grouped_channels 4 \
  --nsamples 512 \
  --device_map mps:0

# Group every 16 channels together
auto-round \
  --model Qwen/Qwen3-0.6B \
  --bits 4 \
  --iters 200 \
  --output_dir ./output-gc16 \
  --format fake \
  --group_size -1 \
  --grouped_channels 16 \
  --nsamples 512 \
  --device_map mps:0
```

### Key Parameters

- `--group_size -1`: Per-channel quantization (each channel/group has its own scale)
- `--grouped_channels 8`: Groups every 8 output channels together for quantization
- `--format fake`: Saves in fake format (dequantized weights as floats)

### Python API Usage

```python
from auto_round import AutoRound

autoround = AutoRound(
    model="Qwen/Qwen3-0.6B",
    device_map="mps:0",
    scheme="W4A16",
    dataset="NeelNanda/pile-10k",
    iters=200,
    seqlen=1024,
    nsamples=512,
    batch_size=1,
    # Quantization scheme parameters
    bits=4,
    group_size=-1,  # Per-channel
    grouped_channels=8,  # Group every 8 channels
    scale_dtype="fp16",
    model_dtype="fp16",
    enable_torch_compile=True,
    data_type="int",
)

model, folders = autoround.quantize_and_save("./output", format="fake")
```

## How to Test if Grouped Channel Quantization is Correct

### Using the Verification Script

The `verify_fake_quantization.py` script can verify that your quantized model actually has quantized weights, including support for grouped channels.

#### Basic Usage

```bash
# Test a model quantized with standard per-channel (grouped_channels=1)
python test/test_ane/verify_fake_quantization.py /path/to/your/model

# Test a model quantized with grouped_channels=8
python test/test_ane/verify_fake_quantization.py /path/to/your/model --grouped-channels 8

# Quiet mode (only shows pass/fail)
python test/test_ane/verify_fake_quantization.py /path/to/your/model --grouped-channels 8 --quiet
```

#### What the Script Checks

1. **Layer Selection**: Automatically finds quantized layers (skips embedding layers which are typically not quantized)

2. **Per-Channel Analysis**: For each channel, checks if it has ≤16 unique values (for 4-bit quantization)

3. **Grouped Channels Analysis** (when `--grouped-channels` is specified):
   - Groups channels together as specified
   - Checks if each group has ≤16 unique values
   - Verifies that quantization patterns match the expected grouping

4. **Verification Results**:
   - ✓ PASS: Model shows correct quantization patterns
   - ❌ FAIL: Model does not show quantization patterns (may not be quantized)

### Example Output

#### Standard Per-Channel Quantization (grouped_channels=1)

```
============================================================
Verifying Quantization in Fake Format Model
============================================================

Layer: model.layers.0.self_attn.q_proj
Weight shape: torch.Size([2048, 1024])

Checking multiple channels:
  Channel 0: 14 unique values ✓
  Channel 1: 16 unique values ✓
  Channel 2: 14 unique values ✓
  ...

✓ PASS: 5/5 channels show quantization patterns
```

#### Grouped Channel Quantization (grouped_channels=8)

```
============================================================
Grouped Channels Analysis (grouped_channels=8)
============================================================

Total channels: 2048
Group size: 8
Number of groups: 256

Checking multiple groups:
  Group 0 (channels 0-7): 12 unique values ✓
  Group 1 (channels 8-15): 14 unique values ✓
  Group 2 (channels 16-23): 13 unique values ✓
  ...

✓ PASS: Groups show correct quantization patterns
```

### Understanding the Results

#### What "Unique Values" Mean

- **For 4-bit quantization**: Each channel/group should have ≤16 unique values
- **Why you see many float values**: The dequantized weights are floats, but they come from only 16 integer values (0-15) multiplied by different scales

#### Expected Patterns

1. **Per-channel (grouped_channels=1)**:
   - Each individual channel should have ≤16 unique values
   - Different channels may have different scales, so values vary

2. **Grouped channels (grouped_channels=8)**:
   - Groups of 8 channels should have ≤16 unique values when flattened together
   - Individual channels within a group may have more unique values, but the group as a whole should be quantized

### Troubleshooting

#### If the test fails:

1. **Check the layer being tested**: The script skips embedding layers. Make sure it's testing an attention or MLP layer.

2. **Verify quantization parameters**: Ensure you used the correct `--grouped-channels` value when quantizing.

3. **Check model format**: Make sure the model was saved with `--format fake`.

4. **Verify quantization actually ran**: Check that the quantization process completed without errors.

### Running Unit Tests

To run the full test suite:

```bash
# Activate your virtual environment
source .venv/bin/activate

# Run all ANE tests
python -m pytest test/test_ane/ -v

# Run a specific test
python -m pytest test/test_ane/test_autoround.py::TestAutoRoundANE::test_verify_fake_quantization -v
```

## Converting Models to ANE Format

After quantizing a model with AutoRound, you can convert it to Apple Neural Engine (ANE) format for optimized inference on Apple Silicon.

### Example Conversion

```bash
./anemll/utils/convert_model.sh \
  --model /Users/anemll/Models/PTQ/new/qwen3-0.6b-4bit-gc8-200/Qwen3-0.6B-w4a16 \
  --output "/Users/anemll/Models/ANE/qwen3-0.6b-ane-gc8-200-ctx1024" \
  --lut1 6 \
  --lut2 4 \
  --lut3 6 \
  --context 1024 \
  --batch 64 \
  --chunk 1
```

### Conversion Parameters

- `--model`: Path to the quantized model (AutoRound format)
- `--output`: Output directory for the ANE-converted model
- `--lut1`, `--lut2`, `--lut3`: Lookup table parameters for quantization
- `--context`: Context length (sequence length) for the model
- `--batch`: Batch size for processing
- `--chunk`: Chunk size for processing

### Notes on Conversion

- The input model should be in AutoRound format (quantized with `--format fake` or `--format auto_round`)
- ANE format is optimized for inference on Apple Neural Engine
- Context length should match the sequence length used during quantization if possible

## Evaluating ANE Models

After converting a model to ANE format, you can evaluate it using the ANE evaluation script.

### Example Evaluation

```bash
./evaluate/ane/run_eval.sh \
  --model ~/Models/ANE/qwen3-0.6b-ane-gc8-200-ctx1024 \
  --tasks "boolq"
```

### Evaluation Parameters

- `--model`: Path to the ANE-converted model directory
- `--tasks`: Comma-separated list of evaluation tasks (e.g., `"boolq"`, `"boolq,hellaswag,piqa"`)

### Multiple Tasks

```bash
./evaluate/ane/run_eval.sh \
  --model ~/Models/ANE/qwen3-0.6b-ane-gc8-200-ctx1024 \
  --tasks "boolq,hellaswag,piqa,mmlu"
```

### Notes on ANE Evaluation

- The evaluation script is specifically designed for ANE-optimized models
- ANE models are optimized for inference on Apple Neural Engine hardware
- Evaluation tasks should be compatible with the model's capabilities

## Quality Testing with HuggingFace Models

The `eval_hf_model.py` script provides a convenient way to evaluate quantized or dequantized HuggingFace models on various tasks.

### Basic Usage

```bash
# Evaluate on boolq task
python test/test_ane/eval_hf_model.py \
  --model /path/to/your/model \
  --tasks boolq
```

### Multiple Tasks

```bash
# Evaluate on multiple tasks
python test/test_ane/eval_hf_model.py \
  --model /path/to/your/model \
  --tasks boolq,hellaswag,piqa
```

### With Custom Settings

```bash
# Use MPS device with custom batch size
python test/test_ane/eval_hf_model.py \
  --model /path/to/your/model \
  --tasks boolq \
  --device mps \
  --batch_size 4 \
  --max_batch_size 16
```

### Limit Examples

```bash
# Test on first 100 examples
python test/test_ane/eval_hf_model.py \
  --model /path/to/your/model \
  --tasks boolq \
  --limit 100
```

### Key Features

- **Flexible task selection**: Specify single or multiple tasks
- **Device support**: CPU, MPS, or CUDA with automatic fallback
- **Batch size control**: Customize batch size for optimal performance
- **Chat template handling**: Automatically disables chat templates for evaluation
- **Comparison notes**: Provides context for BoolQ results
- **MPS error handling**: Automatically reduces batch size or falls back to CPU if MPS encounters errors (e.g., "MPSGaph does not support tensor dims larger than INT_MAX")

### MPS Limitations

If you encounter MPS errors like "MPSGaph does not support tensor dims larger than INT_MAX", this is a known MPS limitation that can vary between:
- Different PyTorch versions
- Different lm-eval versions
- Different Apple Silicon hardware generations

**Solutions:**
1. **Reduce batch size** (recommended): Use smaller batch sizes for MPS (e.g., `--batch_size 4` or `--batch_size 2`)
2. **Automatic handling**: The script will automatically reduce batch size on MPS errors and retry
3. **Use CPU**: For best compatibility, use `--device cpu` on Apple Silicon

**Example with reduced batch size for MPS:**
```bash
python test/test_ane/eval_hf_model.py \
  --model /path/to/model \
  --tasks boolq \
  --device mps \
  --batch_size 4 \
  --max_batch_size 8
```

### Example: Evaluating a Dequantized Model

```bash
python test/test_ane/eval_hf_model.py \
  --model /Users/anemll/Models/QAT/qwen3-0.6b-dequantized-CORRECT \
  --tasks boolq \
  --device cpu \
  --batch_size 8
```

## Handling Outliers in Quantization

AutoRound provides several techniques to handle outliers, though it does not support rotation-based techniques like spin quantization:

### 1. Exclude Layers from Quantization (`--fp_layers`)

The most direct approach is to exclude outlier layers from quantization, keeping them in full precision:

```bash
auto-round \
  --model Qwen/Qwen3-0.6B \
  --bits 4 \
  --iters 200 \
  --output_dir ./output \
  --format fake \
  --fp_layers "model.layers.3.mlp.down_proj,model.layers.5.mlp.down_proj" \
  --device_map mps:0
```

**When to use**: When specific layers have large output values that cause overflow or accuracy issues. This is recommended for layers that show extreme values during inference (e.g., with chat templates).

### 2. Scale Thresholding (`q_scale_thresh`)

AutoRound automatically clips quantization scales to prevent underflow:

- Default: `q_scale_thresh=1e-5` (built into quantization functions)
- Prevents very small scales that can cause numerical instability
- Helps with outlier handling by ensuring scales are within a reasonable range

### 3. Min-Max Scale Tuning

AutoRound supports tuning min/max scales for quantization ranges, which can help handle outliers:

- Enabled by default with `--enable_minmax_tuning` (or `enable_minmax_tuning=True` in API)
- Allows the quantization range to adapt to the actual weight distribution
- Can be disabled with `--disable_minmax_tuning` if needed

### 4. Per-Layer Configuration

You can configure different quantization schemes per layer to handle outliers:

```python
from auto_round import AutoRound

layer_config = {
    "model.layers.3.mlp.down_proj": {"bits": 8},  # Use 8-bit for outlier layer
    "model.layers.5.mlp.down_proj": {"bits": 8},  # Use 8-bit for outlier layer
}

autoround = AutoRound(
    model="Qwen/Qwen3-0.6B",
    bits=4,  # Default 4-bit
    layer_config=layer_config,  # Override for specific layers
    # ... other parameters
)
```

### Comparison with Spin Quantization

**Spin Quantization** uses learned rotation matrices to redistribute quantization error:
- Finds a rotation matrix R that redistributes quantization error
- Applies R to one layer: `W1 → R·W1`
- Applies R⁻¹ to another layer: `W2 → W2·R⁻¹`
- The rotations cancel out mathematically (`R·W1 · W2·R⁻¹ = W1·W2`), so the computation is unchanged
- But quantization error is redistributed, making quantization easier

**AutoRound** does not support rotation-based quantization. Instead, it uses different techniques:

| Feature | Spin Quant | AutoRound |
|---------|-----------|-----------|
| **Method** | Rotation matrices (R applied to one layer, R⁻¹ to another) | Layer exclusion or higher bit widths |
| **Outlier Handling** | Redistributes error via rotations | Exclude entire layers or use higher bits per layer |
| **Granularity** | Per-layer pairs (rotation + inverse rotation) | Per-layer (coarse-grained) |
| **Mathematical Property** | Preserves exact computation (rotations cancel) | Changes computation (excluded layers stay FP16) |

**Key Difference**: Spin quantization uses rotation matrices to redistribute quantization error without changing the mathematical computation, while AutoRound handles outliers by excluding layers from quantization or using higher bit widths, which does change the computation.

### Example: Handling Outlier Layers

```bash
# Step 1: Quantize and identify problematic layers
auto-round \
  --model Qwen/Qwen3-0.6B \
  --bits 4 \
  --iters 200 \
  --output_dir ./output \
  --format fake \
  --device_map mps:0

# Step 2: If certain layers cause issues, re-quantize excluding them
auto-round \
  --model Qwen/Qwen3-0.6B \
  --bits 4 \
  --iters 200 \
  --output_dir ./output-fixed \
  --format fake \
  --fp_layers "model.layers.3.mlp.down_proj,model.layers.5.mlp.down_proj" \
  --device_map mps:0
```

### References

- See [Tips and Tricks](../../docs/tips_and_tricks.md) section 4 for details on handling overflow caused by chat templates
- The `--fp_layers` parameter is documented in the main AutoRound CLI help

## Notes

- **MPS Requirement**: These tests require MPS (Metal Performance Shaders) which is available on Apple Silicon Macs (M1, M2, M3, etc.)
- **Embedding Layers**: Embedding layers are typically not quantized by default. The verification script automatically skips them.
- **Fake Format**: The fake format stores dequantized weights as floats, but they still contain quantization error patterns that can be verified.

## Running Evaluation with Standard lm-eval-harness

You can also use the standard `lm_eval` command directly with AutoRound models. For AutoRound format models, you need to ensure `AutoRoundConfig` is imported first.

### Basic Command

```bash
lm_eval \
  --model hf \
  --model_args pretrained=/path/to/your/model \
  --tasks boolq \
  --device cuda:0 \
  --batch_size 16
```

### For Apple Silicon (MPS)

**Important**: MPS has limitations with device_map. The recommended approach is to use CPU for evaluation on Apple Silicon, or use explicit device placement:

**Option 1: Use CPU (Recommended for MPS)**

```bash
lm_eval \
  --model hf \
  --model_args pretrained=/path/to/your/model \
  --tasks boolq \
  --device cpu \
  --batch_size 16
```

**Option 2: Use explicit device_map without placeholders**

```bash
lm_eval \
  --model hf \
  --model_args pretrained=/path/to/your/model,device_map=mps:0 \
  --tasks boolq \
  --device mps:0 \
  --batch_size 16
```

**Note**: `device_map=auto` can cause "Placeholder storage has not been allocated on MPS device!" errors. If you encounter this, use CPU or explicit device mapping.

### Multiple Tasks

```bash
# Using CPU (recommended for Apple Silicon)
lm_eval \
  --model hf \
  --model_args pretrained=/path/to/your/model \
  --tasks boolq,hellaswag,piqa \
  --device cpu \
  --batch_size 16
```

### With Model Dtype

```bash
# Using CPU (recommended for Apple Silicon)
lm_eval \
  --model hf \
  --model_args pretrained=/path/to/your/model,dtype=bfloat16 \
  --tasks boolq \
  --device cpu \
  --batch_size 16
```

### Important Notes

1. **AutoRoundConfig Import**: When using AutoRound format models, the `AutoRoundConfig` is automatically loaded when you import from `auto_round`. If you're using lm-eval-harness directly, make sure the `auto_round` package is installed and importable.

2. **Model Path**: The model path can be:
   - A local directory containing the quantized model
   - A HuggingFace model identifier
   - Path to a model saved in AutoRound format

3. **Device Support**: 
   - `cuda:0` for NVIDIA GPUs
   - `mps:0` for Apple Silicon (M1/M2/M3) - **Note**: MPS has limitations. See MPS-specific notes below.
   - `cpu` for CPU evaluation (recommended for Apple Silicon to avoid MPS issues)

4. **MPS Device Limitations**: 
   - safetensors doesn't support loading directly to MPS - use `device_map=cpu` or `device_map=mps:0` (not `auto`)
   - `device_map=auto` can cause "Placeholder storage has not been allocated on MPS device!" errors
   - **Recommendation**: Use `--device cpu` for evaluation on Apple Silicon to avoid MPS compatibility issues

### Example: Evaluating a Quantized Model

```bash
# Evaluate a model quantized with AutoRound
# Using CPU (recommended for Apple Silicon to avoid MPS issues)
lm_eval \
  --model hf \
  --model_args pretrained=/Users/anemll/Models/PTQ/new/qwen3-0.6b-4bit-gc8-200/Qwen3-0.6B-w4a16 \
  --tasks boolq \
  --device cpu \
  --batch_size 16 \
  --limit 100
```

If you want to try MPS (may have issues):

```bash
lm_eval \
  --model hf \
  --model_args pretrained=/Users/anemll/Models/PTQ/new/qwen3-0.6b-4bit-gc8-200/Qwen3-0.6B-w4a16,device_map=mps:0 \
  --tasks boolq \
  --device mps:0 \
  --batch_size 16 \
  --limit 100
```

### Python Script Alternative

If you prefer to use Python directly:

```python
from lm_eval import simple_evaluate
from transformers import AutoRoundConfig  # This import ensures AutoRound models load correctly

result = simple_evaluate(
    model="hf",
    model_args="pretrained=/path/to/your/model",
    tasks=["boolq"],
    device="mps:0",
    batch_size=16
)

print(result)
```

## References

- [AutoRound Documentation](../../README.md)
- [Quantization Schemes](../../docs/)
- [Tips and Tricks](../../docs/tips_and_tricks.md)
- [lm-eval-harness Documentation](https://github.com/EleutherAI/lm-evaluation-harness)

