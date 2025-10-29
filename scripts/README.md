# Scripts

This directory contains utility scripts for working with neural-cherche-go.

## convert_to_onnx.py

Convert neural-cherche models to ONNX format for use with neural-cherche-go.

### Installation

```bash
pip install torch transformers neural-cherche onnx onnxruntime
```

### Usage

#### Convert ColBERT

```bash
# Use pre-trained model from HuggingFace
python convert_to_onnx.py \
    --model raphaelsty/neural-cherche-colbert \
    --output ./models/colbert.onnx \
    --type colbert \
    --max-length 256

# Use local fine-tuned model
python convert_to_onnx.py \
    --model ./my_colbert_checkpoint \
    --output ./models/my_colbert.onnx \
    --type colbert
```

#### Convert Splade

```bash
python convert_to_onnx.py \
    --model raphaelsty/neural-cherche-sparse-embed \
    --output ./models/splade.onnx \
    --type splade \
    --max-length 256
```

### Output Files

The script creates three files:

1. **`model.onnx`** - The ONNX model file
2. **`model_linear.npy`** - Linear projection weights (ColBERT only)
3. **`model_metadata.json`** - Model metadata

### Copying Tokenizer

After conversion, copy the tokenizer files:

```bash
# If using HuggingFace model
cp ~/.cache/huggingface/hub/models--*/snapshots/*/tokenizer.json ./models/

# If using local model
cp ./my_checkpoint/tokenizer.json ./models/
```

### Verification

The script automatically:
- Validates the ONNX model structure
- Tests inference with dummy inputs
- Reports output shapes

### Troubleshooting

**Error: "No module named 'neural_cherche'"**
```bash
pip install neural-cherche
```

**Error: "ONNX export failed"**
- Ensure PyTorch >= 1.13
- Try different opset version: `--opset-version 13`

**Error: "Model output shape mismatch"**
- Check max_length parameter
- Verify model architecture

## Example: Full Workflow

```bash
# 1. Convert model
python scripts/convert_to_onnx.py \
    --model raphaelsty/neural-cherche-colbert \
    --output ./models/colbert.onnx

# 2. Copy tokenizer
mkdir -p models
# Find and copy tokenizer.json from HuggingFace cache

# 3. Use in Go
cd examples
go run example_colbert.go
```

## Model Sizes

| Model | ONNX Size | Linear Weights | Total |
|-------|-----------|----------------|-------|
| ColBERT (base) | ~420 MB | ~50 MB | ~470 MB |
| Splade (base) | ~420 MB | - | ~420 MB |
| SparseEmbed | ~420 MB | - | ~420 MB |

## Performance

ONNX models typically have:
- **CPU**: Similar to PyTorch (~5-10% slower)
- **GPU**: 10-20% slower than PyTorch CUDA
- **Memory**: Same as PyTorch
- **Startup**: Much faster (no Python overhead)

## Advanced Options

### Custom Opset Version

```bash
# Use ONNX opset 13 for compatibility
python convert_to_onnx.py --opset-version 13
```

### Dynamic Batch Size

By default, the exporter creates dynamic axes for batch size and sequence length. This allows flexible input sizes in Go.

### Quantization (Experimental)

To reduce model size:

```bash
# After conversion, quantize the model
python -c "
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'colbert.onnx'
model_quant = 'colbert_quant.onnx'

quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
"
```

This can reduce model size by ~4x with minimal accuracy loss.

## Contributing

To add support for new model types:

1. Implement conversion function in `convert_to_onnx.py`
2. Add model type to choices
3. Test with example model
4. Update documentation
