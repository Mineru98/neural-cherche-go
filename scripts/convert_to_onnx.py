#!/usr/bin/env python3
"""
Convert neural-cherche ColBERT model to ONNX format for use with neural-cherche-go.

Usage:
    python convert_to_onnx.py --model raphaelsty/neural-cherche-colbert \
                               --output ./models/colbert.onnx

Requirements:
    pip install torch transformers neural-cherche onnx
"""

import argparse
import json
import os

import numpy as np
import torch
from neural_cherche import models


def convert_colbert_to_onnx(
    model_name_or_path: str,
    output_path: str,
    max_length: int = 256,
    opset_version: int = 14,
):
    """Convert ColBERT model to ONNX format.

    Args:
        model_name_or_path: HuggingFace model path or local path
        output_path: Output path for ONNX model
        max_length: Maximum sequence length
        opset_version: ONNX opset version
    """
    print(f"Loading ColBERT model: {model_name_or_path}")
    model = models.ColBERT(
        model_name_or_path=model_name_or_path,
        device="cpu",
    )

    print("Preparing for ONNX export...")

    # Set model to eval mode
    model.model.eval()

    # Create dummy inputs
    batch_size = 1
    dummy_input_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
    dummy_attention_mask = torch.ones((batch_size, max_length), dtype=torch.long)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Exporting to ONNX: {output_path}")

    # Export to ONNX
    torch.onnx.export(
        model.model,  # The underlying transformer model
        (
            {"input_ids": dummy_input_ids, "attention_mask": dummy_attention_mask}
            if hasattr(model.model, "forward")
            else (dummy_input_ids, dummy_attention_mask)
        ),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )

    print(f"✓ ONNX model saved to {output_path}")

    # Save linear projection weights
    linear_path = output_path.replace(".onnx", "_linear.npy")
    linear_weights = model.linear.weight.detach().cpu().numpy()
    np.save(linear_path, linear_weights)
    print(f"✓ Linear weights saved to {linear_path}")
    print(f"  Shape: {linear_weights.shape}")

    # Save metadata
    metadata_path = output_path.replace(".onnx", "_metadata.json")
    metadata = {
        "model_name": model_name_or_path,
        "max_length_query": model.max_length_query,
        "max_length_document": model.max_length_document,
        "embedding_size": model.embedding_size,
        "query_prefix": model.query_prefix,
        "document_prefix": model.document_prefix,
        "padding": model.padding,
        "truncation": model.truncation,
        "add_special_tokens": model.add_special_tokens,
        "linear_weights_shape": list(linear_weights.shape),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to {metadata_path}")

    # Verify ONNX model
    try:
        import onnx

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model verification passed")
    except ImportError:
        print("⚠ onnx package not installed, skipping verification")
    except Exception as e:
        print(f"⚠ ONNX model verification failed: {e}")

    # Test inference
    print("\nTesting ONNX inference...")
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(output_path)

        # Test with dummy inputs
        inputs = {
            "input_ids": dummy_input_ids.numpy(),
            "attention_mask": dummy_attention_mask.numpy(),
        }
        outputs = session.run(None, inputs)

        print(f"✓ ONNX inference test passed")
        print(f"  Output shape: {outputs[0].shape}")

    except ImportError:
        print("⚠ onnxruntime package not installed, skipping inference test")
    except Exception as e:
        print(f"⚠ ONNX inference test failed: {e}")

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print("\nFiles created:")
    print(f"  - {output_path} (ONNX model)")
    print(f"  - {linear_path} (linear weights)")
    print(f"  - {metadata_path} (metadata)")
    print("\nNext steps:")
    print("  1. Copy the tokenizer.json file from the model directory")
    print("  2. Use these files with neural-cherche-go ColBERT model")
    print("\nExample usage in Go:")
    print("""
  colbert, err := models.NewColBERT(models.ColBERTConfig{
      ModelPath:         "path/to/colbert.onnx",
      MaxLengthQuery:    64,
      MaxLengthDocument: 256,
      EmbeddingSize:     128,
      LinearWeights:     loadLinearWeights("path/to/colbert_linear.npy"),
  })
""")


def convert_splade_to_onnx(
    model_name_or_path: str,
    output_path: str,
    max_length: int = 256,
    opset_version: int = 14,
):
    """Convert Splade model to ONNX format.

    Args:
        model_name_or_path: HuggingFace model path or local path
        output_path: Output path for ONNX model
        max_length: Maximum sequence length
        opset_version: ONNX opset version
    """
    print(f"Loading Splade model: {model_name_or_path}")
    model = models.Splade(
        model_name_or_path=model_name_or_path,
        device="cpu",
    )

    print("Preparing for ONNX export...")
    model.model.eval()

    # Create dummy inputs
    batch_size = 1
    dummy_input_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
    dummy_attention_mask = torch.ones((batch_size, max_length), dtype=torch.long)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Exporting to ONNX: {output_path}")

    # Export to ONNX
    torch.onnx.export(
        model.model,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    print(f"✓ ONNX model saved to {output_path}")

    # Save metadata
    metadata_path = output_path.replace(".onnx", "_metadata.json")
    metadata = {
        "model_name": model_name_or_path,
        "max_length_query": model.max_length_query,
        "max_length_document": model.max_length_document,
        "query_prefix": model.query_prefix,
        "document_prefix": model.document_prefix,
        "n_mask_tokens": model.n_mask_tokens,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert neural-cherche models to ONNX format"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="raphaelsty/neural-cherche-colbert",
        help="Model name or path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./colbert.onnx",
        help="Output ONNX model path",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="colbert",
        choices=["colbert", "splade"],
        help="Model type",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Neural-Cherche to ONNX Converter")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Type: {args.type}")
    print(f"Output: {args.output}")
    print(f"Max length: {args.max_length}")
    print(f"Opset version: {args.opset_version}")
    print("=" * 60)
    print()

    if args.type == "colbert":
        convert_colbert_to_onnx(
            args.model,
            args.output,
            args.max_length,
            args.opset_version,
        )
    elif args.type == "splade":
        convert_splade_to_onnx(
            args.model,
            args.output,
            args.max_length,
            args.opset_version,
        )
    else:
        raise ValueError(f"Unknown model type: {args.type}")


if __name__ == "__main__":
    main()
