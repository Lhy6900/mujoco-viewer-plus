#!/usr/bin/env python3
"""
test_onnx_policy.py

A reusable utility to inspect and benchmark ONNX models.

Usage:
    python test_onnx_policy.py path/to/model.onnx [--runs 10] [--gpu]

Requirements:
    pip install onnx onnxruntime numpy
"""

import argparse
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


def print_model_info(model_path):
    print("=" * 60)
    print(f"🔍 Inspecting model: {model_path}")
    print("=" * 60)

    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print("✅ Model is valid.\n")

    # --- Metadata ---
    print("📘 Model Metadata Properties:")
    if model.metadata_props:
        for prop in model.metadata_props:
            print(f"  {prop.key}: {prop.value}")
    else:
        print("  (none)")

    print("\n📦 Inputs:")
    for i, inp in enumerate(model.graph.input):
        shape = [d.dim_value if d.dim_value != 0 else "?" for d in inp.type.tensor_type.shape.dim]
        dtype = inp.type.tensor_type.elem_type
        print(f"  [{i}] {inp.name}: shape={shape}, dtype={dtype}")

    print("\n📤 Outputs:")
    for i, out in enumerate(model.graph.output):
        shape = [d.dim_value if d.dim_value != 0 else "?" for d in out.type.tensor_type.shape.dim]
        dtype = out.type.tensor_type.elem_type
        print(f"  [{i}] {out.name}: shape={shape}, dtype={dtype}")

    print("🔹 Initializers:")
    for init in model.graph.initializer:
        print(f"  {init.name}: shape={[d for d in init.dims]}")

    print("\n🔹 Constant nodes:")
    for node in model.graph.node:
        if node.op_type == "Constant":
            tensor_attr = next((attr for attr in node.attribute if attr.type == onnx.AttributeProto.TENSOR), None)
            if tensor_attr:
                tensor = numpy_helper.to_array(tensor_attr.t)
                print(f"  {node.name or '(anonymous)'}: shape={tensor.shape}, dtype={tensor.dtype}")

    print("\n⚙️  Total nodes in graph:", len(model.graph.node))
    print("⚙️  Number of parameters:", len(model.graph.initializer))
    print()


def benchmark_inference(model_path, runs=10, use_gpu=False):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    print("=" * 60)
    print(f"🚀 Running inference benchmark ({runs} runs)")
    print(f"Using providers: {providers}")
    print("=" * 60)

    # Create session
    session = ort.InferenceSession(model_path, providers=providers)
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    # Prepare dummy inputs
    feed = {}
    for inp in inputs:
        shape = [s if isinstance(s, int) and s > 0 else 1 for s in inp.shape]
        feed[inp.name] = np.random.randn(*shape).astype(np.float32)

    # Warm-up
    session.run(None, feed)

    # Benchmark
    start = time.time()
    for _ in range(runs):
        _ = session.run(None, feed)
    end = time.time()

    avg = (end - start) / runs
    print(f"✅ Inference completed.")
    print(f"🕒 Average inference time: {avg * 1000:.2f} ms per run\n")

    # Print example output
    for i, o in enumerate(outputs):
        print(f"Output[{i}] name={o.name}, shape={_ [i].shape}, dtype={_ [i].dtype}")


def main():
    parser = argparse.ArgumentParser(description="Inspect and benchmark an ONNX model.")
    parser.add_argument("model", type=str, help="Path to the .onnx model file")
    parser.add_argument("--runs", type=int, default=10, help="Number of inference runs")
    parser.add_argument("--gpu", action="store_true", help="Use CUDAExecutionProvider if available")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate")
    args = parser.parse_args()

    print_model_info(args.model)
    benchmark_inference(args.model, runs=args.runs, use_gpu=args.gpu)


if __name__ == "__main__":
    main()
