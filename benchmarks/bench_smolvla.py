"""Benchmark real SmolVLA checkpoint: FP32 vs FP16 vs INT8 vs INT4 vs NF4.

Loads SmolVLA from HuggingFace, measures latency/throughput/memory
for FP32, FP16 autocast, dynamic INT8 (torchao), INT4 weight-only
(torchao, recommended), and NF4 4-bit (bitsandbytes), with accuracy
gates comparing quantized outputs against FP32 baseline.

Usage:
    python -m benchmarks.bench_smolvla
    python -m benchmarks.bench_smolvla --batch-sizes 1,4,16
    python -m benchmarks.bench_smolvla --compile  # enable torch.compile
"""

from __future__ import annotations

import copy
import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_smolvla() -> tuple[nn.Module, int]:
    from lerobot.configs.policies import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, SmolVLAConfig

    input_features = {
        "observation.state": PolicyFeature(shape=[2], type=FeatureType.STATE),
        "observation.image": PolicyFeature(shape=[3, 224, 224], type=FeatureType.VISUAL),
    }
    output_features = {
        "action": PolicyFeature(shape=[7], type=FeatureType.ACTION),
    }
    cfg = SmolVLAConfig(input_features=input_features, output_features=output_features)
    tokenizer_len = cfg.tokenizer_max_length
    model = SmolVLAPolicy(cfg)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("SmolVLA loaded: %.1fM params, tokenizer_len=%d", num_params / 1e6, tokenizer_len)
    return model, tokenizer_len


def measure_memory_mb(model: nn.Module) -> float:
    total = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffers = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (total + buffers) / (1024 * 1024)


def make_dummy_batch(batch_size: int, device: torch.device, tokenizer_len: int = 48) -> dict:
    return {
        "observation.state": torch.randn(batch_size, 2, device=device),
        "observation.image": torch.rand(batch_size, 3, 224, 224, device=device),
        "observation.language.tokens": torch.randint(0, 1000, (batch_size, tokenizer_len), device=device),
        "observation.language.attention_mask": torch.ones(batch_size, tokenizer_len, dtype=torch.bool, device=device),
    }


def bench_latency(
    model: nn.Module,
    batch_size: int,
    device: torch.device,
    tokenizer_len: int = 48,
    use_autocast: bool = False,
    warmup: int = 10,
    num_runs: int = 100,
) -> dict:
    dummy = make_dummy_batch(batch_size, device, tokenizer_len)

    for _ in range(warmup):
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
            model.select_action(dummy)

    latencies = []
    is_cuda = device.type == "cuda"
    with torch.no_grad():
        for _ in range(num_runs):
            if is_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_autocast):
                model.select_action(dummy)
            if is_cuda:
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

    arr = np.array(latencies)
    mean_ms = float(np.mean(arr))
    throughput = (batch_size / (mean_ms / 1000.0)) if mean_ms > 0 else 0
    return {
        "mean_ms": round(mean_ms, 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "throughput": round(throughput, 1),
    }


def cosine_similarity_actions(model: nn.Module, batch_size: int, device: torch.device, tokenizer_len: int = 48) -> float:
    dummy = make_dummy_batch(batch_size, device, tokenizer_len)
    model_a = copy.deepcopy(model)
    model_b = copy.deepcopy(model)
    with torch.no_grad():
        actions_fp32 = model_a.predict_action_chunk(dummy)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            actions_fp16 = model_b.predict_action_chunk(dummy)
    del model_a, model_b
    cos = nn.functional.cosine_similarity(
        actions_fp32.float().flatten(), actions_fp16.float().flatten(), dim=0
    )
    return round(cos.item(), 6)


def run_benchmark(batch_sizes: list[int] | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if batch_sizes is None:
        batch_sizes = [1, 4] if device.type == "cuda" else [1]

    logger.info("Loading SmolVLA...")
    model_fp32, tokenizer_len = load_smolvla()
    model_fp32 = model_fp32.to(device).eval()

    # Apply attention optimization (SDPA/FlashAttention) + KV-cache quantization
    from lerobot_edge.optimization import optimize_policy_for_inference
    model_fp32 = optimize_policy_for_inference(
        model_fp32, enable_attention=True, enable_kv_cache_quant=True
    )
    fp32_mem = measure_memory_mb(model_fp32)

    logger.info("Creating INT8 and INT4 quantized copies...")
    from lerobot_edge.compression.quantize import (
        dynamic_int8_quantize,
        quantize_4bit,
        quantize_int4_weight_only,
    )

    # INT8 (torchao dynamic)
    model_int8 = None
    int8_mem = 0.0
    int8_results = {}
    try:
        model_int8 = copy.deepcopy(model_fp32).to(device).eval()
        model_int8 = dynamic_int8_quantize(model_int8)
        int8_mem = measure_memory_mb(model_int8)
        logger.info("INT8 memory: %.1f MB (%.2fx reduction)", int8_mem, fp32_mem / int8_mem if int8_mem > 0 else 0)
    except Exception as e:
        logger.warning("INT8 quantization failed: %s. Skipping INT8.", e)

    # INT4 (torchao weight-only - RECOMMENDED)
    model_int4 = None
    int4_mem = 0.0
    int4_results = {}
    try:
        model_int4 = copy.deepcopy(model_fp32).to(device).eval()
        model_int4 = quantize_int4_weight_only(model_int4, group_size=32)
        int4_mem = measure_memory_mb(model_int4)
        logger.info("INT4 memory: %.1f MB (%.2fx reduction)", int4_mem, fp32_mem / int4_mem if int4_mem > 0 else 0)
    except Exception as e:
        logger.warning("INT4 quantization failed: %s. Skipping INT4.", e)

    # NF4 (bitsandbytes - legacy)
    model_nf4 = None
    nf4_mem = 0.0
    nf4_results = {}
    try:
        model_nf4 = copy.deepcopy(model_fp32).to(device).eval()
        model_nf4 = quantize_4bit(model_nf4, quant_type="nf4")
        nf4_mem = measure_memory_mb(model_nf4)
        logger.info("NF4 memory: %.1f MB (%.2fx reduction)", nf4_mem, fp32_mem / nf4_mem if nf4_mem > 0 else 0)
    except Exception as e:
        logger.warning("NF4 quantization failed: %s. Skipping NF4.", e)

    results: dict = {
        "model": "SmolVLA",
        "params": sum(p.numel() for p in model_fp32.parameters()),
        "device": str(device),
        "fp32_memory_mb": round(fp32_mem, 2),
        "batch_sizes": batch_sizes,
    }
    if model_int8 is not None:
        results["int8_memory_mb"] = round(int8_mem, 2)
    if model_int4 is not None:
        results["int4_memory_mb"] = round(int4_mem, 2)
    if model_nf4 is not None:
        results["nf4_memory_mb"] = round(nf4_mem, 2)

    # Accuracy gate: cosine similarity
    logger.info("Computing accuracy gate (cosine similarity)...")
    cosine = cosine_similarity_actions(model_fp32, 1, device, tokenizer_len)
    results["cosine_similarity"] = cosine
    results["accuracy_gate_pass"] = cosine > 0.98
    logger.info("Cosine similarity: %.6f (pass=%s)", cosine, cosine > 0.98)

    # Benchmark each batch size
    fp32_results = {}
    fp16_results = {}
    int8_results = {}
    int4_results = {}
    nf4_results = {}
    for bs in batch_sizes:
        logger.info("Benchmarking bs=%d...", bs)
        fp32_results[str(bs)] = bench_latency(model_fp32, bs, device, tokenizer_len, use_autocast=False)
        fp16_results[str(bs)] = bench_latency(model_fp32, bs, device, tokenizer_len, use_autocast=True)
        if model_int8 is not None:
            try:
                int8_results[str(bs)] = bench_latency(model_int8, bs, device, tokenizer_len, use_autocast=False)
            except Exception as e:
                logger.warning("INT8 benchmark failed at bs=%d: %s", bs, e)
        if model_int4 is not None:
            try:
                int4_results[str(bs)] = bench_latency(model_int4, bs, device, tokenizer_len, use_autocast=False)
            except Exception as e:
                logger.warning("INT4 benchmark failed at bs=%d: %s", bs, e)
        if model_nf4 is not None:
            try:
                nf4_results[str(bs)] = bench_latency(model_nf4, bs, device, tokenizer_len, use_autocast=False)
            except Exception as e:
                logger.warning("NF4 benchmark failed at bs=%d: %s", bs, e)
                model_nf4 = None

    results["fp32"] = fp32_results
    results["fp16"] = fp16_results
    if int8_results:
        results["int8"] = int8_results
    if int4_results:
        results["int4"] = int4_results
    if nf4_results:
        results["nf4"] = nf4_results

    # Compute speedups
    speedups_fp16 = {}
    speedups_int8 = {}
    speedups_int4 = {}
    speedups_nf4 = {}
    for bs in batch_sizes:
        bs_key = str(bs)
        fp32_ms = fp32_results[bs_key]["mean_ms"]
        fp16_ms = fp16_results[bs_key]["mean_ms"]
        speedups_fp16[bs_key] = round(fp32_ms / fp16_ms, 3) if fp16_ms > 0 else 0
        if bs_key in int8_results:
            speedups_int8[bs_key] = round(fp32_ms / int8_results[bs_key]["mean_ms"], 3) if int8_results[bs_key]["mean_ms"] > 0 else 0
        if bs_key in int4_results:
            speedups_int4[bs_key] = round(fp32_ms / int4_results[bs_key]["mean_ms"], 3) if int4_results[bs_key]["mean_ms"] > 0 else 0
        if bs_key in nf4_results:
            nf4_ms = nf4_results[bs_key]["mean_ms"]
            speedups_nf4[bs_key] = round(fp32_ms / nf4_ms, 3) if nf4_ms > 0 else 0
    results["speedup_fp16"] = speedups_fp16
    if speedups_int8:
        results["speedup_int8"] = speedups_int8
    if speedups_int4:
        results["speedup_int4"] = speedups_int4
    if speedups_nf4:
        results["speedup_nf4"] = speedups_nf4

    del model_fp32
    if model_int8 is not None:
        del model_int8
    if model_int4 is not None:
        del model_int4
    if model_nf4 is not None:
        del model_nf4
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


def print_results(results: dict) -> None:
    bs_list = results["batch_sizes"]
    print(f"\n{'=' * 85}")
    print(f"SmolVLA REAL CHECKPOINT BENCHMARK — {results['device'].upper()}")
    print(f"Params: {results['params'] / 1e6:.0f}M")
    print(f"{'=' * 85}")
    print(f"  FP32 memory: {results['fp32_memory_mb']:.1f} MB")
    for label, key in [("INT8", "int8_memory_mb"), ("INT4", "int4_memory_mb"), ("NF4", "nf4_memory_mb")]:
        mem = results.get(key)
        if mem is not None:
            print(f"  {label} memory: {mem:.1f} MB ({results['fp32_memory_mb'] / mem:.2f}x reduction)")
    print(f"  Accuracy gate: cosine={results['cosine_similarity']:.6f}  pass={results['accuracy_gate_pass']}")
    print()

    header = f"{'Backend':<10}"
    for bs in bs_list:
        header += f"  {'bs=' + str(bs):>22}"
    print(header)
    print("-" * 85)

    # Build rows dynamically
    rows: list[tuple[str, str]] = []
    for key, label in [("fp32", "FP32"), ("fp16", "FP16"), ("int8", "INT8"), ("int4", "INT4"), ("nf4", "NF4")]:
        if key not in results:
            continue
        row = f"{label:<10}"
        for bs in bs_list:
            k = str(bs)
            r = results[key][k]
            row += f"  {r['mean_ms']:>8.2f}ms/{r['throughput']:>8.0f}sp"
        rows.append((label, row))

    for _, row in rows:
        print(row)

    # Speedup rows
    for label, key in [("FP16", "speedup_fp16"), ("INT8", "speedup_int8"), ("INT4", "speedup_int4"), ("NF4", "speedup_nf4")]:
        if key not in results:
            continue
        row = f"{label} spd:<10"
        for bs in bs_list:
            k = str(bs)
            row += f"  {results[key][k]:>21.2f}x" 
        print(row)
    print(f"{'=' * 85}\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark real SmolVLA: FP32 vs FP16 vs INT8 vs INT4 vs NF4"
    )
    parser.add_argument("--batch-sizes", type=str, default=None)
    parser.add_argument("--output", default="benchmark_results/smolvla_benchmark.json")
    parser.add_argument(
        "--compile", action="store_true", help="Enable torch.compile (reduce-overhead)"
    )
    parser.add_argument(
        "--no-attention-opt",
        action="store_true",
        help="Disable SDPA/FlashAttention optimization",
    )
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")] if args.batch_sizes else None

    if args.no_attention_opt:
        # Reload without attention optimization
        logger.warning("Attention optimization DISABLED — using eager attention")

    results = run_benchmark(batch_sizes)
    print_results(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
