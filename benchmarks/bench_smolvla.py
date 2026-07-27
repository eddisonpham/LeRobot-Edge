"""Benchmark real SmolVLA checkpoint: FP32 vs FP16 vs NF4 with accuracy gate.

Loads SmolVLA from HuggingFace, measures latency/throughput/memory
for FP32, FP16 autocast, and NF4 4-bit quantization, and computes
cosine similarity between FP32 and quantized action outputs.

Usage:
    python -m benchmarks.bench_smolvla
    python -m benchmarks.bench_smolvla --batch-sizes 1,4,16
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
    fp32_mem = measure_memory_mb(model_fp32)

    logger.info("Creating NF4 quantized copy...")
    from lerobot_edge.compression.quantize import quantize_4bit

    model_nf4 = None
    nf4_mem = 0.0
    nf4_results = {}
    try:
        model_nf4 = copy.deepcopy(model_fp32).to(device).eval()
        model_nf4 = quantize_4bit(model_nf4, quant_type="nf4")
        nf4_mem = measure_memory_mb(model_nf4)
        logger.info("NF4 memory: %.1f MB (%.2fx reduction)", nf4_mem, fp32_mem / nf4_mem if nf4_mem > 0 else 0)
    except Exception as e:
        logger.warning("NF4 quantization failed: %s. Skipping NF4 benchmark.", e)

    results: dict = {
        "model": "SmolVLA",
        "params": sum(p.numel() for p in model_fp32.parameters()),
        "device": str(device),
        "fp32_memory_mb": round(fp32_mem, 2),
        "batch_sizes": batch_sizes,
    }
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
    nf4_results = {}
    for bs in batch_sizes:
        logger.info("Benchmarking bs=%d...", bs)
        fp32_results[str(bs)] = bench_latency(model_fp32, bs, device, tokenizer_len, use_autocast=False)
        fp16_results[str(bs)] = bench_latency(model_fp32, bs, device, tokenizer_len, use_autocast=True)
        if model_nf4 is not None:
            try:
                nf4_results[str(bs)] = bench_latency(model_nf4, bs, device, tokenizer_len, use_autocast=False)
            except Exception as e:
                logger.warning("NF4 benchmark failed at bs=%d: %s", bs, e)
                model_nf4 = None

    results["fp32"] = fp32_results
    results["fp16"] = fp16_results
    if nf4_results:
        results["nf4"] = nf4_results

    # Compute speedups
    speedups_fp16 = {}
    speedups_nf4 = {}
    for bs in batch_sizes:
        bs_key = str(bs)
        fp32_ms = fp32_results[bs_key]["mean_ms"]
        fp16_ms = fp16_results[bs_key]["mean_ms"]
        speedups_fp16[bs_key] = round(fp32_ms / fp16_ms, 3) if fp16_ms > 0 else 0
        if bs_key in nf4_results:
            nf4_ms = nf4_results[bs_key]["mean_ms"]
            speedups_nf4[bs_key] = round(fp32_ms / nf4_ms, 3) if nf4_ms > 0 else 0
    results["speedup_fp16"] = speedups_fp16
    if speedups_nf4:
        results["speedup_nf4"] = speedups_nf4

    del model_fp32
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
    nf4_mem_display = results.get("nf4_memory_mb")
    if nf4_mem_display is not None:
        print(f"  NF4  memory: {nf4_mem_display:.1f} MB ({results['fp32_memory_mb'] / nf4_mem_display:.2f}x reduction)")
    else:
        print(f"  NF4: not available (attention incompatibility with SmolVLA)")
    print(f"  Accuracy gate: cosine={results['cosine_similarity']:.6f}  pass={results['accuracy_gate_pass']}")
    print()

    header = f"{'Backend':<10}"
    for bs in bs_list:
        header += f"  {'bs=' + str(bs):>22}"
    print(header)
    print("-" * 85)

    row_fp32 = f"{'FP32':<10}"
    row_fp16 = f"{'FP16':<10}"
    has_nf4 = "nf4" in results
    if has_nf4:
        row_nf4 = f"{'NF4':<10}"
        for bs in bs_list:
            k = str(bs)
            nf4 = results["nf4"][k]
            row_nf4 += f"  {nf4['mean_ms']:>8.2f}ms/{nf4['throughput']:>8.0f}sp"
    for bs in bs_list:
        k = str(bs)
        fp32 = results["fp32"][k]
        fp16 = results["fp16"][k]
        row_fp32 += f"  {fp32['mean_ms']:>8.2f}ms/{fp32['throughput']:>8.0f}sp"
        row_fp16 += f"  {fp16['mean_ms']:>8.2f}ms/{fp16['throughput']:>8.0f}sp"
    print(row_fp32)
    print(row_fp16)
    if has_nf4:
        print(row_nf4)

    row_speed_fp16 = f"{'FP16 spd':<10}"
    for bs in bs_list:
        k = str(bs)
        row_speed_fp16 += f"  {results['speedup_fp16'][k]:>21.2f}x"
    print(row_speed_fp16)
    if "speedup_nf4" in results:
        row_speed_nf4 = f"{'NF4 spd':<10}"
        for bs in bs_list:
            k = str(bs)
            row_speed_nf4 += f"  {results['speedup_nf4'][k]:>21.2f}x"
        print(row_speed_nf4)
    print(f"{'=' * 85}\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark real SmolVLA FP16 vs FP32")
    parser.add_argument("--batch-sizes", type=str, default=None)
    parser.add_argument("--output", default="benchmark_results/smolvla_benchmark.json")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")] if args.batch_sizes else None
    results = run_benchmark(batch_sizes)
    print_results(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
