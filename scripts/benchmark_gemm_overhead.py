#!/usr/bin/env python3
"""Measure large CUDA GEMM throughput and end-to-end launch time."""

import argparse
import json
import math
import statistics
import time



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=32768)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--label", default="benchmark")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    torch.manual_seed(0)
    device = torch.device("cuda")
    a = torch.randn((args.size, args.size), device=device, dtype=torch.float16)
    b = torch.randn((args.size, args.size), device=device, dtype=torch.float16)

    for _ in range(args.warmup):
        torch.mm(a, b)
    torch.cuda.synchronize()

    samples_ms = []
    for _ in range(args.iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start_event.record()
        out = torch.mm(a, b)
        end_event.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        samples_ms.append((start_event.elapsed_time(end_event), wall_ms))

    gpu_ms = [sample[0] for sample in samples_ms]
    wall_ms = [sample[1] for sample in samples_ms]
    flops = 2.0 * args.size**3
    result = {
        "label": args.label,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dtype": "float16",
        "size": args.size,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "gpu_ms_median": statistics.median(gpu_ms),
        "gpu_ms_mean": statistics.mean(gpu_ms),
        "wall_ms_median": statistics.median(wall_ms),
        "wall_ms_mean": statistics.mean(wall_ms),
        "wall_ms_p95": sorted(wall_ms)[math.ceil(0.95 * len(wall_ms)) - 1],
        "wall_ms_stdev": statistics.stdev(wall_ms) if len(wall_ms) > 1 else 0.0,
        "tflops_from_wall_median": flops / (statistics.median(wall_ms) / 1000.0) / 1e12,
        "checksum": float(out[0, 0]),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
