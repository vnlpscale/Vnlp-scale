"""Measure store decode latency for one tensor and emit machine-readable JSON."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from vnlp_scale import __version__
from vnlp_scale.store import StoreReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-stages", type=int)
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    if arguments.iterations <= 0:
        parser.error("--iterations must be positive")

    with StoreReader(arguments.store, verify_on_open=arguments.verify) as reader:
        info = reader.info(arguments.tensor)
        timings = []
        checksum = 0.0
        decoded_bytes = int(np.prod(info["shape"])) * np.dtype(np.float32).itemsize
        for _ in range(arguments.iterations):
            started = time.perf_counter()
            values = reader.decode(arguments.tensor, max_stages=arguments.max_stages)
            timings.append(time.perf_counter() - started)
            checksum += float(values.reshape(-1)[0])
        payload = {
            "benchmark": "tensor-decode",
            "vnlp_scale_version": __version__,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "store": str(arguments.store.resolve()),
            "store_format": reader.manifest["version"],
            "tensor": arguments.tensor,
            "shape": info["shape"],
            "chunks": len(info["chunks"]),
            "decoded_bytes": decoded_bytes,
            "max_stages": arguments.max_stages,
            "iterations": arguments.iterations,
            "first_seconds": timings[0],
            "warm_median_seconds": float(np.median(timings[1:] or timings)),
            "warm_decoded_gb_per_second": decoded_bytes
            / max(float(np.median(timings[1:] or timings)), 1e-30)
            / 1e9,
            "all_seconds": timings,
            "checksum_guard": checksum,
            "note": "first run is not guaranteed to be a cold OS-page-cache measurement",
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
