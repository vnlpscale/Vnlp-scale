"""Hardware feasibility and roofline estimates for streamed inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelProfile:
    total_parameters: float
    active_parameters: float
    stored_bits_per_parameter: float
    layers: int
    decoded_bits_per_parameter: float = 16.0

    def __post_init__(self) -> None:
        if self.total_parameters <= 0 or self.active_parameters <= 0:
            raise ValueError("parameter counts must be positive")
        if self.active_parameters > self.total_parameters:
            raise ValueError("active_parameters cannot exceed total_parameters")
        if self.stored_bits_per_parameter <= 0 or self.decoded_bits_per_parameter <= 0:
            raise ValueError("bit widths must be positive")
        if self.layers <= 0:
            raise ValueError("layers must be positive")


@dataclass(frozen=True)
class HardwareProfile:
    storage_bandwidth_gbps: float
    compute_tflops: float
    vram_gb: float
    ram_gb: float
    batch_size: int = 1
    prefetch_depth: int = 1
    storage_capacity_gb: float | None = None
    efficiency: float = 0.70

    def __post_init__(self) -> None:
        if (
            min(
                self.storage_bandwidth_gbps,
                self.compute_tflops,
                self.vram_gb,
                self.ram_gb,
                self.efficiency,
            )
            <= 0
        ):
            raise ValueError("hardware values and efficiency must be positive")
        if self.batch_size <= 0 or self.prefetch_depth < 0:
            raise ValueError("batch_size must be positive and prefetch_depth non-negative")
        if self.efficiency > 1.0:
            raise ValueError("efficiency cannot exceed 1.0")
        if self.storage_capacity_gb is not None and self.storage_capacity_gb <= 0:
            raise ValueError("storage_capacity_gb must be positive")


def plan_inference(model: ModelProfile, hardware: HardwareProfile) -> dict:
    """Return an explicit lower-bound plan for one autoregressive decode step.

    This is a roofline estimate, not a benchmark. It assumes streamed weights are read
    once per batch step and that compute and I/O overlap perfectly, so real throughput
    can only be equal or lower until measured efficiency is calibrated.
    """

    storage_bytes = model.total_parameters * model.stored_bits_per_parameter / 8.0
    transfer_bytes = model.active_parameters * model.stored_bits_per_parameter / 8.0
    decoded_layer_bytes = (
        model.active_parameters / model.layers * model.decoded_bits_per_parameter / 8.0
    )
    full_layer_working_set = decoded_layer_bytes * (1 + hardware.prefetch_depth)
    usable_vram = hardware.vram_gb * 1e9 * 0.80

    effective_bandwidth = hardware.storage_bandwidth_gbps * 1e9 * hardware.efficiency
    effective_flops = hardware.compute_tflops * 1e12 * hardware.efficiency
    io_seconds_per_step = transfer_bytes / effective_bandwidth
    compute_seconds_per_step = 2.0 * model.active_parameters * hardware.batch_size / effective_flops
    step_seconds = max(io_seconds_per_step, compute_seconds_per_step)
    tokens_per_second = hardware.batch_size / step_seconds

    storage_fits = (
        True
        if hardware.storage_capacity_gb is None
        else storage_bytes <= hardware.storage_capacity_gb * 1e9
    )
    layer_fits = full_layer_working_set <= usable_vram
    if not storage_fits:
        execution_mode = "insufficient-storage"
    elif layer_fits:
        execution_mode = "full-layer-streaming"
    else:
        execution_mode = "block-streaming-required"

    ram_cache_fraction = min(1.0, hardware.ram_gb * 1e9 / max(storage_bytes, 1.0))
    return {
        "model": asdict(model),
        "hardware": asdict(hardware),
        "storage_gb": storage_bytes / 1e9,
        "active_transfer_gb_per_step": transfer_bytes / 1e9,
        "decoded_layer_gb": decoded_layer_bytes / 1e9,
        "full_layer_working_set_gb": full_layer_working_set / 1e9,
        "io_seconds_per_step": io_seconds_per_step,
        "compute_seconds_per_step": compute_seconds_per_step,
        "seconds_per_token": step_seconds / hardware.batch_size,
        "tokens_per_second": tokens_per_second,
        "bound": "io" if io_seconds_per_step >= compute_seconds_per_step else "compute",
        "execution_mode": execution_mode,
        "storage_fits": storage_fits,
        "full_layer_fits_vram": layer_fits,
        "ram_cache_fraction": ram_cache_fraction,
        "assumptions": [
            "one read of all active compressed weights per batch decode step",
            "perfect overlap between I/O and compute",
            "KV cache, activations, allocator fragmentation, and codec compute excluded",
            "reported throughput is an optimistic ceiling, not a measured result",
        ],
    }


def scenario_table() -> str:
    scenarios = [
        (
            "1T dense, 1.5 bit, NVMe 7 GB/s",
            ModelProfile(1e12, 1e12, 1.5, 120),
            HardwareProfile(7, 80, 24, 128, efficiency=1.0),
        ),
        (
            "1T dense, 1.5 bit, RAM 80 GB/s",
            ModelProfile(1e12, 1e12, 1.5, 120),
            HardwareProfile(80, 80, 24, 256, efficiency=1.0),
        ),
        (
            "1T MoE, 32B active, 4 bit, PCIe 64 GB/s",
            ModelProfile(1e12, 32e9, 4.0, 120),
            HardwareProfile(64, 80, 24, 128, efficiency=1.0),
        ),
        (
            "1T MoE, 32B active, 1.5 bit, PCIe 64 GB/s",
            ModelProfile(1e12, 32e9, 1.5, 120),
            HardwareProfile(64, 80, 24, 128, efficiency=1.0),
        ),
    ]
    lines = [
        f"{'scenario':<49} {'store':>9} {'xfer/step':>10} {'s/token':>9} {'tok/s':>8} {'mode'}",
        "-" * 112,
    ]
    for label, model, hardware in scenarios:
        result = plan_inference(model, hardware)
        lines.append(
            f"{label:<49} {result['storage_gb']:>7.1f}GB "
            f"{result['active_transfer_gb_per_step']:>8.1f}GB "
            f"{result['seconds_per_token']:>9.3f} "
            f"{result['tokens_per_second']:>8.3f} {result['execution_mode']}"
        )
    return "\n".join(lines)
