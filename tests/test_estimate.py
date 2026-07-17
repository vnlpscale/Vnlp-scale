import pytest

from vnlp_scale.estimate import HardwareProfile, ModelProfile, plan_inference


def test_one_trillion_dense_nvme_roofline():
    result = plan_inference(
        ModelProfile(1e12, 1e12, 1.5, 120),
        HardwareProfile(7, 80, 24, 128, efficiency=1.0),
    )
    assert result["storage_gb"] == pytest.approx(187.5)
    assert result["io_seconds_per_step"] == pytest.approx(187.5 / 7)
    assert result["tokens_per_second"] == pytest.approx(7 / 187.5)
    assert result["bound"] == "io"
    assert result["execution_mode"] == "block-streaming-required"


def test_moe_active_parameters_reduce_transfer():
    dense = plan_inference(
        ModelProfile(1e12, 1e12, 4, 100),
        HardwareProfile(64, 100, 48, 256, efficiency=1.0),
    )
    moe = plan_inference(
        ModelProfile(1e12, 32e9, 4, 100),
        HardwareProfile(64, 100, 48, 256, efficiency=1.0),
    )
    assert moe["active_transfer_gb_per_step"] < dense["active_transfer_gb_per_step"]
    assert moe["tokens_per_second"] > dense["tokens_per_second"]
