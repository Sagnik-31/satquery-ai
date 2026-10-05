from __future__ import annotations

import pytest
import torch

from ml.models.siamese_unet import SiameseUNet


def test_siamese_unet_returns_one_full_resolution_logit_channel():
    model = SiameseUNet(base_channels=4)
    output = model(torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 1, 32, 32)


def test_siamese_unet_encoder_is_one_shared_module_for_both_dates():
    model = SiameseUNet(base_channels=4)
    first_weight = model.encoder.stem[0].weight
    features_t1 = model.encoder(torch.randn(1, 3, 32, 32))
    features_t2 = model.encoder(torch.randn(1, 3, 32, 32))
    assert first_weight is model.encoder.stem[0].weight
    assert len(features_t1) == len(features_t2) == 5
    assert len([name for name, _ in model.named_modules() if name == "encoder"]) == 1


@pytest.mark.parametrize("size", [32, 48, 64])
def test_siamese_unet_supports_sizes_divisible_by_downsampling_factor(size):
    model = SiameseUNet(base_channels=4).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, size, size), torch.zeros(1, 3, size, size))
    assert output.shape == (1, 1, size, size)


@pytest.mark.parametrize(
    ("t1", "t2", "message"),
    [
        (torch.zeros(1, 3, 32, 32), torch.zeros(1, 3, 16, 32), "identical shapes"),
        (torch.zeros(1, 4, 32, 32), torch.zeros(1, 4, 32, 32), "Expected 3 input channels"),
        (torch.zeros(1, 3, 30, 32), torch.zeros(1, 3, 30, 32), "divisible by 16"),
    ],
)
def test_siamese_unet_rejects_invalid_inputs(t1, t2, message):
    with pytest.raises(ValueError, match=message):
        SiameseUNet(base_channels=4)(t1, t2)


def test_siamese_unet_cpu_forward_pass():
    model = SiameseUNet(base_channels=4).cpu().eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32))
    assert output.device.type == "cpu"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable on this host")
def test_siamese_unet_mps_smoke_forward_pass():
    device = torch.device("mps")
    model = SiameseUNet(base_channels=4).to(device).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 32, 32, device=device), torch.randn(1, 3, 32, 32, device=device))
    assert output.shape == (1, 1, 32, 32)
    assert output.device.type == "mps"
