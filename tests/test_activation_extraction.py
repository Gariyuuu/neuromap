"""Tests the forward-hook activation-extraction + pooling mechanics in neuromap.models
against a tiny synthetic network (no pretrained weights downloaded)."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuromap.models import ModelSpec, extract_layer_activations, _pool


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def _tiny_cnn_loader():
    model = TinyCNN().eval()
    return model, {"conv1": model.conv1, "conv2": model.conv2}


def test_pool_averages_spatial_dims():
    x = torch.ones(2, 4, 8, 8) * 3.0
    pooled = _pool(x)
    assert pooled.shape == (2, 4)
    assert torch.allclose(pooled, torch.full((2, 4), 3.0))


def test_pool_averages_token_dim():
    x = torch.ones(2, 10, 6) * 5.0
    pooled = _pool(x)
    assert pooled.shape == (2, 6)
    assert torch.allclose(pooled, torch.full((2, 6), 5.0))


def test_extract_layer_activations_shapes():
    spec = ModelSpec("tiny_cnn", "classical", "none", "none", _tiny_cnn_loader, ["conv1", "conv2"])
    images = torch.randn(6, 3, 16, 16)
    acts = extract_layer_activations(spec, images, batch_size=4)
    assert acts["conv1"].shape == (6, 4)
    assert acts["conv2"].shape == (6, 8)


def test_extract_layer_activations_deterministic():
    """Batch size must not change the extracted activations for a fixed model."""
    torch.manual_seed(0)
    shared_model = TinyCNN().eval()

    def fixed_loader():
        return shared_model, {"conv1": shared_model.conv1, "conv2": shared_model.conv2}

    spec = ModelSpec("tiny_cnn", "classical", "none", "none", fixed_loader, ["conv1"])
    images = torch.randn(4, 3, 16, 16)
    acts_a = extract_layer_activations(spec, images, batch_size=4)
    acts_b = extract_layer_activations(spec, images, batch_size=2)  # different batching
    assert torch.allclose(torch.from_numpy(acts_a["conv1"]), torch.from_numpy(acts_b["conv1"]), atol=1e-5)
