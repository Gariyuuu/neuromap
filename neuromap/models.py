"""Registry of artificial vision models compared against neural data, and a generic
forward-hook activation extractor. Each layer's output is global-average-pooled (CNN
spatial maps) or mean-pooled over tokens (ViT), collapsing it to one feature vector per
image — done to keep dimensionality tractable for the linear encoding models used
downstream (H: overly flexible mappings can obscure representation quality)."""
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torchvision.models as tvm


@dataclass
class ModelSpec:
    name: str
    family: str  # "classical" | "cnn_supervised" | "vit_supervised" | "self_supervised"
    task_objective: str  # e.g. "imagenet_classification", "self_supervised_dino", "none"
    weights_id: str
    loader: callable
    layer_names: list = field(default_factory=list)


def _resnet18():
    model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    layers = {
        "layer1": model.layer1, "layer2": model.layer2,
        "layer3": model.layer3, "layer4": model.layer4,
        "avgpool": model.avgpool,
    }
    return model, layers


def _resnet50():
    model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    layers = {
        "layer1": model.layer1, "layer2": model.layer2,
        "layer3": model.layer3, "layer4": model.layer4,
        "avgpool": model.avgpool,
    }
    return model, layers


def _vit_b_16():
    model = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
    model.eval()
    blocks = model.encoder.layers
    layers = {
        "block2": blocks[2], "block5": blocks[5],
        "block8": blocks[8], "block11": blocks[11],
    }
    return model, layers


def _dino_vits16():
    model = torch.hub.load("facebookresearch/dino:main", "dino_vits16", pretrained=True)
    model.eval()
    blocks = model.blocks
    layers = {
        "block2": blocks[2], "block5": blocks[5],
        "block8": blocks[8], "block11": blocks[11],
    }
    return model, layers


REGISTRY = {
    "resnet18": ModelSpec("resnet18", "cnn_supervised", "imagenet_classification",
                           "IMAGENET1K_V1", _resnet18,
                           ["layer1", "layer2", "layer3", "layer4", "avgpool"]),
    "resnet50": ModelSpec("resnet50", "cnn_supervised", "imagenet_classification",
                           "IMAGENET1K_V2", _resnet50,
                           ["layer1", "layer2", "layer3", "layer4", "avgpool"]),
    "vit_b_16": ModelSpec("vit_b_16", "vit_supervised", "imagenet_classification",
                           "IMAGENET1K_V1", _vit_b_16,
                           ["block2", "block5", "block8", "block11"]),
    "dino_vits16": ModelSpec("dino_vits16", "self_supervised", "self_supervised_dino",
                              "dino_deitsmall16_pretrain", _dino_vits16,
                              ["block2", "block5", "block8", "block11"]),
}


def _pool(output: torch.Tensor) -> torch.Tensor:
    if output.dim() == 4:  # CNN feature map: (B, C, H, W)
        return output.mean(dim=(2, 3))
    if output.dim() == 3:  # transformer tokens: (B, T, D)
        return output.mean(dim=1)
    return output  # already (B, D), e.g. avgpool output pre-flatten handled by caller


@torch.no_grad()
def extract_layer_activations(spec: ModelSpec, image_batch: torch.Tensor, batch_size: int = 32,
                               device: str = "cpu") -> dict:
    """Returns {layer_name: np.ndarray (n_images, n_features)}."""
    model, layer_modules = spec.loader()
    model = model.to(device)

    captured = {name: [] for name in spec.layer_names}
    handles = []

    def make_hook(name):
        def hook(module, inp, out):
            pooled = _pool(out).flatten(1)
            captured[name].append(pooled.detach().cpu())
        return hook

    for name in spec.layer_names:
        handles.append(layer_modules[name].register_forward_hook(make_hook(name)))

    for i in range(0, image_batch.shape[0], batch_size):
        batch = image_batch[i : i + batch_size].to(device)
        model(batch)

    for h in handles:
        h.remove()

    return {name: torch.cat(chunks, dim=0).numpy() for name, chunks in captured.items()}
