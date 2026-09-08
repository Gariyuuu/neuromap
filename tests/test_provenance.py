import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuromap.provenance import hash_array, hash_config


def test_hash_array_deterministic():
    arr = np.arange(100).reshape(10, 10)
    assert hash_array(arr) == hash_array(arr.copy())


def test_hash_array_sensitive_to_changes():
    arr = np.arange(100).reshape(10, 10)
    arr2 = arr.copy()
    arr2[0, 0] = 999
    assert hash_array(arr) != hash_array(arr2)


def test_hash_config_order_independent():
    a = {"x": 1, "y": 2}
    b = {"y": 2, "x": 1}
    assert hash_config(a) == hash_config(b)


def test_result_parquet_roundtrip_preserves_list_columns():
    df = pd.DataFrame({
        "model": ["resnet18", "vit_b_16"],
        "per_neuron_r": [[0.1, 0.2, 0.3], [0.4, 0.5]],
        "mean_r": [0.2, 0.45],
    })
    path = Path("/tmp") / "neuromap_test_roundtrip.parquet"
    df.to_parquet(path)
    loaded = pd.read_parquet(path)
    assert loaded["mean_r"].tolist() == df["mean_r"].tolist()
    assert list(loaded["per_neuron_r"].iloc[0]) == df["per_neuron_r"].iloc[0]
    path.unlink()
