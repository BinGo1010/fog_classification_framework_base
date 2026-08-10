import numpy as np
import torch

import scripts.run_daphnet_nbm300_c_vs_raw_ablation as ablation
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import RobustScaler
from scripts.run_daphnet_transformer_nbm300_fold import PatchTransformerNBM


def test_patchify_and_fold_are_exact_inverses() -> None:
    x = torch.arange(2 * 9 * 128, dtype=torch.float32).reshape(2, 9, 128)
    patches = PatchTransformerNBM.patchify(x)
    assert patches.shape == (2, 16, 72)
    restored = PatchTransformerNBM.fold_patches(patches)
    assert torch.equal(restored, x)


def test_transformer_nbm_requested_shapes_and_contract() -> None:
    model = PatchTransformerNBM(dropout=0.10).eval()
    x = torch.zeros(1, 9, 128)
    with torch.no_grad():
        encoded_tokens = model.encode_tokens(x)
        z = model.encode(x)
        reconstruction = model(x)
    assert encoded_tokens.shape == (1, 16, 192)
    assert z.shape == (1, 8, 64)
    assert reconstruction.shape == x.shape
    assert torch.isfinite(reconstruction).all()

    config = model.architecture_config()
    assert config["patchify"]["patch_size"] == 8
    assert config["encoder"] == {
        "layers": 4,
        "d_model": 192,
        "heads": 6,
        "ffn": 576,
        "activation": "GELU",
        "dropout": 0.10,
        "normalization": "PyTorch post-norm TransformerEncoderLayer",
    }
    assert config["bottleneck_shape"] == ["B", 8, 64]
    assert config["decoder"]["layers"] == 2
    assert config["encoder_decoder_skip_connections"] is False
    assert config["parameter_count"] == 2_329_736


def test_transformer_branch_generates_scheme_c_27_channels(monkeypatch) -> None:
    model = PatchTransformerNBM(dropout=0.10)
    scaler = RobustScaler(
        median=np.zeros(9, dtype=np.float32),
        iqr=np.ones(9, dtype=np.float32),
    )
    monkeypatch.setattr(
        ablation,
        "load_frozen_transformer_nbm",
        lambda _source, _fold, _device: (
            model,
            scaler,
            np.zeros(9, dtype=np.float32),
            np.ones(9, dtype=np.float32),
            {"checkpoint": "test"},
        ),
    )
    raw = np.zeros((2, 128, 9), dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int8)
    features, metadata = ablation.make_features(
        "FULL_C",
        scaler,
        raw,
        labels,
        torch.device("cpu"),
        None,
        0,
        "transformer",
        128,
    )
    assert features.shape == (2, 128, 27)
    assert np.isfinite(features).all()
    assert metadata["nbm_kind"] == "transformer"
    assert metadata["uses_sigma"] is True
