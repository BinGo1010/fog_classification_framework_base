from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
import torch
from torch import nn

from scripts.run_daphnet_gru_v15_nbm300_fold import (
    ARCHITECTURE_NAME,
    PARAMETER_COUNT,
    GRUV15Decoder96NBM,
    calibrate_gru_v15,
    parse_csv_ints,
    reconstruct_gru_v15,
    write_role4_scaler_artifact,
)
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import RobustScaler


def test_gru_v15_shapes_parameter_count_and_frozen_architecture() -> None:
    model = GRUV15Decoder96NBM().eval()
    x = torch.zeros(3, 128, 9)
    with torch.no_grad():
        z = model.encode(x)
        reconstruction = model(x)

    assert z.shape == (3, 16)
    assert reconstruction.shape == x.shape
    assert torch.isfinite(reconstruction).all()
    assert sum(parameter.numel() for parameter in model.parameters()) == 48_761
    assert PARAMETER_COUNT == 48_761

    config = model.architecture_config()
    assert config["name"] == ARCHITECTURE_NAME
    assert config["encoder"] == {
        "type": "unidirectional GRU",
        "layers": 1,
        "input_size": 9,
        "hidden": 64,
        "dropout": 0.0,
        "summary": "last hidden state [B,64]",
    }
    assert config["latent_shape"] == ["B", 16]
    assert config["encoder_gru"]["hidden_size"] == 64
    assert config["decoder_gru"]["hidden_size"] == 96
    assert config["decoder"]["hidden"] == 96
    assert config["decoder"]["layers"] == 1
    assert config["decoder_conditioning"]["per_step_input"].startswith(
        "128-step all-zero"
    )
    assert config["encoder_decoder_skip_connections"] is False
    assert config["skip_connections"] is False
    assert config["teacher_forcing"] is False
    assert config["fourier_or_positional_code"] is False
    assert config["time_code"] is False
    assert config["normalization"] is None
    assert config["output_activation"] is None


def test_gru_v15_contains_only_requested_trainable_layers() -> None:
    model = GRUV15Decoder96NBM()
    assert isinstance(model.encoder, nn.GRU)
    assert model.encoder.input_size == 9
    assert model.encoder.hidden_size == 64
    assert model.encoder.num_layers == 1
    assert model.encoder.bidirectional is False
    assert isinstance(model.to_bottleneck, nn.Linear)
    assert (model.to_bottleneck.in_features, model.to_bottleneck.out_features) == (
        64,
        16,
    )
    assert isinstance(model.to_decoder_hidden, nn.Linear)
    assert (
        model.to_decoder_hidden.in_features,
        model.to_decoder_hidden.out_features,
    ) == (16, 96)
    assert isinstance(model.decoder, nn.GRU)
    assert model.decoder.input_size == 9
    assert model.decoder.hidden_size == 96
    assert model.decoder.num_layers == 1
    assert model.decoder.bidirectional is False
    assert isinstance(model.output, nn.Linear)
    assert (model.output.in_features, model.output.out_features) == (96, 9)

    forbidden = (nn.Dropout, nn.LayerNorm, nn.BatchNorm1d, nn.Tanh)
    assert not any(isinstance(module, forbidden) for module in model.modules())
    assert len(list(model.children())) == 5


def test_gru_v15_zero_input_decoder_has_no_teacher_forcing() -> None:
    torch.manual_seed(17)
    model = GRUV15Decoder96NBM().eval()
    captured: dict[str, torch.Tensor] = {}

    def capture_decoder_input(
        _module: nn.Module,
        arguments: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        captured["sequence"] = arguments[0].detach().clone()

    handle = model.decoder.register_forward_pre_hook(capture_decoder_input)
    try:
        with torch.no_grad():
            output = model(torch.randn(2, 128, 9))
    finally:
        handle.remove()
    assert output.shape == (2, 128, 9)
    assert torch.count_nonzero(captured["sequence"]) == 0


def test_gru_v15_state_roundtrip_is_exact(tmp_path) -> None:
    torch.manual_seed(7)
    source = GRUV15Decoder96NBM().eval()
    x = torch.randn(2, 128, 9)
    with torch.no_grad():
        expected = source(x)
    checkpoint = tmp_path / "gru_v15.pt"
    torch.save(
        {
            "model_state": source.state_dict(),
            "architecture": source.architecture_config(),
        },
        checkpoint,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored = GRUV15Decoder96NBM().eval()
    restored.load_state_dict(payload["model_state"])
    with torch.no_grad():
        actual = restored(x)
    assert payload["architecture"] == restored.architecture_config()
    assert torch.equal(actual, expected)


def test_gru_v15_reconstruction_and_role5_mad_calibration() -> None:
    model = GRUV15Decoder96NBM().eval()
    x = np.zeros((3, 128, 9), dtype=np.float32)
    reconstruction = reconstruct_gru_v15(model, x, torch.device("cpu"), 2)
    assert reconstruction.shape == x.shape
    assert reconstruction.dtype == np.float32
    bias, sigma, metadata = calibrate_gru_v15(model, x, torch.device("cpu"))
    assert bias.shape == sigma.shape == (9,)
    assert np.isfinite(bias).all() and np.isfinite(sigma).all()
    assert np.all(sigma >= 0.05)
    assert metadata["calibration_windows"] == 3
    assert metadata["sigma_floor"] == 0.05


def test_gru_v15_input_and_seed_contracts() -> None:
    model = GRUV15Decoder96NBM()
    with pytest.raises(ValueError, match=r"expected \[B,128,9\]"):
        model(torch.zeros(2, 64, 9))
    with pytest.raises(ValueError, match="frozen"):
        GRUV15Decoder96NBM(decoder_hidden=64)
    with pytest.raises(ValueError, match="empty"):
        reconstruct_gru_v15(
            model,
            np.empty((0, 128, 9), dtype=np.float32),
            torch.device("cpu"),
        )
    assert parse_csv_ints("0,52,161,5216,52161") == (0, 52, 161, 5216, 52161)
    with pytest.raises(ValueError, match="unique"):
        parse_csv_ints("0,52,52")


def test_gru_v15_worker_uses_atomic_checkpoint_and_clean_validation() -> None:
    # Static safeguards make accidental regression visible without a slow fit.
    from scripts.run_daphnet_gru_v15_nbm300_fold import train_gru_v15_nbm

    source = inspect.getsource(train_gru_v15_nbm)
    assert "atomic_torch_save(" in source
    validation_source = source.split("with torch.no_grad():", 1)[1]
    assert "corrupt(clean" not in validation_source.split("train_loss =", 1)[0]
    assert '"gru_v15_nbm_best.pt"' in source


def test_gru_v15_writes_independent_atomic_role4_scaler(tmp_path) -> None:
    scaler = RobustScaler(
        median=np.arange(9, dtype=np.float32),
        iqr=np.arange(1, 10, dtype=np.float32),
    )
    expected = write_role4_scaler_artifact(
        tmp_path,
        fold=2,
        seed=52161,
        scaler=scaler,
        unique_raw_points=12345,
        scientific_data_sha256="a" * 64,
    )
    path = tmp_path / "scaler_role4.json"
    assert path.is_file()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == expected
    assert loaded["fold"] == 2
    assert loaded["seed"] == 52161
    assert loaded["scaler_fit_role"] == 4
    assert loaded["scaler_unique_raw_points"] == 12345
    assert loaded["scientific_data_sha256"] == "a" * 64
    assert set(loaded["scaler"]) == {"median", "iqr", "epsilon"}
    forbidden = {"bias", "b", "sigma", "sigma_raw", "calibration"}
    assert forbidden.isdisjoint(loaded)
    assert not list(tmp_path.glob(".scaler_role4.json.tmp-*"))
    assert "atomic_json_dump(" in inspect.getsource(write_role4_scaler_artifact)
