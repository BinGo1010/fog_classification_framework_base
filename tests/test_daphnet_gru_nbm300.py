import torch

from scripts.run_daphnet_gru_nbm300_fold import architecture
from scripts.run_daphnet_s01_nonfog_gru_reconstruction_tcnm import GRUReconstructionNBM


def test_retained_gru_nbm_architecture_and_shapes() -> None:
    model = GRUReconstructionNBM(channels=9, hidden=64, bottleneck=16)
    x = torch.zeros(3, 128, 9)
    with torch.no_grad():
        prediction = model(x)
    assert prediction.shape == x.shape
    config = architecture(64, 16)
    assert config["latent_shape"] == ["B", 16]
    assert config["skip_connections"] is False
    assert config["input_shape"] == ["B", 128, 9]
    assert config["output_shape"] == ["B", 128, 9]
