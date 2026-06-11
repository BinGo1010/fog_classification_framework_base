from utils.config import load_config
from data_provider.build import build_dataloaders
from models.build import build_model


def test_build():
    cfg = load_config("configs/default.yaml")
    loaders = build_dataloaders(cfg)
    model = build_model(cfg)
    batch = next(iter(loaders["train"]))
    out = model(batch["x"])
    assert out.shape[0] == batch["x"].shape[0]
    assert out.shape[1] == cfg["model"]["num_classes"]
