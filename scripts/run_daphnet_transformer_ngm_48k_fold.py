#!/usr/bin/env python3
"""Train one compact Transformer-NGM fold on 64-Hz Daphnet processed_NBM.

This worker deliberately reuses the audited role/scaler/training/calibration
pipeline from ``run_daphnet_transformer_nbm300_fold`` while replacing only the
normal-gait model.  The model has one global Z=[B,16] bottleneck and no temporal
skip, cross-attention, raw-input bypass, or teacher forcing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_daphnet_transformer_nbm300_fold as base


CHANNELS = 9
WINDOW_SAMPLES = 128
PATCH_SIZE = 8
TOKEN_COUNT = WINDOW_SAMPLES // PATCH_SIZE
PATCH_DIM = CHANNELS * PATCH_SIZE
MODEL_DIM = 40
HEADS = 4
FFN_DIM = 80
ENCODER_LAYERS = 2
DECODER_LAYERS = 1
BOTTLENECK_DIM = 16
PARAMETER_COUNT = 48_208
ARCHITECTURE_NAME = "tiny_patch_transformer_ngm_global_z16_48k_v1"


class PatchTransformerNGM48K(nn.Module):
    """Skip-free compact patch Transformer for BCT input [B,9,128]."""

    def __init__(self, dropout: float = 0.10) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.patch_projection = nn.Linear(PATCH_DIM, MODEL_DIM)
        self.encoder_position = nn.Parameter(
            torch.zeros(1, TOKEN_COUNT, MODEL_DIM)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=MODEL_DIM,
            nhead=HEADS,
            dim_feedforward=FFN_DIM,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=ENCODER_LAYERS,
            enable_nested_tensor=False,
        )
        self.to_bottleneck = nn.Linear(MODEL_DIM, BOTTLENECK_DIM)
        self.from_bottleneck = nn.Linear(BOTTLENECK_DIM, MODEL_DIM)
        self.decoder_position = nn.Parameter(
            torch.zeros(1, TOKEN_COUNT, MODEL_DIM)
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=MODEL_DIM,
            nhead=HEADS,
            dim_feedforward=FFN_DIM,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        # These decoder blocks are self-attention only.  Encoder tokens are not
        # provided to the decoder; all reconstruction information passes Z16.
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=DECODER_LAYERS,
            enable_nested_tensor=False,
        )
        self.patch_output = nn.Linear(MODEL_DIM, PATCH_DIM)
        nn.init.trunc_normal_(self.encoder_position, std=0.02)
        nn.init.trunc_normal_(self.decoder_position, std=0.02)
        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != PARAMETER_COUNT:
            raise RuntimeError(
                f"Transformer-NGM parameter contract changed: {actual}"
            )

    @staticmethod
    def patchify(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (CHANNELS, WINDOW_SAMPLES):
            raise ValueError(f"expected [B,9,128], got {tuple(x.shape)}")
        patches = x.unfold(dimension=2, size=PATCH_SIZE, step=PATCH_SIZE)
        return patches.permute(0, 2, 1, 3).reshape(
            x.shape[0], TOKEN_COUNT, PATCH_DIM
        )

    @staticmethod
    def fold_patches(patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 3 or tuple(patches.shape[1:]) != (
            TOKEN_COUNT,
            PATCH_DIM,
        ):
            raise ValueError(f"expected [B,16,72], got {tuple(patches.shape)}")
        values = patches.reshape(
            patches.shape[0], TOKEN_COUNT, CHANNELS, PATCH_SIZE
        )
        return values.permute(0, 2, 1, 3).reshape(
            patches.shape[0], CHANNELS, WINDOW_SAMPLES
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_projection(self.patchify(x))
        encoded = self.encoder(tokens + self.encoder_position)
        z = self.to_bottleneck(encoded.mean(dim=1))
        if tuple(z.shape[1:]) != (BOTTLENECK_DIM,):
            raise RuntimeError(f"unexpected bottleneck shape: {tuple(z.shape)}")
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != BOTTLENECK_DIM:
            raise ValueError(f"expected [B,16], got {tuple(z.shape)}")
        global_token = self.from_bottleneck(z).unsqueeze(1)
        tokens = global_token.expand(-1, TOKEN_COUNT, -1)
        decoded = self.decoder(tokens + self.decoder_position)
        return self.fold_patches(self.patch_output(decoded))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction = self.decode(self.encode(x))
        if reconstruction.shape != x.shape:
            raise RuntimeError(
                f"reconstruction {tuple(reconstruction.shape)} != input {tuple(x.shape)}"
            )
        return reconstruction

    def architecture_config(self) -> dict[str, Any]:
        return {
            "name": ARCHITECTURE_NAME,
            "dropout": self.dropout,
            "input_shape": ["B", 9, 128],
            "patchify": {
                "patch_size": PATCH_SIZE,
                "non_overlapping": True,
                "token_shape": ["B", TOKEN_COUNT, PATCH_DIM],
                "patch_order": "time-major; each token flattens 9 axes x 8 samples",
            },
            "patch_projection": f"Linear({PATCH_DIM},{MODEL_DIM})",
            "encoder_position": [1, TOKEN_COUNT, MODEL_DIM],
            "encoder": {
                "layers": ENCODER_LAYERS,
                "d_model": MODEL_DIM,
                "heads": HEADS,
                "ffn": FFN_DIM,
                "activation": "GELU",
                "dropout": self.dropout,
                "normalization": "post-norm",
            },
            "token_pooling": "parameter-free temporal mean",
            "bottleneck_projection": f"Linear({MODEL_DIM},{BOTTLENECK_DIM})",
            "bottleneck_shape": ["B", BOTTLENECK_DIM],
            "decoder_conditioning": (
                "Linear(16,40), broadcast to 16 tokens, add learned position"
            ),
            "decoder_position": [1, TOKEN_COUNT, MODEL_DIM],
            "decoder": {
                "layers": DECODER_LAYERS,
                "type": "self-attention only; no encoder-memory cross-attention",
                "d_model": MODEL_DIM,
                "heads": HEADS,
                "ffn": FFN_DIM,
                "activation": "GELU",
                "dropout": self.dropout,
                "normalization": "post-norm",
            },
            "patch_output": f"Linear({MODEL_DIM},{PATCH_DIM}), exact fold",
            "encoder_decoder_skip_connections": False,
            "cross_attention": False,
            "teacher_forcing": False,
            "raw_input_bypass": False,
            "output_activation": None,
            "output_shape": ["B", 9, 128],
            "parameter_count": PARAMETER_COUNT,
        }


@torch.no_grad()
def reconstruct_transformer_48k(
    model: PatchTransformerNGM48K,
    x: Any,
    device: torch.device,
    batch_size: int = 128,
) -> Any:
    """Deterministically reconstruct a non-empty BCT window collection."""
    model.eval()
    outputs: list[np.ndarray] = []
    for (batch,) in base.make_loader(x, batch_size, False, 0, 0):
        prediction = model(batch.to(device, non_blocking=True))
        outputs.append(prediction.cpu().numpy().astype(np.float32))
    if not outputs:
        raise ValueError("cannot reconstruct an empty window collection")
    return np.concatenate(outputs, axis=0)


def configure_base() -> None:
    """Inject only the compact architecture into the audited NBM pipeline."""
    base.__doc__ = __doc__
    base.PatchTransformerNBM = PatchTransformerNGM48K
    base.reconstruct_transformer = reconstruct_transformer_48k


def main() -> None:
    configure_base()
    base.main()


if __name__ == "__main__":
    main()
