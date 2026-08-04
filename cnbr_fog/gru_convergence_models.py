"""GRU mean-forecast models used by the S01 convergence experiments.

The models in this module deliberately separate the common context encoder
from the future decoder.  Every mean decoder has the same target-free
contract::

    delta = decoder(state, last_observation)

where ``state`` is ``[batch, hidden]``, ``last_observation`` is
``[batch, channel]``, and ``delta`` is ``[batch, channel, horizon]``.  The
forecaster applies the persistence anchor exactly once:

    mean = last_observation[..., None] + delta

Mean-only models return a unit standard deviation solely to preserve the
repository-wide ``mean, sigma = model(context)`` interface.  That unit scale
must not be used to construct diagnostic residuals; a scale model must be
fitted first (for example :class:`JointDirectGRUForecaster`).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable

import torch
from torch import nn

from .models import CausalResidualBlock
from .nbm import NormalBehaviourModel


DECODER_NAMES = ("direct", "shared_horizon", "tcn", "gru")


def _positive_int(value: int, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _nearest_width(
    target_parameters: int,
    count_for_width: Callable[[int], int],
    *,
    maximum_width: int = 4096,
) -> int:
    """Return the positive integer width closest to a parameter budget."""

    target = _positive_int(target_parameters, "target_parameters")
    maximum = _positive_int(maximum_width, "maximum_width")
    best_width = 1
    best_error = abs(int(count_for_width(1)) - target)
    for width in range(2, maximum + 1):
        count = int(count_for_width(width))
        error = abs(count - target)
        if error < best_error:
            best_width = width
            best_error = error
        # All count functions used here are strictly increasing.  Once the
        # budget is crossed and the error has started increasing, no larger
        # width can be better.
        if count >= target and error > best_error:
            break
    return best_width


def _initialise_small_delta_head(layer: nn.Module) -> None:
    """Start all decoder families close to the same persistence forecast."""

    weight = getattr(layer, "weight", None)
    bias = getattr(layer, "bias", None)
    if weight is None:
        raise TypeError("delta output layer must expose a weight parameter")
    nn.init.normal_(weight, mean=0.0, std=1e-3)
    if bias is not None:
        nn.init.zeros_(bias)


class GRUContextEncoder(nn.Module):
    """The encoder shared without architectural changes by every forecaster."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 48,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.hidden_channels = _positive_int(hidden_channels, "hidden_channels")
        self.num_layers = _positive_int(num_layers, "num_layers")
        self.dropout = float(dropout)
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.gru = nn.GRU(
            input_size=self.in_channels,
            hidden_size=self.hidden_channels,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.summary = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 3:
            raise ValueError("context must have shape [batch, channel, time]")
        if int(context.shape[1]) != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels, got {context.shape[1]}"
            )
        if int(context.shape[2]) < 1:
            raise ValueError("context must contain at least one time sample")
        _, hidden = self.gru(context.transpose(1, 2))
        return self.summary(hidden[-1])

    def model_config(self) -> dict[str, Any]:
        return {
            "name": "gru_context_encoder",
            "in_channels": self.in_channels,
            "hidden_channels": self.hidden_channels,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }


class MeanDecoder(nn.Module):
    """Base class for target-free, persistence-residual mean decoders."""

    decoder_name = "base"
    uses_teacher_forcing = False

    def __init__(self, state_size: int, in_channels: int, horizon: int) -> None:
        super().__init__()
        self.state_size = _positive_int(state_size, "state_size")
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.horizon = _positive_int(horizon, "horizon")

    def _check_inputs(self, state: torch.Tensor, last: torch.Tensor) -> None:
        if state.ndim != 2 or int(state.shape[1]) != self.state_size:
            raise ValueError(
                f"state must have shape [batch,{self.state_size}], got "
                f"{tuple(state.shape)}"
            )
        if last.ndim != 2 or int(last.shape[1]) != self.in_channels:
            raise ValueError(
                f"last must have shape [batch,{self.in_channels}], got "
                f"{tuple(last.shape)}"
            )
        if int(state.shape[0]) != int(last.shape[0]):
            raise ValueError("state and last batch sizes differ")

    def model_config(self) -> dict[str, Any]:
        return {
            "name": self.decoder_name,
            "state_size": self.state_size,
            "in_channels": self.in_channels,
            "horizon": self.horizon,
            "uses_teacher_forcing": self.uses_teacher_forcing,
            "parameter_count": _parameter_count(self),
        }


class DirectMeanDecoder(MeanDecoder):
    """One independent affine state map for every channel and lead."""

    decoder_name = "direct"

    def __init__(self, state_size: int, in_channels: int, horizon: int) -> None:
        super().__init__(state_size, in_channels, horizon)
        self.output_head = nn.Linear(
            self.state_size, self.in_channels * self.horizon
        )
        _initialise_small_delta_head(self.output_head)

    def forward(self, state: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        self._check_inputs(state, last)
        return self.output_head(state).reshape(
            state.shape[0], self.in_channels, self.horizon
        )


class SharedHorizonMeanDecoder(MeanDecoder):
    """Factorised decoder with shared state/output maps across forecast leads."""

    decoder_name = "shared_horizon"

    @staticmethod
    def parameter_count_for_width(
        width: int, state_size: int, in_channels: int, horizon: int
    ) -> int:
        width = int(width)
        return width * (state_size + horizon + in_channels + 1) + in_channels

    @classmethod
    def matched_width(cls, state_size: int, in_channels: int, horizon: int) -> int:
        target = (state_size + 1) * in_channels * horizon
        return _nearest_width(
            target,
            lambda width: cls.parameter_count_for_width(
                width, state_size, in_channels, horizon
            ),
        )

    def __init__(
        self,
        state_size: int,
        in_channels: int,
        horizon: int,
        width: int | None = None,
    ) -> None:
        super().__init__(state_size, in_channels, horizon)
        self.width = (
            self.matched_width(state_size, in_channels, horizon)
            if width is None
            else _positive_int(width, "width")
        )
        self.state_projection = nn.Linear(self.state_size, self.width)
        self.lead_embedding = nn.Embedding(self.horizon, self.width)
        self.activation = nn.GELU()
        self.output_head = nn.Linear(self.width, self.in_channels)
        nn.init.normal_(self.lead_embedding.weight, mean=0.0, std=0.02)
        _initialise_small_delta_head(self.output_head)

    def forward(self, state: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        self._check_inputs(state, last)
        latent = self.state_projection(state).unsqueeze(1)
        latent = latent + self.lead_embedding.weight.unsqueeze(0)
        delta = self.output_head(self.activation(latent))
        return delta.transpose(1, 2).contiguous()

    def model_config(self) -> dict[str, Any]:
        return {**super().model_config(), "width": self.width}


class TCNMeanDecoder(MeanDecoder):
    """Parallel causal TCN decoder seeded only by state and lead embeddings."""

    decoder_name = "tcn"

    @staticmethod
    def parameter_count_for_width(
        width: int,
        state_size: int,
        in_channels: int,
        horizon: int,
        kernel_size: int,
        block_count: int,
    ) -> int:
        width = int(width)
        return (
            2 * kernel_size * block_count * width * width
            + width * (state_size + horizon + in_channels + 1 + 4 * block_count)
            + in_channels
        )

    @classmethod
    def matched_width(
        cls,
        state_size: int,
        in_channels: int,
        horizon: int,
        kernel_size: int,
        block_count: int,
    ) -> int:
        target = (state_size + 1) * in_channels * horizon
        return _nearest_width(
            target,
            lambda width: cls.parameter_count_for_width(
                width,
                state_size,
                in_channels,
                horizon,
                kernel_size,
                block_count,
            ),
        )

    def __init__(
        self,
        state_size: int,
        in_channels: int,
        horizon: int,
        width: int | None = None,
        dilations: Iterable[int] = (1, 2, 4, 8, 16),
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(state_size, in_channels, horizon)
        self.dilations = tuple(_positive_int(value, "dilation") for value in dilations)
        if not self.dilations:
            raise ValueError("dilations must not be empty")
        self.kernel_size = _positive_int(kernel_size, "kernel_size")
        self.dropout = float(dropout)
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.width = (
            self.matched_width(
                state_size,
                in_channels,
                horizon,
                self.kernel_size,
                len(self.dilations),
            )
            if width is None
            else _positive_int(width, "width")
        )
        self.state_projection = nn.Linear(self.state_size, self.width)
        self.lead_embedding = nn.Embedding(self.horizon, self.width)
        self.activation = nn.GELU()
        self.blocks = nn.Sequential(
            *[
                CausalResidualBlock(
                    self.width, self.kernel_size, dilation, self.dropout
                )
                for dilation in self.dilations
            ]
        )
        self.output_head = nn.Conv1d(self.width, self.in_channels, kernel_size=1)
        nn.init.normal_(self.lead_embedding.weight, mean=0.0, std=0.02)
        _initialise_small_delta_head(self.output_head)

    def forward(self, state: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        self._check_inputs(state, last)
        latent = self.state_projection(state).unsqueeze(-1)
        latent = latent + self.lead_embedding.weight.transpose(0, 1).unsqueeze(0)
        return self.output_head(self.blocks(self.activation(latent)))

    def model_config(self) -> dict[str, Any]:
        receptive_field = 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)
        return {
            **super().model_config(),
            "width": self.width,
            "dilations": list(self.dilations),
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "receptive_field_samples": receptive_field,
        }


class AutoregressiveGRUMeanDecoder(MeanDecoder):
    """Free-running GRU decoder; ground-truth future values are never inputs."""

    decoder_name = "gru"
    uses_teacher_forcing = False

    @staticmethod
    def parameter_count_for_width(
        width: int, state_size: int, in_channels: int
    ) -> int:
        width = int(width)
        return (
            3 * width * width
            + width * (state_size + 4 * in_channels + 7)
            + in_channels
        )

    @classmethod
    def matched_width(cls, state_size: int, in_channels: int, horizon: int) -> int:
        target = (state_size + 1) * in_channels * horizon
        return _nearest_width(
            target,
            lambda width: cls.parameter_count_for_width(
                width, state_size, in_channels
            ),
        )

    def __init__(
        self,
        state_size: int,
        in_channels: int,
        horizon: int,
        width: int | None = None,
    ) -> None:
        super().__init__(state_size, in_channels, horizon)
        self.width = (
            self.matched_width(state_size, in_channels, horizon)
            if width is None
            else _positive_int(width, "width")
        )
        self.initial_state = nn.Linear(self.state_size, self.width)
        self.cell = nn.GRUCell(self.in_channels, self.width)
        self.output_head = nn.Linear(self.width, self.in_channels)
        _initialise_small_delta_head(self.output_head)

    def forward(self, state: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        self._check_inputs(state, last)
        hidden = torch.tanh(self.initial_state(state))
        previous = last
        deltas: list[torch.Tensor] = []
        for _ in range(self.horizon):
            hidden = self.cell(previous, hidden)
            delta = self.output_head(hidden)
            deltas.append(delta)
            # Free running: the next input is this model's prediction, never a
            # ground-truth target.  Each output remains anchored to ``last``.
            previous = last + delta
        return torch.stack(deltas, dim=-1)

    def model_config(self) -> dict[str, Any]:
        return {**super().model_config(), "width": self.width}


# Short aliases make experiment tables and imports less verbose.
DirectDecoder = DirectMeanDecoder
SharedHorizonDecoder = SharedHorizonMeanDecoder
TCNDecoder = TCNMeanDecoder
GRUDecoder = AutoregressiveGRUMeanDecoder


def build_mean_decoder(
    name: str,
    state_size: int,
    in_channels: int,
    horizon: int,
    *,
    width: int | None = None,
    tcn_dilations: Iterable[int] = (1, 2, 4, 8, 16),
    tcn_kernel_size: int = 3,
    decoder_dropout: float = 0.0,
) -> MeanDecoder:
    canonical = str(name).strip().lower().replace("-", "_")
    aliases = {
        "direct": "direct",
        "shared": "shared_horizon",
        "shared_horizon": "shared_horizon",
        "tcn": "tcn",
        "gru": "gru",
        "autoregressive_gru": "gru",
    }
    if canonical not in aliases:
        raise ValueError(f"unknown decoder {name!r}; choose from {DECODER_NAMES}")
    canonical = aliases[canonical]
    if canonical == "direct":
        if width is not None:
            raise ValueError("the direct decoder has no width hyperparameter")
        return DirectMeanDecoder(state_size, in_channels, horizon)
    if canonical == "shared_horizon":
        return SharedHorizonMeanDecoder(state_size, in_channels, horizon, width)
    if canonical == "tcn":
        return TCNMeanDecoder(
            state_size,
            in_channels,
            horizon,
            width,
            tcn_dilations,
            tcn_kernel_size,
            decoder_dropout,
        )
    if canonical == "gru":
        return AutoregressiveGRUMeanDecoder(
            state_size, in_channels, horizon, width
        )
    raise AssertionError(canonical)


class GRUMeanModelBase(NormalBehaviourModel):
    """Common encoder, copy protocol, and mean-only Gaussian compatibility."""

    model_name = "gru_mean"

    def __init__(
        self,
        in_channels: int,
        horizon: int,
        hidden_channels: int = 48,
        num_layers: int = 1,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(in_channels, horizon, **kwargs)
        self.encoder = GRUContextEncoder(
            self.in_channels, hidden_channels, num_layers, dropout
        )
        self.register_buffer(
            "_unit_log_sigma",
            torch.zeros(1, self.in_channels, self.horizon),
            persistent=False,
        )

    @property
    def hidden_channels(self) -> int:
        return self.encoder.hidden_channels

    def copy_encoder_from(self, other: "GRUMeanModelBase") -> "GRUMeanModelBase":
        """Copy a compatible encoder exactly, leaving both decoders untouched."""

        if not isinstance(other, GRUMeanModelBase):
            raise TypeError("other must be a GRUMeanModelBase")
        if self.encoder.model_config() != other.encoder.model_config():
            raise ValueError("encoder configurations differ")
        self.encoder.load_state_dict(other.encoder.state_dict())
        return self

    def forward_mean(self, context: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.forward_mean(context)
        return self._distribution(mean, self._unit_log_sigma)

    def model_config(self) -> dict[str, Any]:
        return {
            **super().model_config(),
            "encoder": self.encoder.model_config(),
            "persistence_anchor": "context_last_sample",
            "mean_only_sigma": "fixed_unit_adapter_not_for_residuals",
        }


class GRUMeanForecaster(GRUMeanModelBase):
    """Common S01 GRU encoder paired with one interchangeable mean decoder."""

    def __init__(
        self,
        in_channels: int,
        horizon: int,
        hidden_channels: int = 48,
        num_layers: int = 1,
        dropout: float = 0.1,
        decoder: str = "direct",
        decoder_width: int | None = None,
        tcn_dilations: Iterable[int] = (1, 2, 4, 8, 16),
        tcn_kernel_size: int = 3,
        decoder_dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        # Constructing the encoder before the decoder guarantees that resetting
        # torch's RNG to the same seed gives every variant identical encoder
        # initialisation.
        super().__init__(
            in_channels,
            horizon,
            hidden_channels,
            num_layers,
            dropout,
            **kwargs,
        )
        self.mean_decoder = build_mean_decoder(
            decoder,
            self.hidden_channels,
            self.in_channels,
            self.horizon,
            width=decoder_width,
            tcn_dilations=tcn_dilations,
            tcn_kernel_size=tcn_kernel_size,
            decoder_dropout=decoder_dropout,
        )

    def forward_mean(self, context: torch.Tensor) -> torch.Tensor:
        self._check_context(context)
        state = self.encoder(context)
        last = context[:, :, -1]
        delta = self.mean_decoder(state, last)
        return last.unsqueeze(-1) + delta

    def model_config(self) -> dict[str, Any]:
        direct_budget = (self.hidden_channels + 1) * self.in_channels * self.horizon
        decoder_parameters = _parameter_count(self.mean_decoder)
        return {
            **super().model_config(),
            "decoder": self.mean_decoder.model_config(),
            "direct_decoder_parameter_budget": direct_budget,
            "decoder_to_direct_parameter_ratio": decoder_parameters / direct_budget,
        }


class DirectGRUMeanForecaster(GRUMeanForecaster):
    def __init__(self, in_channels: int, horizon: int, **kwargs: Any) -> None:
        super().__init__(in_channels, horizon, decoder="direct", **kwargs)


class SharedHorizonGRUMeanForecaster(GRUMeanForecaster):
    def __init__(self, in_channels: int, horizon: int, **kwargs: Any) -> None:
        super().__init__(in_channels, horizon, decoder="shared_horizon", **kwargs)


class TCNGRUMeanForecaster(GRUMeanForecaster):
    def __init__(self, in_channels: int, horizon: int, **kwargs: Any) -> None:
        super().__init__(in_channels, horizon, decoder="tcn", **kwargs)


class AutoregressiveGRUMeanForecaster(GRUMeanForecaster):
    def __init__(self, in_channels: int, horizon: int, **kwargs: Any) -> None:
        super().__init__(in_channels, horizon, decoder="gru", **kwargs)


class JointDirectGRUForecaster(GRUMeanModelBase):
    """Direct Gaussian model with optional sigma-to-encoder gradient blocking."""

    model_name = "gru_joint_direct"

    def __init__(
        self,
        in_channels: int,
        horizon: int,
        hidden_channels: int = 48,
        num_layers: int = 1,
        dropout: float = 0.1,
        sigma_state_detach: bool = False,
        initial_log_sigma: float = -0.75,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            in_channels,
            horizon,
            hidden_channels,
            num_layers,
            dropout,
            **kwargs,
        )
        self.sigma_state_detach = bool(sigma_state_detach)
        self.initial_log_sigma = float(initial_log_sigma)
        self.mean_decoder = DirectMeanDecoder(
            self.hidden_channels, self.in_channels, self.horizon
        )
        self.log_sigma_head = nn.Linear(
            self.hidden_channels, self.in_channels * self.horizon
        )
        nn.init.normal_(self.log_sigma_head.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.log_sigma_head.bias, self.initial_log_sigma)

    def _mean_from_state(
        self, context: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        last = context[:, :, -1]
        return last.unsqueeze(-1) + self.mean_decoder(state, last)

    def forward_mean(self, context: torch.Tensor) -> torch.Tensor:
        self._check_context(context)
        state = self.encoder(context)
        return self._mean_from_state(context, state)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_context(context)
        state = self.encoder(context)
        mean = self._mean_from_state(context, state)
        sigma_state = state.detach() if self.sigma_state_detach else state
        log_sigma = self.log_sigma_head(sigma_state).reshape(
            context.shape[0], self.in_channels, self.horizon
        )
        return self._distribution(mean, log_sigma)

    def model_config(self) -> dict[str, Any]:
        return {
            **super().model_config(),
            "mean_decoder": self.mean_decoder.model_config(),
            "sigma_head": "direct_state_to_channel_horizon_log_sigma",
            "sigma_state_detach": self.sigma_state_detach,
            "initial_log_sigma": self.initial_log_sigma,
            "mean_only_sigma": None,
        }


class ClusterConditionedGRUMeanForecaster(GRUMeanModelBase):
    """Soft latent-cluster conditioning followed by one shared mean decoder."""

    model_name = "gru_cluster_conditioned_mean"

    def __init__(
        self,
        in_channels: int,
        horizon: int,
        hidden_channels: int = 48,
        num_layers: int = 1,
        dropout: float = 0.1,
        n_clusters: int = 4,
        cluster_embedding_dim: int = 8,
        routing_temperature: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            in_channels,
            horizon,
            hidden_channels,
            num_layers,
            dropout,
            **kwargs,
        )
        self.n_clusters = _positive_int(n_clusters, "n_clusters")
        self.cluster_embedding_dim = _positive_int(
            cluster_embedding_dim, "cluster_embedding_dim"
        )
        self.routing_temperature = float(routing_temperature)
        if self.routing_temperature <= 0:
            raise ValueError("routing_temperature must be positive")
        self.cluster_gate = nn.Linear(self.hidden_channels, self.n_clusters)
        self.cluster_embeddings = nn.Parameter(
            torch.empty(self.n_clusters, self.cluster_embedding_dim)
        )
        nn.init.normal_(self.cluster_embeddings, mean=0.0, std=0.02)
        self.mean_decoder = DirectMeanDecoder(
            self.hidden_channels + self.cluster_embedding_dim,
            self.in_channels,
            self.horizon,
        )

    def _probabilities_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return torch.softmax(
            self.cluster_gate(state) / self.routing_temperature, dim=-1
        )

    def cluster_probabilities(self, context: torch.Tensor) -> torch.Tensor:
        self._check_context(context)
        return self._probabilities_from_state(self.encoder(context))

    def forward_mean(self, context: torch.Tensor) -> torch.Tensor:
        self._check_context(context)
        state = self.encoder(context)
        probabilities = self._probabilities_from_state(state)
        condition = probabilities @ self.cluster_embeddings
        conditioned_state = torch.cat([state, condition], dim=-1)
        last = context[:, :, -1]
        return last.unsqueeze(-1) + self.mean_decoder(conditioned_state, last)

    def model_config(self) -> dict[str, Any]:
        return {
            **super().model_config(),
            "n_clusters": self.n_clusters,
            "cluster_embedding_dim": self.cluster_embedding_dim,
            "routing_temperature": self.routing_temperature,
            "routing": "context_only_softmax",
            "mean_decoder": self.mean_decoder.model_config(),
        }


class MoEGRUMeanForecaster(GRUMeanModelBase):
    """Context-gated mixture of direct future-mean experts."""

    model_name = "gru_moe_mean"

    def __init__(
        self,
        in_channels: int,
        horizon: int,
        hidden_channels: int = 48,
        num_layers: int = 1,
        dropout: float = 0.1,
        n_experts: int = 4,
        routing_temperature: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            in_channels,
            horizon,
            hidden_channels,
            num_layers,
            dropout,
            **kwargs,
        )
        self.n_experts = _positive_int(n_experts, "n_experts")
        self.routing_temperature = float(routing_temperature)
        if self.routing_temperature <= 0:
            raise ValueError("routing_temperature must be positive")
        self.gate = nn.Linear(self.hidden_channels, self.n_experts)
        self.experts = nn.ModuleList(
            [
                DirectMeanDecoder(
                    self.hidden_channels, self.in_channels, self.horizon
                )
                for _ in range(self.n_experts)
            ]
        )

    def _probabilities_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.gate(state) / self.routing_temperature, dim=-1)

    def routing_probabilities(self, context: torch.Tensor) -> torch.Tensor:
        self._check_context(context)
        return self._probabilities_from_state(self.encoder(context))

    def forward_mean(self, context: torch.Tensor) -> torch.Tensor:
        self._check_context(context)
        state = self.encoder(context)
        probabilities = self._probabilities_from_state(state)
        last = context[:, :, -1]
        expert_delta = torch.stack(
            [expert(state, last) for expert in self.experts], dim=1
        )
        delta = torch.sum(
            probabilities[:, :, None, None] * expert_delta, dim=1
        )
        return last.unsqueeze(-1) + delta

    def model_config(self) -> dict[str, Any]:
        return {
            **super().model_config(),
            "n_experts": self.n_experts,
            "routing_temperature": self.routing_temperature,
            "routing": "context_only_softmax",
            "expert": self.experts[0].model_config(),
        }


# Descriptive aliases for callers that prefer the full terminology.
ClusterConditionedMeanForecaster = ClusterConditionedGRUMeanForecaster
MixtureOfExpertsGRUMeanForecaster = MoEGRUMeanForecaster


__all__ = [
    "DECODER_NAMES",
    "AutoregressiveGRUMeanDecoder",
    "AutoregressiveGRUMeanForecaster",
    "ClusterConditionedGRUMeanForecaster",
    "ClusterConditionedMeanForecaster",
    "DirectDecoder",
    "DirectGRUMeanForecaster",
    "DirectMeanDecoder",
    "GRUContextEncoder",
    "GRUDecoder",
    "GRUMeanForecaster",
    "GRUMeanModelBase",
    "JointDirectGRUForecaster",
    "MeanDecoder",
    "MixtureOfExpertsGRUMeanForecaster",
    "MoEGRUMeanForecaster",
    "SharedHorizonDecoder",
    "SharedHorizonGRUMeanForecaster",
    "SharedHorizonMeanDecoder",
    "TCNDecoder",
    "TCNGRUMeanForecaster",
    "TCNMeanDecoder",
    "build_mean_decoder",
]
