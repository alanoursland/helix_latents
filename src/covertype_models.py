"""Model definitions for Covertype tabular classification experiment."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from covertype_config import CovertypeConfig


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─── Geometric Layers ─────────────────────────────────────────────────────────


class CircleLayer(nn.Module):
    """Learned circular feature layer for tabular data.

    Each unit projects input into a 2D subspace (u, v), computes radius and
    phase, then emits 5 features: sin_theta, cos_theta, r, r*sin, r*cos.
    """

    def __init__(
        self,
        input_dim: int,
        units: int,
        output_dim: int,
        eps: float = 1e-6,
        use_layernorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.eps = eps

        scale = 1.0 / math.sqrt(input_dim)
        self.u = nn.Parameter(torch.empty(input_dim, units).normal_(0, scale))
        self.v = nn.Parameter(torch.empty(input_dim, units).normal_(0, scale))
        self.bias_u = nn.Parameter(torch.zeros(units))
        self.bias_v = nn.Parameter(torch.zeros(units))

        self.out = nn.Linear(units * 5, output_dim)
        self.layernorm = nn.LayerNorm(output_dim) if use_layernorm else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x @ self.u + self.bias_u
        b = x @ self.v + self.bias_v

        r = torch.sqrt(a * a + b * b + self.eps)
        cos_theta = a / r
        sin_theta = b / r

        features = torch.cat([
            sin_theta,
            cos_theta,
            r,
            r * sin_theta,
            r * cos_theta,
        ], dim=-1)

        features = self.drop(features)
        y = self.out(features)
        y = self.layernorm(y)
        return y


class HelixLayer(nn.Module):
    """Learned helical feature layer for tabular data.

    Each unit projects input into a 3D subspace (u, v, w), computes radius,
    phase, and axis position, then emits 8 features per unit.
    """

    def __init__(
        self,
        input_dim: int,
        units: int,
        output_dim: int,
        eps: float = 1e-6,
        use_layernorm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.eps = eps

        scale = 1.0 / math.sqrt(input_dim)
        self.u = nn.Parameter(torch.empty(input_dim, units).normal_(0, scale))
        self.v = nn.Parameter(torch.empty(input_dim, units).normal_(0, scale))
        self.w = nn.Parameter(torch.empty(input_dim, units).normal_(0, scale))
        self.bias_u = nn.Parameter(torch.zeros(units))
        self.bias_v = nn.Parameter(torch.zeros(units))
        self.bias_w = nn.Parameter(torch.zeros(units))

        self.out = nn.Linear(units * 8, output_dim)
        self.layernorm = nn.LayerNorm(output_dim) if use_layernorm else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x @ self.u + self.bias_u
        b = x @ self.v + self.bias_v
        z = x @ self.w + self.bias_w

        r = torch.sqrt(a * a + b * b + self.eps)
        cos_theta = a / r
        sin_theta = b / r

        features = torch.cat([
            sin_theta,
            cos_theta,
            r,
            z,
            r * sin_theta,
            r * cos_theta,
            torch.tanh(z),
            r * torch.tanh(z),
        ], dim=-1)

        features = self.drop(features)
        y = self.out(features)
        y = self.layernorm(y)
        return y


# ─── Full Models ──────────────────────────────────────────────────────────────


class StandardMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 54,
        num_classes: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CircleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 54,
        num_classes: int = 7,
        hidden_dim: int = 128,
        units: int = 64,
        num_layers: int = 2,
        use_layernorm: bool = True,
        dropout: float = 0.05,
    ):
        super().__init__()

        geo_layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            geo_layers.append(CircleLayer(in_dim, units, hidden_dim, use_layernorm=use_layernorm, dropout=dropout))
            in_dim = hidden_dim
        self.geo_layers = nn.ModuleList(geo_layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.geo_layers:
            x = self.dropout(F.gelu(layer(x)))
        return self.classifier(x)


class HelixMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 54,
        num_classes: int = 7,
        hidden_dim: int = 128,
        units: int = 64,
        num_layers: int = 2,
        use_layernorm: bool = True,
        dropout: float = 0.05,
    ):
        super().__init__()

        geo_layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            geo_layers.append(HelixLayer(in_dim, units, hidden_dim, use_layernorm=use_layernorm, dropout=dropout))
            in_dim = hidden_dim
        self.geo_layers = nn.ModuleList(geo_layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.geo_layers:
            x = self.dropout(F.gelu(layer(x)))
        return self.classifier(x)


# ─── Factory ──────────────────────────────────────────────────────────────────


def build_covertype_model(config: CovertypeConfig) -> nn.Module:
    if config.model_type == "standard_mlp":
        return StandardMLP(
            input_dim=config.input_dim,
            num_classes=config.num_classes,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    elif config.model_type == "standard_mlp_matched":
        return StandardMLP(
            input_dim=config.input_dim,
            num_classes=config.num_classes,
            hidden_dim=config.matched_hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    elif config.model_type == "circle_mlp":
        return CircleMLP(
            input_dim=config.input_dim,
            num_classes=config.num_classes,
            hidden_dim=config.hidden_dim,
            units=config.circle_units,
            num_layers=config.num_layers,
            use_layernorm=config.use_layernorm,
            dropout=config.dropout,
        )
    elif config.model_type == "helix_mlp":
        return HelixMLP(
            input_dim=config.input_dim,
            num_classes=config.num_classes,
            hidden_dim=config.hidden_dim,
            units=config.helix_units,
            num_layers=config.num_layers,
            use_layernorm=config.use_layernorm,
            dropout=config.dropout,
        )
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
