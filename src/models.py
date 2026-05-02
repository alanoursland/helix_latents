"""Model definitions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from config import ExperimentConfig
from geometry import circle_encode, helix_encode


@dataclass
class ModelOutput:
    logits: torch.Tensor
    latents: dict[str, torch.Tensor]


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_hidden_layers: int,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.GELU())
    if dropout > 0:
        layers.append(nn.Dropout(dropout))

    for _ in range(num_hidden_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class BaselineMLP(nn.Module):
    def __init__(self, modulus: int, embedding_dim: int = 32, hidden_dim: int = 128,
                 num_hidden_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.modulus = modulus
        self.embed_a = nn.Embedding(modulus, embedding_dim)
        self.embed_b = nn.Embedding(modulus, embedding_dim)
        self.mlp = build_mlp(2 * embedding_dim, hidden_dim, modulus, num_hidden_layers, dropout)

    def forward(self, a: torch.Tensor, b: torch.Tensor,
                latent_override: dict[str, torch.Tensor] | None = None) -> ModelOutput:
        a_embed = self.embed_a(a)
        b_embed = self.embed_b(b)
        if latent_override:
            if "a" in latent_override:
                a_embed = latent_override["a"]
            if "b" in latent_override:
                b_embed = latent_override["b"]
        x = torch.cat([a_embed, b_embed], dim=-1)
        logits = self.mlp(x)
        return ModelOutput(logits=logits, latents={"a": a_embed, "b": b_embed})


class CircleBottleneckMLP(nn.Module):
    def __init__(self, modulus: int, hidden_dim: int = 128,
                 num_hidden_layers: int = 2, dropout: float = 0.0, **kwargs):
        super().__init__()
        self.modulus = modulus
        self.mlp = build_mlp(4, hidden_dim, modulus, num_hidden_layers, dropout)

    def forward(self, a: torch.Tensor, b: torch.Tensor,
                latent_override: dict[str, torch.Tensor] | None = None) -> ModelOutput:
        a_circle = circle_encode(a, self.modulus)
        b_circle = circle_encode(b, self.modulus)
        if latent_override:
            if "a" in latent_override:
                a_circle = latent_override["a"]
            if "b" in latent_override:
                b_circle = latent_override["b"]
        x = torch.cat([a_circle, b_circle], dim=-1)
        logits = self.mlp(x)
        return ModelOutput(logits=logits, latents={"a": a_circle, "b": b_circle})


class HelixBottleneckMLP(nn.Module):
    def __init__(self, modulus: int, hidden_dim: int = 128,
                 num_hidden_layers: int = 2, dropout: float = 0.0,
                 alpha: float | None = None, **kwargs):
        super().__init__()
        self.modulus = modulus
        self.alpha = alpha if alpha is not None else 1.0 / modulus
        self.mlp = build_mlp(6, hidden_dim, modulus, num_hidden_layers, dropout)

    def forward(self, a: torch.Tensor, b: torch.Tensor,
                latent_override: dict[str, torch.Tensor] | None = None) -> ModelOutput:
        a_helix = helix_encode(a, self.modulus, self.alpha)
        b_helix = helix_encode(b, self.modulus, self.alpha)
        if latent_override:
            if "a" in latent_override:
                a_helix = latent_override["a"]
            if "b" in latent_override:
                b_helix = latent_override["b"]
        x = torch.cat([a_helix, b_helix], dim=-1)
        logits = self.mlp(x)
        return ModelOutput(logits=logits, latents={"a": a_helix, "b": b_helix})


def build_model(config: ExperimentConfig) -> nn.Module:
    if config.model_type == "baseline_mlp":
        return BaselineMLP(config.modulus, config.embedding_dim, config.hidden_dim,
                           config.num_hidden_layers, config.dropout)
    elif config.model_type == "circle_bottleneck_mlp":
        return CircleBottleneckMLP(config.modulus, config.hidden_dim,
                                   config.num_hidden_layers, config.dropout)
    elif config.model_type == "helix_bottleneck_mlp":
        return HelixBottleneckMLP(config.modulus, config.hidden_dim,
                                  config.num_hidden_layers, config.dropout, config.helix_alpha)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
