"""
Rotated MNIST HelixConv Models
===============================

StandardCNN, CircleConv2d, HelixConv2d, CircleCNN, HelixCNN, and
model factory for the rotated MNIST experiment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rot_mnist_config import RotMNISTConfig


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def quadrature_target(W_u: torch.Tensor) -> torch.Tensor:
    """The W_v that pairs with W_u as a 90° quadrature filter.

    torch.rot90 with k=-1 (clockwise) matches the convention in
    rot_mnist_analyze_filters: the phi* sweep correlates W_u against
    rotate(W_v, phi), so W_v = rot90(W_u, k=-1) yields phi* = 90.
    """
    return torch.rot90(W_u, k=-1, dims=(-2, -1))


# ---------------------------------------------------------------------------
# Standard CNN
# ---------------------------------------------------------------------------

class StandardCNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        hidden_channels: int = 32,
        kernel_size: int = 5,
        num_conv_blocks: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = kernel_size // 2

        layers: list[nn.Module] = []
        in_ch = input_channels
        for _ in range(num_conv_blocks):
            layers.append(nn.Conv2d(in_ch, hidden_channels, kernel_size, padding=padding))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            layers.append(nn.MaxPool2d(2))
            in_ch = hidden_channels

        self.features = nn.Sequential(*layers)

        # After num_conv_blocks MaxPool2d(2) on 28x28: spatial = 28 / 2^blocks
        spatial = 28 // (2 ** num_conv_blocks)
        self.classifier = nn.Linear(hidden_channels * spatial * spatial, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# CircleConv2d
# ---------------------------------------------------------------------------

class CircleConv2d(nn.Module):
    """Circle convolutional layer: 2 independent filter banks, 5 features per unit."""

    def __init__(
        self,
        in_channels: int,
        units: int,
        out_channels: int,
        kernel_size: int = 5,
        padding: int = 2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.units = units

        self.conv_u = nn.Conv2d(in_channels, units, kernel_size, padding=padding)
        self.conv_v = nn.Conv2d(in_channels, units, kernel_size, padding=padding)

        # 5 features per unit: sin_t, cos_t, r, r*sin_t, r*cos_t
        self.project = nn.Conv2d(units * 5, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.conv_u(x)
        b = self.conv_v(x)

        r = torch.sqrt(a * a + b * b + self.eps)
        sin_t = b / r
        cos_t = a / r

        feats = torch.cat([
            sin_t,
            cos_t,
            r,
            r * sin_t,
            r * cos_t,
        ], dim=1)

        return self.project(feats)

    def forward_with_intermediates(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a = self.conv_u(x)
        b = self.conv_v(x)

        r = torch.sqrt(a * a + b * b + self.eps)
        sin_t = b / r
        cos_t = a / r

        feats = torch.cat([
            sin_t,
            cos_t,
            r,
            r * sin_t,
            r * cos_t,
        ], dim=1)

        out = self.project(feats)
        return out, {"a": a, "b": b, "r": r}


# ---------------------------------------------------------------------------
# HelixConv2d
# ---------------------------------------------------------------------------

class HelixConv2d(nn.Module):
    """Helix convolutional layer: 3 independent filter banks, 8 features per unit."""

    def __init__(
        self,
        in_channels: int,
        units: int,
        out_channels: int,
        kernel_size: int = 5,
        padding: int = 2,
        eps: float = 1e-6,
        quadrature_init: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.units = units

        self.conv_u = nn.Conv2d(in_channels, units, kernel_size, padding=padding)
        self.conv_v = nn.Conv2d(in_channels, units, kernel_size, padding=padding)
        self.conv_w = nn.Conv2d(in_channels, units, kernel_size, padding=padding)

        if quadrature_init:
            with torch.no_grad():
                self.conv_v.weight.copy_(quadrature_target(self.conv_u.weight))
                self.conv_v.bias.copy_(self.conv_u.bias)

        # 8 features per unit: sin_t, cos_t, r, z, r*sin_t, r*cos_t, tanh(z), r*tanh(z)
        self.project = nn.Conv2d(units * 8, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.conv_u(x)
        b = self.conv_v(x)
        z = self.conv_w(x)

        r = torch.sqrt(a * a + b * b + self.eps)
        sin_t = b / r
        cos_t = a / r

        feats = torch.cat([
            sin_t,
            cos_t,
            r,
            z,
            r * sin_t,
            r * cos_t,
            torch.tanh(z),
            r * torch.tanh(z),
        ], dim=1)

        return self.project(feats)

    def forward_with_intermediates(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a = self.conv_u(x)
        b = self.conv_v(x)
        z = self.conv_w(x)

        r = torch.sqrt(a * a + b * b + self.eps)
        sin_t = b / r
        cos_t = a / r

        feats = torch.cat([
            sin_t,
            cos_t,
            r,
            z,
            r * sin_t,
            r * cos_t,
            torch.tanh(z),
            r * torch.tanh(z),
        ], dim=1)

        out = self.project(feats)
        return out, {"a": a, "b": b, "z": z, "r": r}

    def quadrature_penalty(self) -> torch.Tensor:
        """Normalized squared distance between W_v and rotate(W_u, 90°).

        ~1.0 for independent random filters, 0.0 for exact quadrature pairs.
        """
        target = quadrature_target(self.conv_u.weight)
        diff = self.conv_v.weight - target
        return (diff * diff).mean() / ((target * target).mean() + self.eps)

    @torch.no_grad()
    def quadrature_alignment(self) -> float:
        """Mean per-unit Pearson correlation between W_v and rotate(W_u, 90°).

        Directly comparable to corr* from the filter analysis: 1.0 means
        perfect quadrature pairing, ~0 means independent filters.
        """
        target = quadrature_target(self.conv_u.weight)
        wv = self.conv_v.weight
        corrs = []
        for u in range(wv.shape[0]):
            a = wv[u].flatten() - wv[u].mean()
            b = target[u].flatten() - target[u].mean()
            denom = torch.sqrt((a * a).sum() * (b * b).sum()) + 1e-12
            corrs.append(((a * b).sum() / denom).item())
        return float(sum(corrs) / len(corrs))


# ---------------------------------------------------------------------------
# CircleCNN
# ---------------------------------------------------------------------------

class CircleCNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        circle_units: int = 16,
        hidden_channels: int = 32,
        kernel_size: int = 5,
        num_conv_blocks: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.num_conv_blocks = num_conv_blocks

        self.conv_blocks = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_ch = input_channels
        for _ in range(num_conv_blocks):
            self.conv_blocks.append(
                CircleConv2d(in_ch, circle_units, hidden_channels,
                             kernel_size=kernel_size, padding=padding)
            )
            act_layers: list[nn.Module] = [nn.GELU()]
            if dropout > 0:
                act_layers.append(nn.Dropout2d(dropout))
            self.activations.append(nn.Sequential(*act_layers))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = hidden_channels

        spatial = 28 // (2 ** num_conv_blocks)
        self.classifier = nn.Linear(hidden_channels * spatial * spatial, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, act, pool in zip(self.conv_blocks, self.activations, self.pools):
            x = pool(act(conv(x)))
        x = x.flatten(1)
        return self.classifier(x)

    def forward_with_intermediates_at_layer(
        self, x: torch.Tensor, layer_idx: int = 0
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run forward, returning intermediates from the specified CircleConv layer."""
        intermediates = {}
        for i, (conv, act, pool) in enumerate(
            zip(self.conv_blocks, self.activations, self.pools)
        ):
            if i == layer_idx:
                out, inter = conv.forward_with_intermediates(x)
                intermediates = inter
                x = pool(act(out))
            else:
                x = pool(act(conv(x)))
        x = x.flatten(1)
        logits = self.classifier(x)
        return logits, intermediates


# ---------------------------------------------------------------------------
# HelixCNN
# ---------------------------------------------------------------------------

class HelixCNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        helix_units: int = 16,
        hidden_channels: int = 32,
        kernel_size: int = 5,
        num_conv_blocks: int = 2,
        dropout: float = 0.0,
        quadrature_init: bool = False,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.num_conv_blocks = num_conv_blocks

        self.conv_blocks = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_ch = input_channels
        for _ in range(num_conv_blocks):
            self.conv_blocks.append(
                HelixConv2d(in_ch, helix_units, hidden_channels,
                            kernel_size=kernel_size, padding=padding,
                            quadrature_init=quadrature_init)
            )
            act_layers: list[nn.Module] = [nn.GELU()]
            if dropout > 0:
                act_layers.append(nn.Dropout2d(dropout))
            self.activations.append(nn.Sequential(*act_layers))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = hidden_channels

        spatial = 28 // (2 ** num_conv_blocks)
        self.classifier = nn.Linear(hidden_channels * spatial * spatial, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, act, pool in zip(self.conv_blocks, self.activations, self.pools):
            x = pool(act(conv(x)))
        x = x.flatten(1)
        return self.classifier(x)

    def forward_with_intermediates_at_layer(
        self, x: torch.Tensor, layer_idx: int = 0
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run forward, returning intermediates from the specified HelixConv layer."""
        intermediates = {}
        for i, (conv, act, pool) in enumerate(
            zip(self.conv_blocks, self.activations, self.pools)
        ):
            if i == layer_idx:
                out, inter = conv.forward_with_intermediates(x)
                intermediates = inter
                x = pool(act(out))
            else:
                x = pool(act(conv(x)))
        x = x.flatten(1)
        logits = self.classifier(x)
        return logits, intermediates

    def quadrature_penalty(self) -> torch.Tensor:
        """Sum of per-block quadrature penalties (for soft regularization)."""
        return sum(block.quadrature_penalty() for block in self.conv_blocks)

    def quadrature_alignment(self) -> list[float]:
        """Per-block mean correlation between W_v and rotate(W_u, 90°)."""
        return [block.quadrature_alignment() for block in self.conv_blocks]


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

def build_rot_mnist_model(config: RotMNISTConfig) -> nn.Module:
    mt = config.model_type
    if mt == "standard_cnn":
        return StandardCNN(
            input_channels=config.input_channels,
            num_classes=config.num_classes,
            hidden_channels=config.hidden_channels,
            kernel_size=config.kernel_size,
            num_conv_blocks=config.num_conv_blocks,
            dropout=config.dropout,
        )
    elif mt == "standard_cnn_matched":
        return StandardCNN(
            input_channels=config.input_channels,
            num_classes=config.num_classes,
            hidden_channels=config.matched_hidden_channels,
            kernel_size=config.kernel_size,
            num_conv_blocks=config.num_conv_blocks,
            dropout=config.dropout,
        )
    elif mt == "circle_conv":
        return CircleCNN(
            input_channels=config.input_channels,
            num_classes=config.num_classes,
            circle_units=config.circle_units,
            hidden_channels=config.hidden_channels,
            kernel_size=config.kernel_size,
            num_conv_blocks=config.num_conv_blocks,
            dropout=config.dropout,
        )
    elif mt in ("helix_conv", "helix_conv_quadinit", "helix_conv_quadreg"):
        return HelixCNN(
            input_channels=config.input_channels,
            num_classes=config.num_classes,
            helix_units=config.helix_units,
            hidden_channels=config.hidden_channels,
            kernel_size=config.kernel_size,
            num_conv_blocks=config.num_conv_blocks,
            dropout=config.dropout,
            quadrature_init=(mt == "helix_conv_quadinit"),
        )
    else:
        raise ValueError(f"Unknown model type: {mt}")
