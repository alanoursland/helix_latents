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
    ):
        super().__init__()
        self.eps = eps
        self.units = units

        self.conv_u = nn.Conv2d(in_channels, units, kernel_size, padding=padding)
        self.conv_v = nn.Conv2d(in_channels, units, kernel_size, padding=padding)
        self.conv_w = nn.Conv2d(in_channels, units, kernel_size, padding=padding)

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
    elif mt == "helix_conv":
        return HelixCNN(
            input_channels=config.input_channels,
            num_classes=config.num_classes,
            helix_units=config.helix_units,
            hidden_channels=config.hidden_channels,
            kernel_size=config.kernel_size,
            num_conv_blocks=config.num_conv_blocks,
            dropout=config.dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {mt}")
