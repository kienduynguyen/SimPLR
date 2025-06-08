import torch
import torch.nn as nn
import torch.nn.functional as F

from e2edet.utils.general import view_with_shape


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MLPv2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(n, k), nn.LayerNorm(k))
            for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.gelu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SegmentMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, kernel_size=1):
        super().__init__()
        self.num_layers = num_layers
        in_dim = [hidden_dim] * (num_layers - 1)
        layers = [
            nn.Sequential(
                nn.ConvTranspose2d(input_dim, hidden_dim, 2, stride=2),
                nn.GELU(),
            )
        ]

        layers.extend(
            [
                nn.Sequential(
                    nn.Conv2d(n, k, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.GELU(),
                )
                for n, k in zip(in_dim, in_dim)
            ]
        )
        layers.append(nn.Conv2d(hidden_dim, output_dim, kernel_size=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        b, l, s, _, c = x.shape
        x = x.view(-1, s, s, c).permute(0, 3, 1, 2).contiguous()
        x = self.layers(x).view(b, l, -1, 2 * s, 2 * s)

        return x


class SegmentMLPv2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(input_dim, hidden_dim, 2, stride=2),
            nn.GroupNorm(32, hidden_dim),
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=hidden_dim
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1),
        )

    def forward(self, x):
        b, l, s, _, c = x.shape
        x = x.view(-1, s, s, c).permute(0, 3, 1, 2).contiguous()
        x = self.layers(x).view(b, l, -1, 2 * s, 2 * s)

        return x
