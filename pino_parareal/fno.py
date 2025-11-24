from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .black_scholes import MultiAssetBlackScholesParams
from .dataset import TemporalFieldDataset


class SpectralConv2d(nn.Module):
    """Fourier layer adapted from the FNO paper (2D version)."""

    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat) * self.scale
        )

    def compl_mul2d(self, input_ft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input_ft, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=x_ft.dtype,
            device=x.device,
        )
        out_ft[:, :, : self.modes_x, : self.modes_y] = self.compl_mul2d(
            x_ft[:, :, : self.modes_x, : self.modes_y], self.weights
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNO2d(nn.Module):
    """Basic Fourier Neural Operator for 2D fields."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        modes_x: int = 16,
        modes_y: int = 16,
        width: int = 48,
    ):
        super().__init__()
        self.width = width
        self.fc0 = nn.Linear(in_channels, width)
        self.conv0 = SpectralConv2d(width, width, modes_x, modes_y)
        self.conv1 = SpectralConv2d(width, width, modes_x, modes_y)
        self.conv2 = SpectralConv2d(width, width, modes_x, modes_y)
        self.conv3 = SpectralConv2d(width, width, modes_x, modes_y)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = self._block(x, self.conv0, self.w0)
        x = self._block(x, self.conv1, self.w1)
        x = self._block(x, self.conv2, self.w2)
        x = self._block(x, self.conv3, self.w3)
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2)

    @staticmethod
    def _block(
        x: torch.Tensor,
        spectral: SpectralConv2d,
        pointwise: nn.Conv2d,
    ) -> torch.Tensor:
        return F.gelu(spectral(x) + pointwise(x))


@dataclass
class FNOTrainingConfig:
    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    log_interval: int = 5
    modes_x: int = 16
    modes_y: int = 16
    width: int = 48


def _build_fno_input(
    state: torch.Tensor,
    tau: torch.Tensor,
    dt: torch.Tensor,
    maturity: float,
) -> torch.Tensor:
    tau_normalized = (tau / maturity).view(-1, 1, 1, 1).to(state.dtype)
    dt_normalized = (dt / maturity).view(-1, 1, 1, 1).to(state.dtype)
    tau_channel = tau_normalized.expand(-1, 1, state.shape[-2], state.shape[-1])
    dt_channel = dt_normalized.expand(-1, 1, state.shape[-2], state.shape[-1])
    return torch.cat([state, tau_channel, dt_channel], dim=1)


def train_fno_model(
    params: MultiAssetBlackScholesParams,
    dataset: TemporalFieldDataset,
    config: FNOTrainingConfig,
    device: torch.device,
) -> FNO2d:
    model = FNO2d(
        in_channels=3,
        out_channels=1,
        modes_x=config.modes_x,
        modes_y=config.modes_y,
        width=config.width,
    ).to(device)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        for state, target, tau, dt, mask in dataloader:
            state = state.to(device)
            target = target.to(device)
            tau = tau.to(device)
            dt = dt.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            inputs = _build_fno_input(state, tau, dt, dataset.maturity).to(device)
            pred = model(inputs)
            diff = mask * (pred - target)
            loss = (diff**2).mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if epoch % config.log_interval == 0 or epoch == 1:
            avg_loss = running_loss / len(dataloader)
            print(f"Epoch {epoch}/{config.epochs} loss={avg_loss:.4e}")
    return model


class FNOCoarsePropagator:
    """Wraps a trained FNO and projects numpy arrays through it."""

    def __init__(
        self,
        model: FNO2d,
        params: MultiAssetBlackScholesParams,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.params = params
        self.device = device

    def __call__(self, state: np.ndarray, tau_start: float, tau_end: float) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.from_numpy(state.astype("float32")).unsqueeze(0).to(self.device)
            delta = torch.tensor([tau_end - tau_start], dtype=tensor.dtype, device=self.device)
            tau = torch.tensor([tau_start], dtype=tensor.dtype, device=self.device)
            inputs = _build_fno_input(tensor, tau, delta, self.params.maturity)
            output = self.model(inputs)
        return output.squeeze(0).squeeze(0).cpu().numpy()
