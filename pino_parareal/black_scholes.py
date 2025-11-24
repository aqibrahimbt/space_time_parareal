from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class MultiAssetBlackScholesParams:
    """Parameters for a multi-asset Black–Scholes basket option."""

    num_assets: int = 2
    r: float = 0.03
    sigma: Sequence[float] | float = 0.4
    strike: float = 100.0
    maturity: float = 1.0
    s_max: Sequence[float] | float = 200.0
    weights: Sequence[float] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.sigma, float):
            self.sigma = tuple(float(self.sigma) for _ in range(self.num_assets))
        elif len(self.sigma) != self.num_assets:
            raise ValueError("sigma must match num_assets.")

        if isinstance(self.s_max, float):
            self.s_max = tuple(float(self.s_max) for _ in range(self.num_assets))
        elif len(self.s_max) != self.num_assets:
            raise ValueError("s_max must match num_assets.")

        if self.weights is None:
            self.weights = tuple(1.0 / self.num_assets for _ in range(self.num_assets))
        elif len(self.weights) != self.num_assets:
            raise ValueError("weights must match num_assets.")

    @property
    def sigma_values(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.sigma)

    @property
    def s_max_values(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.s_max)

    @property
    def weight_values(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.weights)


def build_asset_mesh(grid_axes: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Return the coordinate mesh for each asset axis (indexing='ij')."""
    return list(np.meshgrid(*grid_axes, indexing="ij"))


def basket_payoff(
    grid_axes: Sequence[np.ndarray],
    weights: Sequence[float],
    strike: float,
) -> np.ndarray:
    """European call payoff for a weighted basket of assets."""
    mesh = build_asset_mesh(grid_axes)
    stacked = np.stack(mesh, axis=-1)
    total = np.tensordot(stacked, np.asarray(weights), axes=([-1], [0]))
    return np.maximum(total - strike, 0.0)
