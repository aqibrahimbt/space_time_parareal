"""Sanity check for the Dask-based fine solver against the one-dimensional analytic solution."""

from math import erf, log, sqrt

import numpy as np

from pino_parareal import DaskSpatialFineSolver, MultiAssetBlackScholesParams


def norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def analytic_call(S: np.ndarray, K: float, T: float, r: float, sigma: float) -> np.ndarray:
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)


def verify() -> None:
    params = MultiAssetBlackScholesParams(
        num_assets=1,
        r=0.05,
        sigma=0.2,
        strike=100.0,
        maturity=1.0,
        s_max=300.0,
    )
    solver = DaskSpatialFineSolver(params, spatial_steps=500, chunk_size=128)
    solution = solver.solve_full(steps=100)
    s_grid = solver.grid_axes[0]
    analytic = analytic_call(s_grid, params.strike, params.maturity, params.r, params.sigma_values[0])
    analytic[0] = 0.0  # handle log(0)
    idx = np.abs(s_grid - 100.0).argmin()
    print(f"S ≈ {s_grid[idx]:.2f}")
    print(f"Numeric: {solution[idx]:.4f}")
    print(f"Analytic: {analytic[idx]:.4f}")
    error = abs(solution[idx] - analytic[idx])
    print(f"Error: {error:.4f}")
    if error > 1.0:
        print("FAIL: Error exceeded tolerance.")
    else:
        print("PASS")


if __name__ == "__main__":
    verify()
