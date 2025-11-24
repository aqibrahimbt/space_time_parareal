from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from dask import delayed, compute

from .black_scholes import MultiAssetBlackScholesParams, basket_payoff, build_asset_mesh


class DaskSpatialFineSolver:
    """Explicit Euler fine propagator that parallelizes spatial updates with dask."""

    def __init__(
        self,
        params: MultiAssetBlackScholesParams,
        spatial_steps: int | Sequence[int] = 64,
        chunk_size: int | None = None,
        *,
        scheduler: str = "threads",
        num_workers: int = 1,
    ):
        self.params = params
        self.sigmas = params.sigma_values
        self.weights = np.asarray(params.weight_values, dtype=float)
        steps = (
            tuple(spatial_steps for _ in range(params.num_assets))
            if isinstance(spatial_steps, int)
            else tuple(spatial_steps)
        )
        if len(steps) != params.num_assets:
            raise ValueError("spatial_steps must be a scalar or match num_assets.")

        grid_axes = []
        grid_shape = []
        for axis_steps, s_max in zip(steps, params.s_max_values):
            points = axis_steps + 1
            if points < 3:
                raise ValueError("Each axis must allocate at least 3 grid points for interior updates.")
            axis_grid = np.linspace(0.0, s_max, points, dtype=float)
            grid_axes.append(axis_grid)
            grid_shape.append(points)

        self.grid_axes = grid_axes
        self.grid_shape = tuple(grid_shape)
        self.delta_s = tuple(axis[1] - axis[0] for axis in grid_axes)
        self.mesh = build_asset_mesh(grid_axes)
        self.initial_state = basket_payoff(grid_axes, self.weights, params.strike)
        self.chunk_size = chunk_size or max(4, self.grid_shape[0] // 4)
        self.chunk_ranges = list(self._build_axis_chunks())
        self._validate_solver()
        self.scheduler = scheduler
        self.num_workers = max(1, num_workers)

    def _validate_solver(self) -> None:
        if self.grid_shape[0] < 3:
            raise ValueError("Axis 0 requires at least three points for interior differentiation.")
        if not self.chunk_ranges:
            raise ValueError("No interior chunk ranges could be computed; increase spatial resolution.")

    def _build_axis_chunks(self) -> Iterable[tuple[int, int]]:
        start = 1
        end = self.grid_shape[0] - 1
        step = max(1, min(self.chunk_size, end - start))
        current = start
        while current < end:
            chunk_end = min(current + step, end)
            yield current, chunk_end
            current = chunk_end

    def _extract_mesh_patch(self, axis_range: tuple[int, int]) -> list[np.ndarray]:
        start, end = axis_range
        axis_slices = []
        for axis in range(self.params.num_assets):
            if axis == 0:
                axis_slices.append(slice(start, end))
            else:
                axis_slices.append(slice(1, -1))
        return [m[tuple(axis_slices)] for m in self.mesh]

    def _apply_boundary(self, state: np.ndarray, tau: float) -> None:
        discount = self.params.strike * np.exp(-self.params.r * tau)
        for axis in range(self.params.num_assets):
            left = [slice(None)] * self.params.num_assets
            left[axis] = slice(0, 1)
            state[tuple(left)] = 0.0
            right = [slice(None)] * self.params.num_assets
            right[axis] = slice(-1, None)
            coords = self._surface_coords(axis, slice(-1, None))
            boundary_values = np.maximum(self._weighted_sum(coords) - discount, 0.0)
            state[tuple(right)] = boundary_values

    def _surface_coords(self, axis: int, boundary_slice: slice) -> list[np.ndarray]:
        slices = [slice(None)] * self.params.num_assets
        slices[axis] = boundary_slice
        return [m[tuple(slices)] for m in self.mesh]

    def _weighted_sum(self, coords: Sequence[np.ndarray]) -> np.ndarray:
        stacked = np.stack(coords, axis=-1)
        return np.tensordot(stacked, self.weights, axes=([-1], [0]))

    def _step(self, state: np.ndarray, tau: float, dt: float) -> np.ndarray:
        next_state = state.copy()
        tasks = []
        assignments: list[tuple[int, int]] = []
        for chunk_range in self.chunk_ranges:
            start, end = chunk_range
            block_slice = [
                slice(start - 1, end + 1) if axis == 0 else slice(None)
                for axis in range(self.params.num_assets)
            ]
            block = state[tuple(block_slice)]
            mesh_patch = self._extract_mesh_patch(chunk_range)
            if mesh_patch[0].size == 0:
                continue
            boundary_slices = [
                slice(0, 1),
                *([slice(1, -1)] * (self.params.num_assets - 1)),
            ]
            left_boundary = block[tuple(boundary_slices)]
            boundary_slices[0] = slice(-1, None)
            right_boundary = block[tuple(boundary_slices)]
            tasks.append(
                delayed(self._update_chunk)(
                    block.copy(),
                    mesh_patch,
                    self.delta_s,
                    self.sigmas,
                    self.params.r,
                    dt,
                    self.grid_axes[0][start:end],
                    left_boundary,
                    right_boundary,
                )
            )
            assignments.append(chunk_range)
        if tasks:
            results = compute(
                *tasks,
                scheduler=self.scheduler,
                num_workers=self.num_workers,
            )
            for (start, end), patch in zip(assignments, results):
                assign_slice = [slice(start, end)] + [
                    slice(1, -1) for _ in range(self.params.num_assets - 1)
                ]
                next_state[tuple(assign_slice)] = patch
        self._apply_boundary(next_state, tau + dt)
        return next_state

    @staticmethod
    def _update_chunk(
        block: np.ndarray,
        mesh_patch: list[np.ndarray],
        delta_s: Sequence[float],
        sigmas: Sequence[float],
        r: float,
        dt: float,
        axis0_coords: np.ndarray,
        left_boundary: np.ndarray,
        right_boundary: np.ndarray,
    ) -> np.ndarray:
        ndim = block.ndim
        center_slices = tuple(slice(1, -1) for _ in range(ndim))
        center = block[center_slices]
        rest_update = np.zeros_like(center)
        for axis in range(1, ndim):
            left_slices = list(center_slices)
            right_slices = list(center_slices)
            left_slices[axis] = slice(0, -2)
            right_slices[axis] = slice(2, None)
            left = block[tuple(left_slices)]
            right = block[tuple(right_slices)]
            ds = delta_s[axis]
            sigma = sigmas[axis]
            s_values = mesh_patch[axis]
            rest_update += 0.5 * (sigma**2) * (s_values**2) * (right - 2 * center + left) / (ds**2)
            rest_update += r * s_values * (right - left) / (2 * ds)
        rhs = center + dt * rest_update
        return DaskSpatialFineSolver._solve_axis0_implicit(
            rhs,
            axis0_coords,
            delta_s[0],
            sigmas[0],
            r,
            dt,
            left_boundary,
            right_boundary,
        )

    @staticmethod
    def _solve_axis0_implicit(
        rhs: np.ndarray,
        axis0_coords: np.ndarray,
        delta_s0: float,
        sigma0: float,
        r: float,
        dt: float,
        left_boundary: np.ndarray,
        right_boundary: np.ndarray,
    ) -> np.ndarray:
        n0 = axis0_coords.shape[0]
        if n0 == 0:
            return rhs
        flat_rhs = rhs.reshape(n0, -1)
        if left_boundary.size:
            flat_rhs[0] -= lower[0] * left_boundary.reshape(-1)
        if right_boundary.size:
            flat_rhs[-1] -= upper[-1] * right_boundary.reshape(-1)
        a = 0.5 * sigma0**2 * axis0_coords**2 / (delta_s0**2)
        b = r * axis0_coords / (2 * delta_s0)
        lower = -dt * (a - b)
        upper = -dt * (a + b)
        center_diag = 1 + dt * (2 * a + r)
        num_cols = flat_rhs.shape[1]
        cp = np.empty(n0 - 1, dtype=rhs.dtype)
        dp = np.empty((n0, num_cols), dtype=rhs.dtype)
        cp[0] = upper[0] / center_diag[0]
        dp[0] = flat_rhs[0] / center_diag[0]
        for i in range(1, n0):
            denom = center_diag[i] - lower[i - 1] * cp[i - 1]
            if i < n0 - 1:
                cp[i] = upper[i] / denom
            dp[i] = (flat_rhs[i] - lower[i - 1] * dp[i - 1]) / denom
        solution = np.empty_like(flat_rhs)
        solution[-1] = dp[-1]
        for i in range(n0 - 2, -1, -1):
            solution[i] = dp[i] - cp[i] * solution[i + 1]
        return solution.reshape(rhs.shape)

    def propagate(
        self, u_init: np.ndarray, tau_start: float, tau_end: float, steps: int
    ) -> np.ndarray:
        steps = max(1, steps)
        dt = (tau_end - tau_start) / steps
        tau = tau_start
        state = u_init.copy()
        self._apply_boundary(state, tau)
        for _ in range(steps):
            state = self._step(state, tau, dt)
            tau += dt
        return state

    def solve_full(self, steps: int) -> np.ndarray:
        return self.propagate(self.initial_state, 0.0, self.params.maturity, steps)
