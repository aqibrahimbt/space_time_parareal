from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset

from .solver import DaskSpatialFineSolver


class TemporalFieldDataset(Dataset):
    """Dataset of successive field snapshots, optionally filtered by collocation masks."""

    def __init__(
        self,
        states: Sequence[torch.Tensor],
        times: Sequence[float],
        collocation_fraction: float = 1.0,
    ):
        if len(states) != len(times):
            raise ValueError("States and times must have matching lengths.")
        if len(states) < 2:
            raise ValueError("At least two states are required for dataset construction.")
        self.states = states
        self.times = times
        self.collocation_fraction = max(0.0, min(1.0, collocation_fraction))
        self.maturity = float(times[-1])

    def __len__(self) -> int:
        return len(self.states) - 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, float, float, torch.Tensor]:
        three = self.states[idx]
        next_state = self.states[idx + 1]
        tau = float(self.times[idx])
        dt = float(self.times[idx + 1] - self.times[idx])
        field = three.unsqueeze(0)
        target = next_state.unsqueeze(0)
        mask = (torch.rand_like(field) < self.collocation_fraction)
        return field, target, tau, dt, mask


def build_time_series(
    solver: DaskSpatialFineSolver,
    time_slices: int,
    fine_steps_per_slice: int,
) -> tuple[list[torch.Tensor], list[float]]:
    """Run the fine solver and collect field snapshots for each coarse slice."""
    dt = solver.params.maturity / time_slices
    states: list[torch.Tensor] = []
    times: list[float] = []
    state = solver.initial_state
    tau = 0.0
    states.append(torch.from_numpy(state.astype("float32")))
    times.append(tau)
    for _ in range(time_slices):
        tau_next = tau + dt
        state = solver.propagate(state, tau, tau_next, fine_steps_per_slice)
        tau = tau_next
        states.append(torch.from_numpy(state.astype("float32")))
        times.append(tau)
    return states, times


def make_collocation_dataset(
    solver: DaskSpatialFineSolver,
    time_slices: int,
    fine_steps_per_slice: int,
    collocation_fraction: float = 1.0,
) -> TemporalFieldDataset:
    """Helper that builds the temporal dataset based on solver snapshots."""
    states, times = build_time_series(solver, time_slices, fine_steps_per_slice)
    return TemporalFieldDataset(states, times, collocation_fraction=collocation_fraction)
