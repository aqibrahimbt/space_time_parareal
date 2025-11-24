import concurrent.futures
import os
import time
from typing import Callable, Iterable, Sequence, cast

import numpy as np
try:
    from mpi4py.futures import MPIPoolExecutor
except ImportError:  # pragma: no cover - optional dependency
    MPIPoolExecutor = None

Propagator = Callable[[np.ndarray, float, float], np.ndarray]
FinePropagator = Callable[[np.ndarray, float, float, int], np.ndarray]

FineParallelTask = tuple[np.ndarray, float, float, int]


class FineParallelExecutor:
    def run(self, tasks: Sequence[FineParallelTask]) -> list[np.ndarray]:
        raise NotImplementedError


class MultiprocessingFineExecutor(FineParallelExecutor):
    def __init__(self, fine_func: FinePropagator, max_workers: int | None = None):
        self.fine_func = fine_func
        self.max_workers = max_workers or max(1, os.cpu_count() or 1)

    def run(self, tasks: Sequence[FineParallelTask]) -> list[np.ndarray]:
        serialized = [
            (np.ascontiguousarray(state), start, end, steps)
            for state, start, end, steps in tasks
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.fine_func, state, start, end, steps): idx
                for idx, (state, start, end, steps) in enumerate(serialized)
            }
            results: list[np.ndarray | None] = [None] * len(serialized)
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return [cast(np.ndarray, entry) for entry in results]


class MpiFineExecutor(FineParallelExecutor):
    def __init__(self, fine_func: FinePropagator, max_workers: int | None = None):
        if MPIPoolExecutor is None:
            raise RuntimeError("MPI parallel mode requires mpi4py to be installed.")
        self.fine_func = fine_func
        self.max_workers = max_workers or max(1, os.cpu_count() or 1)

    def run(self, tasks: Sequence[FineParallelTask]) -> list[np.ndarray]:
        serialized = [
            (np.ascontiguousarray(state), start, end, steps)
            for state, start, end, steps in tasks
        ]
        with MPIPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.fine_func, state, start, end, steps): idx
                for idx, (state, start, end, steps) in enumerate(serialized)
            }
            results: list[np.ndarray | None] = [None] * len(serialized)
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return [cast(np.ndarray, entry) for entry in results]


class PararealIntegrator:
    def __init__(
        self,
        time_points: Iterable[float],
        initial_state: np.ndarray,
        coarse: Propagator,
        fine: FinePropagator,
        fine_steps_per_slice: int = 20,
        parallel_fine: FineParallelExecutor | None = None,
    ):
        self.time_points = np.asarray(time_points, dtype=float)
        self.initial_state = initial_state
        self.coarse = coarse
        self.fine = fine
        self.fine_steps_per_slice = max(1, fine_steps_per_slice)
        self.metrics = {"coarse_secs": 0.0, "fine_secs": 0.0}
        self.parallel_fine = parallel_fine

    def _call_coarse(self, state: np.ndarray, start: float, end: float) -> np.ndarray:
        before = time.perf_counter()
        result = self.coarse(state, start, end)
        self.metrics["coarse_secs"] += time.perf_counter() - before
        return result

    def _call_fine(self, state: np.ndarray, start: float, end: float) -> np.ndarray:
        before = time.perf_counter()
        result = self.fine(state, start, end, self.fine_steps_per_slice)
        self.metrics["fine_secs"] += time.perf_counter() - before
        return result

    def _execute_parallel_fine(self, tasks: list[FineParallelTask]) -> list[np.ndarray]:
        assert self.parallel_fine is not None
        before = time.perf_counter()
        results = self.parallel_fine.run(tasks)
        self.metrics["fine_secs"] += time.perf_counter() - before
        return results

    def _reference_solution(self) -> np.ndarray:
        total_steps = self.fine_steps_per_slice * (len(self.time_points) - 1)
        return self.fine(self.initial_state, self.time_points[0], self.time_points[-1], total_steps)

    def run(self, iterations: int = 3) -> tuple[list[np.ndarray], list[float]]:
        slices = len(self.time_points) - 1
        solutions: list[np.ndarray] = []
        errors: list[float] = []
        current: list[np.ndarray] = [self.initial_state]
        # initial coarse propagation to bootstrap
        for idx in range(slices):
            next_state = self._call_coarse(current[idx], self.time_points[idx], self.time_points[idx + 1])
            current.append(next_state)
        reference = self._reference_solution()
        solutions.append(np.stack(current))
        errors.append(self._metric(current[-1], reference))
        for _ in range(iterations):
            new_solution = [self.initial_state]
            tasks = [
                (current[idx], self.time_points[idx], self.time_points[idx + 1], self.fine_steps_per_slice)
                for idx in range(slices)
            ]
            if self.parallel_fine is None:
                fine_states = [
                    self._call_fine(state, start, end)
                    for state, start, end, _ in tasks
                ]
            else:
                fine_states = self._execute_parallel_fine(tasks)
            for idx in range(slices):
                fine_state = fine_states[idx]
                coarse_new = self._call_coarse(new_solution[idx], self.time_points[idx], self.time_points[idx + 1])
                coarse_old = self._call_coarse(current[idx], self.time_points[idx], self.time_points[idx + 1])
                updated = coarse_new + fine_state - coarse_old
                new_solution.append(updated)
            current = new_solution
            solutions.append(np.stack(current))
            errors.append(self._metric(current[-1], reference))
        return solutions, errors

    @staticmethod
    def _metric(solution: np.ndarray, reference: np.ndarray) -> float:
        norm = np.linalg.norm(reference)
        if norm == 0:
            return float(np.linalg.norm(solution))
        return float(np.linalg.norm(solution - reference) / norm)
