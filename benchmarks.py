"""Benchmarking tools for PINO-based Parareal: runtime, scaling, and speedup plots."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from pino_parareal import (
    DaskSpatialFineSolver,
    FNO2d,
    FNOCoarsePropagator,
    MultiAssetBlackScholesParams,
    PararealIntegrator,
)


def _parse_sequence(value: str, dtype) -> list:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    parsed = [dtype(part) for part in parts]
    if not parsed:
        raise argparse.ArgumentTypeError("sequence must contain at least one value")
    return parsed


def _build_params(args: argparse.Namespace) -> tuple[MultiAssetBlackScholesParams, tuple[int, ...]]:
    sigma = tuple(float(part) for part in _parse_sequence(args.sigma, float))
    s_max = tuple(float(part) for part in _parse_sequence(args.s_max, float))
    steps = tuple(int(part) for part in _parse_sequence(args.spatial_steps, int))
    weights = None
    if args.weights:
        weights = tuple(float(part) for part in _parse_sequence(args.weights, float))
    params = MultiAssetBlackScholesParams(
        num_assets=args.num_assets,
        r=args.r,
        sigma=sigma,
        strike=args.strike,
        maturity=args.maturity,
        s_max=s_max,
        weights=weights,
    )
    return params, steps


def _load_coarse(args: argparse.Namespace, params: MultiAssetBlackScholesParams) -> FNOCoarsePropagator:
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    model = FNO2d(
        in_channels=3,
        out_channels=1,
        modes_x=args.modes_x,
        modes_y=args.modes_y,
        width=args.width,
    )
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    return FNOCoarsePropagator(model, params, device)


def _make_solver(
    params: MultiAssetBlackScholesParams,
    spatial_steps: tuple[int, ...],
    chunk_size: int,
    scheduler: str,
    workers: int,
) -> DaskSpatialFineSolver:
    return DaskSpatialFineSolver(
        params,
        spatial_steps=spatial_steps,
        chunk_size=chunk_size,
        scheduler=scheduler,
        num_workers=workers,
    )


def _run_parareal(
    solver: DaskSpatialFineSolver,
    coarse: FNOCoarsePropagator,
    params: MultiAssetBlackScholesParams,
    time_slices: int,
    fine_steps: int,
    iterations: int,
) -> tuple[dict, float]:
    time_points = np.linspace(0.0, params.maturity, time_slices + 1)
    coarse_fn = lambda state, start, end: coarse(state, start, end)
    integrator = PararealIntegrator(
        time_points,
        solver.initial_state,
        coarse_fn,
        solver.propagate,
        fine_steps_per_slice=fine_steps,
        parallel_fine=None,
    )
    integrator.run(iterations=iterations)
    parareal_time = integrator.metrics["coarse_secs"] + integrator.metrics["fine_secs"]
    return integrator.metrics.copy(), parareal_time


def _time_full_fine(solver: DaskSpatialFineSolver, steps: int) -> float:
    t0 = perf_counter()
    solver.solve_full(steps)
    return perf_counter() - t0


def _plot_runtime(runtime_data: list[dict], output: Path) -> None:
    plt.figure()
    slices = [entry["time_slices"] for entry in runtime_data]
    coarse = [entry["coarse_secs"] for entry in runtime_data]
    fine = [entry["fine_secs"] for entry in runtime_data]
    parareal = [entry["parareal"] for entry in runtime_data]
    plt.plot(slices, coarse, marker="o", label="FNO coarse")
    plt.plot(slices, fine, marker="s", label="Fine (per slice)")
    plt.plot(slices, parareal, marker="^", label="Parareal (total)")
    plt.xlabel("Time slices")
    plt.ylabel("Time (s)")
    plt.title("Runtime components vs time slices")
    plt.legend()
    plt.grid(True)
    output.mkdir(parents=True, exist_ok=True)
    plt.savefig(output / "runtime_components.png")
    plt.close()


def _plot_scaling(strong_data: list[dict], weak_data: list[dict], output: Path) -> None:
    if strong_data:
        plt.figure()
        workers = [entry["workers"] for entry in strong_data]
        speedup = [entry["speedup"] for entry in strong_data]
        plt.plot(workers, speedup, marker="o")
        plt.xlabel("Workers")
        plt.ylabel("Speedup")
        plt.title("Strong scaling speedup")
        plt.grid(True)
        plt.savefig(output / "strong_speedup.png")
        plt.close()
        plt.figure()
        efficiency = [entry["efficiency"] for entry in strong_data]
        plt.plot(workers, efficiency, marker="s")
        plt.xlabel("Workers")
        plt.ylabel("Efficiency")
        plt.title("Strong scaling efficiency")
        plt.grid(True)
        plt.savefig(output / "strong_efficiency.png")
        plt.close()
    if weak_data:
        plt.figure()
        workers = [entry["workers"] for entry in weak_data]
        parareal_time = [entry["parareal_time"] for entry in weak_data]
        plt.plot(workers, parareal_time, marker="o")
        plt.xlabel("Workers")
        plt.ylabel("Parareal time (s)")
        plt.title("Weak scaling (performance per worker)")
        plt.grid(True)
        plt.savefig(output / "weak_scaling.png")
        plt.close()


def _plot_fno_vs_numeric(runtime_data: list[dict], output: Path) -> None:
    plt.figure()
    slices = [entry["time_slices"] for entry in runtime_data]
    ratio = [entry["fine_secs"] / max(entry["coarse_secs"], 1e-9) for entry in runtime_data]
    plt.plot(slices, ratio, marker="o")
    plt.xlabel("Time slices")
    plt.ylabel("Fine / Coarse runtime")
    plt.title("FNO vs numeric per-slice runtime")
    plt.grid(True)
    plt.savefig(output / "fno_vs_numeric.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--num-assets", type=int, default=2)
    parser.add_argument("--r", type=float, default=0.03)
    parser.add_argument("--sigma", default="0.4,0.4")
    parser.add_argument("--s-max", default="200.0,200.0")
    parser.add_argument("--strike", type=float, default=100.0)
    parser.add_argument("--maturity", type=float, default=1.0)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--spatial-steps", default="64,64")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--time-slices", type=int, default=8)
    parser.add_argument("--fine-steps", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--strong-workers", default="1,2,4")
    parser.add_argument("--weak-workers", default="1,2,4")
    parser.add_argument("--weak-time-slices", default="8,12,16")
    parser.add_argument("--runtime-time-slices", default="6,8,10,12")
    parser.add_argument("--scheduler", default="threads")
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--modes-x", type=int, default=16)
    parser.add_argument("--modes-y", type=int, default=16)
    parser.add_argument("--width", type=int, default=48)
    args = parser.parse_args()

    params, steps = _build_params(args)
    coarse = _load_coarse(args, params)

    runtime_time_slices = _parse_sequence(args.runtime_time_slices, int)
    strong_workers = _parse_sequence(args.strong_workers, int)
    weak_workers = _parse_sequence(args.weak_workers, int)
    weak_slices = _parse_sequence(args.weak_time_slices, int)

    default_worker = strong_workers[0] if strong_workers else 1
    runtime_results = []
    for slices in runtime_time_slices:
        solver = _make_solver(params, steps, args.chunk_size, args.scheduler, default_worker)
        metrics, parareal_time = _run_parareal(solver, coarse, params, slices, args.fine_steps, args.iterations)
        runtime_results.append(
            {
                "time_slices": slices,
                "coarse_secs": metrics["coarse_secs"],
                "fine_secs": metrics["fine_secs"],
                "parareal": parareal_time,
            }
        )

    strong_results = []
    baseline_time = None
    for workers in strong_workers:
        solver = _make_solver(params, steps, args.chunk_size, args.scheduler, workers)
        metrics, parareal_time = _run_parareal(solver, coarse, params, args.time_slices, args.fine_steps, args.iterations)
        if baseline_time is None:
            baseline_time = parareal_time
        strong_results.append(
            {
                "workers": workers,
                "parareal_time": parareal_time,
                "speedup": baseline_time / parareal_time if baseline_time else 1.0,
                "efficiency": (baseline_time / parareal_time if baseline_time else 1.0) / workers,
            }
        )

    weak_results = []
    for workers, slices in zip(weak_workers, weak_slices):
        solver = _make_solver(params, steps, args.chunk_size, args.scheduler, workers)
        _, parareal_time = _run_parareal(solver, coarse, params, slices, args.fine_steps, args.iterations)
        weak_results.append(
            {
                "workers": workers,
                "time_slices": slices,
                "parareal_time": parareal_time,
            }
        )

    output_path = Path(args.output_dir)
    _plot_runtime(runtime_results, output_path)
    _plot_scaling(strong_results, weak_results, output_path)
    _plot_fno_vs_numeric(runtime_results, output_path)

    print(f"Benchmark figures written to {output_path}")


if __name__ == "__main__":
    main()
