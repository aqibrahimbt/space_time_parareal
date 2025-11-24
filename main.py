"""CLI coordinating training of the FNO coarse solver and Parareal experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from pino_parareal import (
    DaskSpatialFineSolver,
    FNO2d,
    FNOCoarsePropagator,
    FNOTrainingConfig,
    MultiAssetBlackScholesParams,
    PararealIntegrator,
    make_collocation_dataset,
    train_fno_model,
)


def _parse_sequence(value: str, length: int, dtype):
    parts = [part.strip() for part in value.split(",") if part.strip()]
    parsed = [dtype(part) for part in parts]
    if len(parsed) == 1 and length > 1:
        parsed = parsed * length
    if len(parsed) != length:
        raise argparse.ArgumentTypeError(f"expected {length} values but got {len(parsed)}")
    return tuple(parsed)


def _build_params(args: argparse.Namespace) -> tuple[MultiAssetBlackScholesParams, tuple[int, ...]]:
    sigma = _parse_sequence(args.sigma, args.num_assets, float)
    s_max = _parse_sequence(args.s_max, args.num_assets, float)
    steps = _parse_sequence(args.spatial_steps, args.num_assets, int)
    weights = None
    if args.weights:
        weights = _parse_sequence(args.weights, args.num_assets, float)
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


def _choose_device(use_cuda: bool) -> torch.device:
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(args: argparse.Namespace) -> None:
    params, steps = _build_params(args)
    solver = DaskSpatialFineSolver(params, spatial_steps=steps, chunk_size=args.chunk_size)
    dataset = make_collocation_dataset(
        solver,
        time_slices=args.time_slices,
        fine_steps_per_slice=args.fine_steps,
        collocation_fraction=args.collocation_fraction,
    )
    config = FNOTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        log_interval=args.log_interval,
        modes_x=args.modes_x,
        modes_y=args.modes_y,
        width=args.width,
    )
    device = _choose_device(args.use_cuda)
    model = train_fno_model(params, dataset, config, device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved FNO checkpoint to {output_path}")


def run(args: argparse.Namespace) -> None:
    params, steps = _build_params(args)
    solver = DaskSpatialFineSolver(params, spatial_steps=steps, chunk_size=args.chunk_size)
    device = _choose_device(args.use_cuda)
    model = FNO2d(in_channels=3, out_channels=1, modes_x=args.modes_x, modes_y=args.modes_y, width=args.width)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    coarse = FNOCoarsePropagator(model, params, device)
    coarse_fn = lambda state, start, end: coarse(state, start, end)
    time_points = np.linspace(0.0, params.maturity, args.time_slices + 1)
    integrator = PararealIntegrator(
        time_points,
        solver.initial_state,
        coarse_fn,
        solver.propagate,
        fine_steps_per_slice=args.fine_steps,
        parallel_fine=None,
    )
    solutions, errors = integrator.run(iterations=args.iterations)
    print(f"Parareal iterations completed; final relative error {errors[-1]:.4e}")
    print(f"Coarse time: {integrator.metrics['coarse_secs']:.2f}s, fine time: {integrator.metrics['fine_secs']:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-cuda", action="store_true", help="Prefer CUDA when available.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train-fno", help="Train the FNO coarse propagator.")
    train_parser.add_argument("--output", default="models/fno.pth")
    train_parser.add_argument("--num-assets", type=int, default=2)
    train_parser.add_argument("--r", type=float, default=0.03)
    train_parser.add_argument("--sigma", default="0.4")
    train_parser.add_argument("--s-max", default="200.0")
    train_parser.add_argument("--strike", type=float, default=100.0)
    train_parser.add_argument("--maturity", type=float, default=1.0)
    train_parser.add_argument("--weights", default=None)
    train_parser.add_argument("--spatial-steps", default="64")
    train_parser.add_argument("--chunk-size", type=int, default=32)
    train_parser.add_argument("--time-slices", type=int, default=8)
    train_parser.add_argument("--fine-steps", type=int, default=20)
    train_parser.add_argument("--collocation-fraction", type=float, default=0.5)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--learning-rate", type=float, default=5e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.0)
    train_parser.add_argument("--log-interval", type=int, default=5)
    train_parser.add_argument("--modes-x", type=int, default=16)
    train_parser.add_argument("--modes-y", type=int, default=16)
    train_parser.add_argument("--width", type=int, default=48)
    train_parser.set_defaults(func=train)

    run_parser = subparsers.add_parser("run-parareal", help="Run time-parallel Parareal with the trained FNO.")
    run_parser.add_argument("--model-path", required=True)
    run_parser.add_argument("--num-assets", type=int, default=2)
    run_parser.add_argument("--r", type=float, default=0.03)
    run_parser.add_argument("--sigma", default="0.4")
    run_parser.add_argument("--s-max", default="200.0")
    run_parser.add_argument("--strike", type=float, default=100.0)
    run_parser.add_argument("--maturity", type=float, default=1.0)
    run_parser.add_argument("--weights", default=None)
    run_parser.add_argument("--spatial-steps", default="64")
    run_parser.add_argument("--chunk-size", type=int, default=32)
    run_parser.add_argument("--time-slices", type=int, default=8)
    run_parser.add_argument("--fine-steps", type=int, default=20)
    run_parser.add_argument("--iterations", type=int, default=3)
    run_parser.add_argument("--modes-x", type=int, default=16)
    run_parser.add_argument("--modes-y", type=int, default=16)
    run_parser.add_argument("--width", type=int, default=48)
    run_parser.set_defaults(func=run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
