"""Multi-asset Parareal toolbox with FNO coarse propagators and dask-parallel fine solvers."""

from .black_scholes import (
    MultiAssetBlackScholesParams,
    basket_payoff,
    build_asset_mesh,
)
from .dataset import (
    TemporalFieldDataset,
    build_time_series,
    make_collocation_dataset,
)
from .fno import (
    FNO2d,
    FNOTrainingConfig,
    FNOCoarsePropagator,
    train_fno_model,
)
from .parareal import (
    MpiFineExecutor,
    MultiprocessingFineExecutor,
    PararealIntegrator,
)
from .solver import DaskSpatialFineSolver

__all__ = [
    "MultiAssetBlackScholesParams",
    "basket_payoff",
    "build_asset_mesh",
    "TemporalFieldDataset",
    "build_time_series",
    "make_collocation_dataset",
    "FNO2d",
    "FNOTrainingConfig",
    "FNOCoarsePropagator",
    "train_fno_model",
    "PararealIntegrator",
    "MultiprocessingFineExecutor",
    "MpiFineExecutor",
    "DaskSpatialFineSolver",
]
