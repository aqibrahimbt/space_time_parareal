## Acknowledgements

This project has received funding from the [European High-Performance
Computing Joint Undertaking](https://eurohpc-ju.europa.eu/) (JU) under
grant agreement No 955701 ([TIME-X](https://www.time-x-eurohpc.eu/)).
The JU receives support from the European Union's Horizon 2020 research
and innovation programme and Belgium, France, Germany, and Switzerland.
This project also received funding from the [German Federal Ministry of
Education and Research](https://www.bmbf.de/bmbf/en/home/home_node.html)
(BMBF) grants  16HPC047 and 16ME0679K. Supported by the European Union - NextGenerationEU. 

<p align="center">
  <img src="EuroHPC.jpg" height="105"/> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="LogoTime-X.png" height="105" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="BMBF_gefoerdert_2017_en.jpg" height="105" />
</p>


# Space–Time Parareal for Multi-Asset Black-Scholes (PINO)

This repository implements a full space–time Parareal workflow for the multi-asset Black–Scholes equation by combining
dask-based spatial parallelism with a learned Physics-Informed Neural Operator (PINO) coarse propagator. The fine solver
runs an implicit Euler update chunked in space via dask, which allows scaling asset grid computation across available workers,
while the PINO (realized with a small Fourier Neural Operator) learns collocation samples produced by the fine solver and
drives the coarse update in the Parareal loop.

## System overview

- `main.py` exposes `train-fno` (builds the collocation dataset and trains the FNO coarse model) and `run-parareal` (launches
  the Parareal controller with the trained PINO and the dask fine propagator).
- `pinn_parareal/black_scholes.py` defines multi-asset parameters, payoff setup, and helper mesh builders for the basket option.
- `pinn_parareal/solver.py` implements `DaskSpatialFineSolver`, which generates chunked spatial updates, enforces boundaries,
  and can emit collocation snapshots for training.
- `pinn_parareal/dataset.py` runs the fine solver over coarse time slices, collects downtemporal snapshots, and exposes masks
  for interior collocation training.
- `pinn_parareal/fno.py` provides the FNO training config, dataset integration, and the PINO coarse propagator wrapper used by Parareal.
- `pinn_parareal/parareal.py` orchestrates the Parareal iteration, measuring coarse/fine runtimes and supporting optional executor hooks.
- `verify_solver.py` checks the Dask-based fine solver against the analytic 1D call price for sanity before scaling up.

## Requirements & setup

1. Create and activate a Python 3.11+ virtual environment (the repo ships with `.venv311` for convenience).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Training and evaluation rely on PyTorch and dask’s distributed scheduler; a CUDA-capable GPU is optional, and dask can run
   locally or connect to a cluster via `DASK_SCHEDULER_ADDRESS`.

## Step-by-step execution

### 1. Generate collocation data & train the FNO coarse propagator

The fine solver runs for the requested number of coarse time slices, emits paired states, and masks interior collocation points.
The FNO is trained on these samples (state + normalized time channels → next slice), making it a PINO-driven coarse propagator.

```bash
python main.py train-fno --output models/fno.pth --time-slices 8 --fine-steps 20 --collocation-fraction 0.6
```

Key arguments include `--spatial-steps`, `--chunk-size`, FNO architecture settings (`--modes-x`, `--width`, …), and training hyperparameters.

### 2. Run Parareal with the trained PINO coarse solver

Provide the checkpoint, adjust the time slicing, and optionally change worker counts for the fine solver:

```bash
python main.py run-parareal --model-path models/fno.pth --time-slices 12 --fine-steps 40 --iterations 3
```

The CLI logs per-iteration relative error, coarse/fine runtimes, and relies on the FNO to provide delta corrections between slices.

### 3. Verify the fine solver (optional)

```bash
python verify_solver.py
```

This compares the Dask-accelerated solver to the 1D analytical Black–Scholes call price and is helpful when tuning spatial discretization.

## Benchmarking

`benchmarks.py` runs both weak and strong scaling experiments, produces runtime/speedup figures, and compares the FNO coarse propagator to the numerical fine solver. The script generates the following plots under `reports/` by default:
1. `runtime_components.png`: coarse/fine/Parareal timings for a sweep of time slices.
2. `strong_speedup.png` and `strong_efficiency.png`: speedup and worker efficiency for the strong-scaling sweep.
3. `weak_scaling.png`: total Parareal time as the workload grows with more workers.
4. `fno_vs_numeric.png`: per-slice runtime ratio of the fine solver to the FNO.

Run it with the desired worker/time-slice configuration, e.g.:

```bash
python benchmarks.py --model-path models/fno.pth \
    --strong-workers 1,2,4,8 \
    --weak-workers 1,2,4 \
    --weak-time-slices 8,12,16 \
    --runtime-time-slices 6,8,10,12 \
    --chunk-size 32
```

The figures in `reports/` document runtime, speedup, and efficiency for later analysis.

## Notes

- The FNO is trained on collocation points sampled from the fine solver, blending supervised next-slice prediction with interior masks so the PINO learns the PDE structure without explicit residual losses.
- The fine solver boundary conditions are enforced at each time step and can be parallelized via Dask’s `delayed` tasks; adjust `--chunk-size` and spatial resolution depending on dataset size.
- Parareal integrates over the precomputed time grid and keeps track of coarse/fine timings so you can benchmark speedups once the PINO is stable.
