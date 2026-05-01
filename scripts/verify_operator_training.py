"""
Smoke verification for the LandscapeNeuralOperator training contract.

It creates a tiny synthetic dataset, trains for one epoch with the current
PDE residual enabled, and verifies that a checkpoint is written.
"""

import argparse
import os
import shutil
import sys
import tempfile

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)


def _make_synthetic_dataset(data_dir, num_samples=4, shape=(8, 8), num_steps=4):
    import torch
    from src.physics import calculate_dhdt_physics

    os.makedirs(data_dir, exist_ok=True)
    height, width = shape
    dx = dy = 100.0
    dt = torch.tensor(10.0, dtype=torch.float32)
    da_params = {
        "method": "soft_mfd",
        "num_iters": 8,
        "temperature": 0.05,
        "slope_power": 1.0,
        "leak_rate": 0.001,
        "positive_fn": "relu",
        "verbose": False,
    }

    for i in range(num_samples):
        initial = torch.rand(1, height, width, dtype=torch.float32) * 10.0
        h = initial.unsqueeze(0)
        uplift = torch.tensor(1e-4 * (i + 1), dtype=torch.float32)
        k_f = torch.tensor(1e-6 * (i + 1), dtype=torch.float32)
        k_d = torch.tensor(1e-3, dtype=torch.float32)
        m = torch.tensor(0.5, dtype=torch.float32)
        n = torch.tensor(1.0, dtype=torch.float32)
        frames = [initial]
        with torch.no_grad():
            for _ in range(num_steps - 1):
                dhdt = calculate_dhdt_physics(
                    h=h,
                    U=uplift,
                    K_f=k_f,
                    m=m,
                    n=n,
                    K_d=k_d,
                    dx=dx,
                    dy=dy,
                    precip=1.0,
                    da_params=da_params,
                )
                h = h + dhdt * dt
                frames.append(h.squeeze(0))
        final = frames[-1]
        trajectory = torch.stack(frames, dim=0)

        torch.save({
            "initial_topo": initial,
            "final_topo": final,
            "trajectory_topo": trajectory,
            "time": torch.arange(num_steps, dtype=torch.float32) * dt,
            "dt": dt,
            "run_time": dt,
            "uplift_rate": uplift,
            "k_f": k_f,
            "k_d": k_d,
            "m": m,
            "n": n,
        }, os.path.join(data_dir, f"sample_{i:03d}.pt"))


def run_verification(work_dir):
    import torch
    from src.data_utils import create_dataloaders
    from src.models import build_model_from_config, cast_floating_module_dtype
    from src.trainer import PINNTrainer
    from src.utils import set_seed, setup_logging

    data_dir = os.path.join(work_dir, "data")
    results_dir = os.path.join(work_dir, "results")
    _make_synthetic_dataset(data_dir)

    config = {
        "data": {
            "processed_dir": data_dir,
            "sample_mode": "trajectory",
            "train_split": 0.75,
            "val_split": 0.25,
            "num_workers": 0,
            "normalization": {"enabled": False},
        },
        "model": {
            "name": "LandscapeNeuralOperator",
            "hidden_channels": 8,
            "channel_multipliers": [1, 2],
            "fno_modes": [4, 4],
            "fno_layers": 1,
            "use_flow_graph": True,
            "flow_graph_hidden_channels": 4,
            "flow_graph_iters": 8,
            "domain_x": [0.0, 800.0],
            "domain_y": [0.0, 800.0],
            "dtype": "float32",
        },
        "physics": {
            "dx": 100.0,
            "dy": 100.0,
            "precip": 1.0,
            "drainage_area_kwargs": {
                "method": "soft_mfd",
                "num_iters": 8,
                "temperature": 0.05,
                "slope_power": 1.0,
                "leak_rate": 0.001,
                "positive_fn": "relu",
                "verbose": False,
            },
        },
        "training": {
            "device": "cpu",
            "seed": 123,
            "max_epochs": 1,
            "batch_size": 2,
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "loss_weights": {
                "data": 1.0,
                "increment": 0.5,
                "derivative_data": 0.1,
                "physics": 0.01,
                "smoothness": 0.0,
            },
            "run_name": "operator_smoke",
            "results_dir": results_dir,
            "val_interval": 1,
            "save_best_only": False,
            "save_interval": 1,
            "use_mixed_precision": False,
            "lr_scheduler": "none",
        },
        "logging": {"log_level": "INFO"},
    }

    setup_logging(log_level="INFO", log_to_console=True)
    set_seed(config["training"]["seed"])
    loaders = create_dataloaders(config)
    model_config = dict(config["model"])
    model_dtype = model_config.pop("dtype")
    model = cast_floating_module_dtype(
        build_model_from_config(model_config),
        torch.float32 if model_dtype == "float32" else torch.float64,
    )
    trainer = PINNTrainer(model, config, loaders["train"], loaders["val"])
    trainer.train()

    checkpoint_path = os.path.join(results_dir, "operator_smoke", "checkpoints", "epoch_0000.pth")
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"Verification failed: checkpoint was not written at {checkpoint_path}")
    if not torch.isfinite(torch.tensor(trainer.best_val_loss)):
        raise RuntimeError("Verification failed: validation loss was not finite.")

    return checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证 LandscapeNeuralOperator 训练链路。")
    parser.add_argument("--work-dir", type=str, default=None, help="可选工作目录；默认使用临时目录。")
    parser.add_argument("--keep", action="store_true", help="保留临时验证目录。")
    args = parser.parse_args()

    work_dir = args.work_dir or tempfile.mkdtemp(prefix="pinn_operator_verify_")
    try:
        checkpoint = run_verification(work_dir)
        print(f"OK: operator training verification passed. checkpoint={checkpoint}")
    finally:
        if args.work_dir is None and not args.keep:
            shutil.rmtree(work_dir, ignore_errors=True)
