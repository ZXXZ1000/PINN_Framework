"""
Multi-step rollout evaluation for a trained Landscape operator.

Unlike one-step validation, this script autoregressively feeds the model's own
prediction back as the next state and checks whether errors grow over time.
"""

import argparse
import glob
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
if script_dir not in sys.path:
    sys.path.append(script_dir)

from evaluate_operator import (  # noqa: E402
    _as_4d_topography,
    _batch_metrics,
    _physics_euler_step,
    _prepare_model_params,
    _prepare_physics_params,
    json_sanitize,
)


def _to_float_tensor(value):
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        return value.float()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).float()
    if isinstance(value, (int, float)):
        return torch.tensor(float(value), dtype=torch.float32)
    raise TypeError(f"Unsupported tensor conversion type: {type(value)}")


def _find_trajectory(sample_data):
    from src.data_utils import TRAJECTORY_TOPO_KEYS

    for key in TRAJECTORY_TOPO_KEYS:
        if key not in sample_data:
            continue
        try:
            trajectory = _to_float_tensor(sample_data[key])
        except Exception:
            continue
        if trajectory.ndim >= 3 and trajectory.shape[0] >= 2:
            if trajectory.ndim == 3:
                trajectory = trajectory.unsqueeze(1)
            if trajectory.ndim == 4 and trajectory.shape[1] == 1:
                return trajectory
    return None


def _find_time_vector(sample_data):
    from src.data_utils import TRAJECTORY_TIME_KEYS

    for key in TRAJECTORY_TIME_KEYS:
        if key in sample_data:
            try:
                return _to_float_tensor(sample_data[key]).flatten()
            except Exception:
                return None
    return None


def _dt_for_step(sample_data, step_idx: int, num_steps: int):
    import torch

    time_vector = _find_time_vector(sample_data)
    if time_vector is not None and time_vector.numel() >= step_idx + 2:
        return (time_vector[step_idx + 1] - time_vector[step_idx]).float()
    if "dt" in sample_data:
        return _to_float_tensor(sample_data["dt"])
    if "run_time" in sample_data:
        return _to_float_tensor(sample_data["run_time"]) / max(num_steps - 1, 1)
    return torch.tensor(1.0, dtype=torch.float32)


def _split_files(config: Dict, data_dir: str) -> Dict[str, List[str]]:
    data_config = config.get("data", {})
    train_split = data_config.get("train_split", 0.8)
    val_split = data_config.get("val_split", 0.1)
    if train_split + val_split > 1.0:
        raise ValueError(f"train_split ({train_split}) + val_split ({val_split}) cannot exceed 1.0")

    files = glob.glob(os.path.join(data_dir, "**", "*.pt"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No .pt files found in {data_dir}")

    seed = config.get("training", {}).get("seed", config.get("seed", 42))
    rng = random.Random(seed)
    rng.shuffle(files)

    num_total = len(files)
    num_train = int(num_total * train_split)
    num_val = int(num_total * val_split)
    if val_split > 0 and num_val == 0 and num_total > num_train:
        num_val = 1
    num_train = min(num_train, num_total - num_val)
    return {
        "train": files[:num_train],
        "val": files[num_train:num_train + num_val],
        "test": files[num_train + num_val:],
    }


def _load_norm_stats(config: Dict, train_files: List[str]) -> Optional[Dict]:
    import json
    from src.data_utils import compute_normalization_stats

    norm_config = config.get("data", {}).get("normalization", {})
    if not norm_config.get("enabled", False):
        return None

    stats_file = norm_config.get("stats_file")
    if stats_file and os.path.exists(stats_file):
        with open(stats_file, "r", encoding="utf-8") as f:
            return json.load(f)

    if norm_config.get("compute_stats", False):
        return compute_normalization_stats(train_files, ["topo", "uplift_rate", "k_f", "k_d"])
    return None


def _normalize_field(tensor, norm_stats: Optional[Dict], field_key: str):
    import torch

    if not norm_stats or field_key not in norm_stats:
        return tensor
    stats = norm_stats[field_key]
    min_val = torch.tensor(stats["min"], device=tensor.device, dtype=tensor.dtype)
    max_val = torch.tensor(stats["max"], device=tensor.device, dtype=tensor.dtype)
    if min_val >= max_val:
        return tensor
    return (tensor - min_val) / (max_val - min_val + 1e-8)


def _prepare_sample_params(sample_data, norm_stats, device, dtype):
    required = ["uplift_rate", "k_f", "k_d", "m", "n"]
    missing = [key for key in required if key not in sample_data]
    if missing:
        raise KeyError(f"Missing required rollout parameter fields: {missing}")

    params = {}
    for key in required:
        params[key] = _to_float_tensor(sample_data[key]).to(device=device, dtype=dtype)
    params["uplift_rate"] = _normalize_field(params["uplift_rate"], norm_stats, "uplift_rate")
    params["k_f"] = _normalize_field(params["k_f"], norm_stats, "k_f")
    params["k_d"] = _normalize_field(params["k_d"], norm_stats, "k_d")
    return params


def _load_model_and_config(config, checkpoint_path, data_dir=None):
    import torch
    from src.models import build_model_from_config, cast_floating_module_dtype
    from src.utils import get_device, set_seed

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config")
    effective_config = checkpoint_config if isinstance(checkpoint_config, dict) else config
    if data_dir is not None:
        effective_config["data"] = dict(effective_config.get("data", {}))
        effective_config["data"]["processed_dir"] = data_dir
    effective_config["training"] = dict(effective_config.get("training", {}))
    effective_config["training"]["device"] = config.get("training", {}).get(
        "device", effective_config["training"].get("device", "auto")
    )
    set_seed(effective_config.get("training", {}).get("seed", effective_config.get("seed", 42)))

    model_config = dict(effective_config.get("model", {}))
    model_dtype_str = model_config.pop("dtype", "float32")
    model_dtype = torch.float32 if model_dtype_str == "float32" else torch.float64
    model = cast_floating_module_dtype(build_model_from_config(model_config), model_dtype)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = get_device(effective_config.get("training", {}).get("device", "auto"))
    model.to(device)
    model.eval()
    return model, effective_config, model_dtype, device


def _merge_sums(target: Dict[str, float], metrics: Dict[str, float]):
    for key, value in metrics.items():
        value = float(value)
        if math.isfinite(value):
            target[key] = target.get(key, 0.0) + value


def _average_sums(metric_sums: Dict[str, float], count: int):
    if count <= 0:
        return {}
    return {key: value / count for key, value in sorted(metric_sums.items())}


def evaluate_rollout(config, checkpoint_path, split="val", rollout_steps=5, data_dir=None, max_files=None):
    import torch

    model, effective_config, model_dtype, device = _load_model_and_config(config, checkpoint_path, data_dir=data_dir)
    data_dir = effective_config.get("data", {}).get("processed_dir", "data/processed")
    split_files = _split_files(effective_config, data_dir)
    files = split_files.get(split, [])
    if max_files is not None:
        files = files[: int(max_files)]

    norm_stats = _load_norm_stats(effective_config, split_files.get("train", []))
    physics_params = effective_config.get("physics", {})
    dx = physics_params.get("dx", 1.0)
    dy = physics_params.get("dy", 1.0)

    per_step = {}
    per_step_counts = {}
    evaluated_files = 0
    skipped_files = 0

    with torch.no_grad():
        for filepath in files:
            sample_data = torch.load(filepath, map_location="cpu", weights_only=False)
            trajectory = _find_trajectory(sample_data)
            if trajectory is None:
                skipped_files += 1
                continue
            max_available_steps = int(trajectory.shape[0]) - 1
            steps = min(int(rollout_steps), max_available_steps)
            if steps <= 0:
                skipped_files += 1
                continue

            trajectory = trajectory.to(device=device, dtype=model_dtype)
            trajectory = _normalize_field(trajectory, norm_stats, "topo")
            params = _prepare_sample_params(sample_data, norm_stats, device, model_dtype)
            params_dict = _prepare_model_params(params, physics_params)
            physics_params_for_loss = _prepare_physics_params(params_dict, physics_params)

            learned_state = _as_4d_topography(trajectory[0])
            persistence_state = learned_state.clone()
            physics_state = learned_state.clone()
            initial_state = learned_state.clone()
            evaluated_files += 1

            for step_idx in range(steps):
                target = _as_4d_topography(trajectory[step_idx + 1])
                dt = _dt_for_step(sample_data, step_idx, int(trajectory.shape[0])).to(device=device, dtype=model_dtype)

                model_input = {
                    "initial_state": learned_state,
                    "params": params_dict,
                    "t_target": dt,
                }
                outputs = model(model_input, mode="predict_state")
                learned_state = outputs["state"]
                physics_state, physics_dhdt = _physics_euler_step(physics_state, dt, physics_params_for_loss)

                learned_metrics = _batch_metrics(
                    learned_state, target, initial_state, None, dt, None, dx, dy
                )
                persistence_metrics = _batch_metrics(
                    persistence_state, target, initial_state, None, dt, None, dx, dy
                )
                physics_metrics = _batch_metrics(
                    physics_state, target, initial_state, None, dt, None, dx, dy
                )

                step_key = str(step_idx + 1)
                bucket = per_step.setdefault(
                    step_key,
                    {"learned": {}, "persistence": {}, "physics_euler": {}},
                )
                _merge_sums(bucket["learned"], learned_metrics)
                _merge_sums(bucket["persistence"], persistence_metrics)
                _merge_sums(bucket["physics_euler"], physics_metrics)
                per_step_counts[step_key] = per_step_counts.get(step_key, 0) + 1

    averaged_steps = {}
    for step_key, buckets in sorted(per_step.items(), key=lambda item: int(item[0])):
        count = per_step_counts.get(step_key, 0)
        learned = _average_sums(buckets["learned"], count)
        persistence = _average_sums(buckets["persistence"], count)
        physics_euler = _average_sums(buckets["physics_euler"], count)

        def improvement(metric, baseline):
            model_value = learned.get(metric, float("nan"))
            baseline_value = baseline.get(metric, float("nan"))
            if not math.isfinite(model_value) or not math.isfinite(baseline_value) or baseline_value <= 0:
                return float("nan")
            return 1.0 - model_value / baseline_value

        averaged_steps[step_key] = {
            "num_rollouts": count,
            "learned": learned,
            "baselines": {
                "persistence": persistence,
                "physics_euler": physics_euler,
            },
            "improvement": {
                "state_rmse_vs_persistence": improvement("state_rmse", persistence),
                "state_rmse_vs_physics_euler": improvement("state_rmse", physics_euler),
                "delta_rmse_vs_persistence": improvement("delta_rmse", persistence),
                "delta_rmse_vs_physics_euler": improvement("delta_rmse", physics_euler),
            },
        }

    return {
        "checkpoint": checkpoint_path,
        "split": split,
        "requested_rollout_steps": int(rollout_steps),
        "evaluated_files": evaluated_files,
        "skipped_files": skipped_files,
        "per_step": averaged_steps,
    }


def main():
    from src.utils import load_config, normalize_training_config, setup_logging

    parser = argparse.ArgumentParser(description="Evaluate multi-step autoregressive rollout.")
    parser.add_argument("--config", default=os.path.join(project_root, "configs", "train_operator_config.yaml"))
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint, usually best_model.pth.")
    parser.add_argument("--split", choices=["val", "test", "train"], default="val")
    parser.add_argument("--steps", type=int, default=5, help="Maximum number of autoregressive steps.")
    parser.add_argument("--data-dir", default=None, help="Optional processed data directory override.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap for quick checks.")
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    setup_logging(log_level="INFO", log_to_console=True)
    config = normalize_training_config(load_config(args.config))
    report = evaluate_rollout(
        config,
        args.checkpoint,
        split=args.split,
        rollout_steps=args.steps,
        data_dir=args.data_dir,
        max_files=args.max_files,
    )

    text = json.dumps(json_sanitize(report), indent=2, sort_keys=True, allow_nan=False)
    print(text)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"Saved rollout report to {args.output}")


if __name__ == "__main__":
    main()
