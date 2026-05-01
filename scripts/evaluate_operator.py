"""
Evaluate a trained Landscape operator against held-out data and simple baselines.

This is the post-training validation entrypoint. It answers:
  - Did the learned model beat a no-change persistence baseline?
  - Did it beat one explicit physics Euler step?
  - Are state, increment, derivative, slope, and PDE residual metrics finite?
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, Iterable, Tuple

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)


def _safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def json_sanitize(value):
    """Converts NaN/Inf values to None so reports are strict JSON."""
    if isinstance(value, dict):
        return {key: json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [json_sanitize(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _as_4d_topography(tensor):
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.ndim == 3:
        return tensor.unsqueeze(1)
    return tensor


def _broadcast_dt(dt, reference):
    import torch

    if not isinstance(dt, torch.Tensor):
        dt = torch.tensor(float(dt), device=reference.device, dtype=reference.dtype)
    dt = dt.to(device=reference.device, dtype=reference.dtype)
    if dt.numel() == 1:
        return dt.view(1, 1, 1, 1).expand_as(reference)
    if dt.ndim == 1 and dt.shape[0] == reference.shape[0]:
        return dt.view(-1, 1, 1, 1).expand_as(reference)
    return dt.expand_as(reference)


def _prepare_model_params(batch_data, physics_params):
    return {
        "K": batch_data.get("k_f"),
        "D": batch_data.get("k_d"),
        "U": batch_data.get("uplift_rate"),
        "m": batch_data.get("m"),
        "n": batch_data.get("n"),
        "precip": batch_data.get("precip", physics_params.get("precip", 1.0)),
        "dx": physics_params.get("dx", 1.0),
        "dy": physics_params.get("dy", 1.0),
        "da_params": physics_params.get("drainage_area_kwargs", {}),
    }


def _prepare_physics_params(params_dict, physics_params):
    physics_params_for_loss = dict(physics_params)
    mapping = {
        "U": "U",
        "K_f": "K",
        "K_d": "D",
        "m": "m",
        "n": "n",
    }
    for physics_key, model_key in mapping.items():
        if params_dict.get(model_key) is not None:
            physics_params_for_loss[physics_key] = params_dict[model_key]
    if params_dict.get("precip") is not None:
        physics_params_for_loss["precip"] = params_dict["precip"]
    physics_params_for_loss["da_params"] = physics_params.get(
        "drainage_area_kwargs", physics_params.get("da_params", {})
    )
    return physics_params_for_loss


def _prepare_param(param_val, target_shape, device, dtype):
    import torch

    if isinstance(param_val, torch.Tensor):
        param_val = param_val.to(device=device, dtype=dtype)
        if param_val.shape == target_shape:
            return param_val
        if param_val.numel() == 1:
            return param_val.reshape(1, 1, 1, 1).expand(target_shape)
        if param_val.ndim == 1 and param_val.shape[0] == target_shape[0]:
            return param_val.view(-1, 1, 1, 1).expand(target_shape)
        return param_val.expand(target_shape)
    return torch.full(target_shape, float(param_val), device=device, dtype=dtype)


def _physics_euler_step(initial_topo, dt, physics_params_for_loss):
    import torch
    from src.physics import calculate_dhdt_physics

    target_shape = initial_topo.shape
    device, dtype = initial_topo.device, initial_topo.dtype
    dhdt = calculate_dhdt_physics(
        h=initial_topo,
        U=_prepare_param(physics_params_for_loss.get("U", 0.0), target_shape, device, dtype),
        K_f=_prepare_param(physics_params_for_loss.get("K_f", 1e-5), target_shape, device, dtype),
        m=_prepare_param(physics_params_for_loss.get("m", 0.5), target_shape, device, dtype),
        n=_prepare_param(physics_params_for_loss.get("n", 1.0), target_shape, device, dtype),
        K_d=_prepare_param(physics_params_for_loss.get("K_d", 0.01), target_shape, device, dtype),
        dx=physics_params_for_loss.get("dx", 1.0),
        dy=physics_params_for_loss.get("dy", 1.0),
        precip=physics_params_for_loss.get("precip", 1.0),
        da_params=physics_params_for_loss.get("da_params", {}),
    )
    return initial_topo + dhdt * _broadcast_dt(dt, initial_topo), dhdt


def _batch_metrics(pred, target, initial, derivative, dt, physics_loss, dx, dy):
    import torch
    import torch.nn.functional as F
    from src.physics import calculate_slope_magnitude

    target = _as_4d_topography(target.float())
    initial = _as_4d_topography(initial.float())
    pred = _as_4d_topography(pred.float())
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
    if initial.shape[-2:] != target.shape[-2:]:
        initial = F.interpolate(initial, size=target.shape[-2:], mode="bilinear", align_corners=False)

    error = pred - target
    delta_error = (pred - initial) - (target - initial)
    metrics = {
        "state_mse": torch.mean(error.square()).item(),
        "state_mae": torch.mean(torch.abs(error)).item(),
        "delta_mse": torch.mean(delta_error.square()).item(),
        "physics_loss": _safe_float(physics_loss.item()) if physics_loss is not None else float("nan"),
    }
    metrics["state_rmse"] = math.sqrt(max(metrics["state_mse"], 0.0))
    metrics["delta_rmse"] = math.sqrt(max(metrics["delta_mse"], 0.0))

    if derivative is not None:
        derivative = _as_4d_topography(derivative.float())
        finite_diff = (target - initial) / torch.clamp(_broadcast_dt(dt, target), min=1e-8)
        derivative_error = derivative - finite_diff
        derivative_mse = torch.mean(derivative_error.square()).item()
        metrics["derivative_rmse"] = math.sqrt(max(derivative_mse, 0.0))

    pred_slope = calculate_slope_magnitude(pred, dx, dy)
    target_slope = calculate_slope_magnitude(target, dx, dy)
    slope_mse = torch.mean((pred_slope - target_slope).square()).item()
    metrics["slope_rmse"] = math.sqrt(max(slope_mse, 0.0))
    return metrics


def _merge_metric_sums(metric_sums: Dict[str, float], metrics: Dict[str, float], batch_size: int):
    for key, value in metrics.items():
        if math.isfinite(float(value)):
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value) * batch_size


def _average_metric_sums(metric_sums: Dict[str, float], total_count: int) -> Dict[str, float]:
    if total_count <= 0:
        return {}
    return {key: value / total_count for key, value in sorted(metric_sums.items())}


def evaluate(config, checkpoint_path, split="val", data_dir=None) -> Tuple[Dict, str]:
    import torch

    from src.data_utils import create_dataloaders
    from src.losses import compute_pde_residual_dual_output
    from src.models import build_model_from_config, cast_floating_module_dtype
    from src.utils import get_device, set_seed

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config")
    effective_config = checkpoint_config if isinstance(checkpoint_config, dict) else config
    if data_dir is not None:
        effective_config["data"] = dict(effective_config.get("data", {}))
        effective_config["data"]["processed_dir"] = data_dir
    effective_config["training"] = dict(effective_config.get("training", {}))
    effective_config["training"]["device"] = config.get("training", {}).get("device", effective_config["training"].get("device", "auto"))
    set_seed(effective_config.get("training", {}).get("seed", effective_config.get("seed", 42)))

    dataloaders = create_dataloaders(effective_config)
    loader = dataloaders.get(split)
    if loader is None:
        raise ValueError(f"Split '{split}' is not available.")

    model_config = dict(effective_config.get("model", {}))
    model_dtype_str = model_config.pop("dtype", "float32")
    model_dtype = torch.float32 if model_dtype_str == "float32" else torch.float64
    model = cast_floating_module_dtype(build_model_from_config(model_config), model_dtype)
    model.load_state_dict(checkpoint["model_state_dict"])

    device = get_device(effective_config.get("training", {}).get("device", "auto"))
    model.to(device)
    model.eval()

    physics_params = effective_config.get("physics", {})
    dx = physics_params.get("dx", 1.0)
    dy = physics_params.get("dy", 1.0)

    learned_sums: Dict[str, float] = {}
    persistence_sums: Dict[str, float] = {}
    physics_euler_sums: Dict[str, float] = {}
    total_count = 0

    with torch.no_grad():
        for batch_data in loader:
            if batch_data is None:
                continue
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch_data.items()
            }
            initial = _as_4d_topography(batch["initial_topo"]).to(dtype=model_dtype)
            target = _as_4d_topography(batch["final_topo"]).to(dtype=model_dtype)
            dt = batch.get("dt", batch.get("run_time", 1.0))
            params_dict = _prepare_model_params(batch, physics_params)
            t_target = batch.get("run_time", dt)
            model_input = {"initial_state": initial, "params": params_dict, "t_target": t_target}

            outputs = model(model_input, mode="predict_state")
            physics_params_for_loss = _prepare_physics_params(params_dict, physics_params)
            physics_loss = compute_pde_residual_dual_output(outputs, physics_params_for_loss)

            physics_step, physics_dhdt = _physics_euler_step(initial, dt, physics_params_for_loss)
            batch_size = int(target.shape[0])
            total_count += batch_size

            learned = _batch_metrics(
                outputs["state"], target, initial, outputs.get("derivative"), dt, physics_loss, dx, dy
            )
            persistence = _batch_metrics(initial, target, initial, None, dt, None, dx, dy)
            physics_euler = _batch_metrics(physics_step, target, initial, physics_dhdt, dt, None, dx, dy)

            _merge_metric_sums(learned_sums, learned, batch_size)
            _merge_metric_sums(persistence_sums, persistence, batch_size)
            _merge_metric_sums(physics_euler_sums, physics_euler, batch_size)

    learned = _average_metric_sums(learned_sums, total_count)
    persistence = _average_metric_sums(persistence_sums, total_count)
    physics_euler = _average_metric_sums(physics_euler_sums, total_count)

    def improvement(metric, baseline):
        model_value = learned.get(metric, float("nan"))
        baseline_value = baseline.get(metric, float("nan"))
        if not math.isfinite(model_value) or not math.isfinite(baseline_value) or baseline_value <= 0:
            return float("nan")
        return 1.0 - model_value / baseline_value

    report = {
        "checkpoint": checkpoint_path,
        "split": split,
        "num_samples": total_count,
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
    return report, split


def main():
    from src.utils import load_config, normalize_training_config, setup_logging

    parser = argparse.ArgumentParser(description="Evaluate a trained Landscape operator checkpoint.")
    parser.add_argument("--config", default=os.path.join(project_root, "configs", "train_operator_config.yaml"))
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint, usually best_model.pth.")
    parser.add_argument("--split", choices=["val", "test", "train"], default="val")
    parser.add_argument("--data-dir", default=None, help="Optional processed data directory override.")
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    setup_logging(log_level="INFO", log_to_console=True)
    config = normalize_training_config(load_config(args.config))
    report, split = evaluate(config, args.checkpoint, split=args.split, data_dir=args.data_dir)

    text = json.dumps(json_sanitize(report), indent=2, sort_keys=True, allow_nan=False)
    print(text)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"Saved evaluation report to {args.output}")


if __name__ == "__main__":
    main()
