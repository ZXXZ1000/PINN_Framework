# PINN_Framework/scripts/optimize.py
"""
使用训练好的 PINN 模型和观测数据优化物理参数（例如，抬升率场）。
"""

import argparse
import logging
import os
import sys
import numpy as np
import torch
import time
from typing import Dict, Any, Optional, Tuple, Union, List

# 将项目根目录添加到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # scripts 目录的上级是项目根目录
if project_root not in sys.path:
    sys.path.append(project_root)

# 从新框架导入必要的模块
try:
    # 假设 utils, models, optimizer_utils 都在 src 目录下
    from src.utils import load_config, setup_logging, get_device
    from src.models import AdaptiveFastscapePINN, TimeDerivativePINN # 导入主线模型和基类
    from src.optimizer_utils import optimize_parameters, interpolate_params_torch # 导入优化函数和可选的插值函数
except ImportError as e:
    print(f"错误：无法导入必要的模块: {e}。请确保你在 PINN_Framework 目录下运行，并且 src 目录包含所需文件。")
    sys.exit(1)

# --- 辅助函数：加载数据 ---

def load_target_dem(filepath: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """加载目标 DEM 并返回 torch 张量 [B=1, C=1, H, W]。"""
    logging.info(f"正在从 {filepath} 加载目标 DEM...")
    try:
        # 假设为 .npy 格式，如果需要其他格式（如 GeoTIFF），则需要调整
        target_dem_np = np.load(filepath)
        target_dem_torch = torch.from_numpy(target_dem_np).to(dtype).unsqueeze(0).unsqueeze(0).to(device)
        logging.info(f"目标 DEM 加载完成，形状: {target_dem_torch.shape}")
        return target_dem_torch
    except FileNotFoundError:
        logging.error(f"目标 DEM 文件未找到: {filepath}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"加载目标 DEM 时出错 {filepath}: {e}", exc_info=True)
        sys.exit(1)

def load_fixed_inputs(config: Dict, target_shape: Tuple[int, int], device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    """加载配置中指定的固定输入（例如，initial_topo, K, D）。"""
    # 使用 OmegaConf 安全访问嵌套字典
    try:
        from omegaconf import OmegaConf
        if not isinstance(config, OmegaConf): config = OmegaConf.create(config)
    except ImportError: pass # 继续使用字典访问

    fixed_input_paths = config.get('fixed_inputs', {})
    inputs = {}
    logging.info("正在加载固定输入...")
    target_h, target_w = target_shape

    # 加载初始地形
    init_topo_path = fixed_input_paths.get('initial_topography')
    if init_topo_path and os.path.exists(init_topo_path):
        try:
            init_topo_np = np.load(init_topo_path) # 假设 .npy
            if init_topo_np.shape != target_shape:
                 # 可以尝试插值或报错
                 raise ValueError(f"初始地形形状 {init_topo_np.shape} 必须与目标 DEM 形状 {target_shape} 匹配。")
            inputs['initial_topography'] = torch.from_numpy(init_topo_np).to(dtype).unsqueeze(0).unsqueeze(0).to(device)
            logging.info(f"已从 {init_topo_path} 加载初始地形")
        except Exception as e:
            logging.error(f"加载初始地形时出错 {init_topo_path}: {e}。使用零初始化。", exc_info=True)
            inputs['initial_topography'] = torch.zeros(1, 1, target_h, target_w, device=device, dtype=dtype)
    else:
        logging.warning("未指定或未找到初始地形路径。使用零初始化。")
        inputs['initial_topography'] = torch.zeros(1, 1, target_h, target_w, device=device, dtype=dtype)

    # 加载其他固定参数 (K, D 等) - 这些将由 ParameterOptimizer 处理为张量
    fixed_params_config = config.get('fixed_parameters', {})
    for key, value in fixed_params_config.items():
         if isinstance(value, str) and os.path.exists(value): # 检查值是否为现有文件路径
              try:
                   param_np = np.load(value)
                   # 转换为张量，ParameterOptimizer 会处理形状
                   inputs[key] = torch.from_numpy(param_np).to(dtype)
                   logging.info(f"已从 {value} 加载固定参数 '{key}'")
              except Exception as e:
                   logging.error(f"加载固定参数 '{key}' 时出错 {value}: {e}。跳过。", exc_info=True)
         else:
              # 假定为标量值，ParameterOptimizer 会处理
              try:
                   inputs[key] = float(value)
                   logging.info(f"使用标量值 {inputs[key]} 作为固定参数 '{key}'。")
              except (ValueError, TypeError):
                   logging.error(f"固定参数 '{key}' 的值无效: {value}。跳过。")

    return inputs


def prepare_initial_param_guess(config: Dict, target_shape: Tuple[int, int], device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """准备待优化参数的初始猜测值。"""
    # 使用 OmegaConf 安全访问嵌套字典
    try:
        from omegaconf import OmegaConf
        if not isinstance(config, OmegaConf): config = OmegaConf.create(config)
    except ImportError: pass # 继续使用字典访问

    params_to_optimize_config = config.get('parameters_to_optimize', {})
    initial_params = {}
    target_h, target_w = target_shape

    if not params_to_optimize_config:
         logging.warning("配置中未找到 'parameters_to_optimize' 部分。")
         return {}

    for name, p_config in params_to_optimize_config.items():
        # 确保 p_config 是字典
        if not isinstance(p_config, dict):
             logging.error(f"参数 '{name}' 的配置无效，应为字典。跳过。")
             continue

        initial_guess_type = p_config.get('initial_guess_type', 'constant')
        initial_guess_value = p_config.get('initial_value', 0.0)
        param_shape_config = p_config.get('parameter_shape') # 低分辨率形状

        if param_shape_config: # 低分辨率参数化
             param_shape = tuple(param_shape_config)
             logging.info(f"初始化低分辨率参数 '{name}'，形状 {param_shape}。")
             if initial_guess_type == 'constant':
                  low_res_guess = torch.full(param_shape, float(initial_guess_value), device=device, dtype=dtype)
             elif initial_guess_type == 'random':
                  low = p_config.get('initial_random_low', 0.0)
                  high = p_config.get('initial_random_high', 1.0)
                  low_res_guess = torch.rand(param_shape, device=device, dtype=dtype) * (high - low) + low
             else:
                  raise ValueError(f"不支持的 initial_guess_type '{initial_guess_type}' (低分辨率)")

             # 插值到目标形状
             try:
                  # 使用 optimizer_utils 中的插值函数
                  initial_params[name] = interpolate_params_torch(
                       low_res_guess, param_shape, target_shape, method='bilinear' # 或从配置读取方法
                  ).unsqueeze(0).unsqueeze(0) # 添加 B, C 维度
                  logging.info(f"已将 '{name}' 的初始猜测插值到形状 {initial_params[name].shape}")
             except Exception as e:
                  logging.error(f"插值低分辨率初始猜测失败 '{name}': {e}。回退到常量值。", exc_info=True)
                  initial_params[name] = torch.full((1, 1, *target_shape), float(initial_guess_value), device=device, dtype=dtype)

        else: # 全分辨率参数化
             logging.info(f"初始化全分辨率参数 '{name}'。")
             target_full_shape = (1, 1, *target_shape) # B=1, C=1
             if initial_guess_type == 'constant':
                  initial_params[name] = torch.full(target_full_shape, float(initial_guess_value), device=device, dtype=dtype)
             elif initial_guess_type == 'random':
                  low = p_config.get('initial_random_low', 0.0)
                  high = p_config.get('initial_random_high', 1.0)
                  initial_params[name] = torch.rand(target_full_shape, device=device, dtype=dtype) * (high - low) + low
             else:
                  raise ValueError(f"不支持的 initial_guess_type '{initial_guess_type}' (全分辨率)")

        # 确保需要梯度
        if name in initial_params: # 检查是否成功创建
             initial_params[name].requires_grad_(True)

    return initial_params


def main(args):
    """主函数，运行参数优化。"""
    # --- 设置 ---
    try:
        config = load_config(args.config) # 返回 OmegaConf 对象
    except Exception as e:
        print(f"错误：无法加载配置文件 {args.config}: {e}")
        sys.exit(1)

    # 使用 OmegaConf 访问配置
    try:
        from omegaconf import OmegaConf
        if not isinstance(config, OmegaConf): config = OmegaConf.create(config)
    except ImportError: pass # 继续使用字典

    opt_config = config.get('optimization', {})
    model_config = config.get('model', {})
    run_config = config.get('run_options', {})

    output_dir = run_config.get('output_dir', 'results/')
    run_name = opt_config.get('run_name', f'optimize_run_{int(time.time())}')
    log_dir = os.path.join(output_dir, run_name, 'logs')
    opt_output_dir = os.path.join(output_dir, run_name, 'optimize_output')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(opt_output_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'optimize.log')
    log_level = config.get('logging', {}).get('log_level', 'INFO')
    setup_logging(log_level=log_level, log_file=log_file_path, log_to_console=True)

    device = get_device(run_config.get('device', 'auto'))
    model_dtype_str = model_config.get('dtype', 'float32')
    model_dtype = torch.float32 if model_dtype_str == 'float32' else torch.float64
    logging.info(f"使用设备: {device}, 模型数据类型: {model_dtype}")

    # --- 加载训练好的 PINN 模型 (固定为 AdaptiveFastscapePINN) ---
    model_args = {k: v for k, v in model_config.items() if k not in ['name', 'dtype']}
    # 确保传递 domain_x, domain_y
    physics_params = config.get('physics', {})
    data_params = config.get('data', {})
    domain_x = physics_params.get('domain_x', data_params.get('domain_x'))
    domain_y = physics_params.get('domain_y', data_params.get('domain_y'))
    if domain_x: model_args['domain_x'] = list(domain_x) if isinstance(domain_x, (list, tuple)) else [0.0, float(domain_x)]
    if domain_y: model_args['domain_y'] = list(domain_y) if isinstance(domain_y, (list, tuple)) else [0.0, float(domain_y)]

    try:
        # 确保传递给模型的配置是标准字典
        model_args_dict = OmegaConf.to_container(model_args, resolve=True) if 'OmegaConf' in locals() and isinstance(model_args, OmegaConf) else model_args
        model = AdaptiveFastscapePINN(**model_args_dict).to(dtype=model_dtype)
    except Exception as e:
        logging.error(f"初始化 AdaptiveFastscapePINN 模型失败: {e}", exc_info=True)
        sys.exit(1)

    checkpoint_path = opt_config.get('pinn_checkpoint_path')
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logging.error(f"未找到或未指定 PINN 检查点路径: {checkpoint_path}")
        sys.exit(1)
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        logging.info(f"已从 {checkpoint_path} 加载训练好的 PINN 模型")
        num_params = sum(p.numel() for p in model.parameters())
        logging.info(f"模型参数数量: {num_params:,}")
    except Exception as e:
        logging.error(f"加载模型检查点时出错: {e}", exc_info=True)
        sys.exit(1)

    # --- 加载目标 DEM ---
    target_dem_path = opt_config.get('target_dem_path')
    if not target_dem_path:
         logging.error("未在优化配置中指定目标 DEM 路径 ('target_dem_path')。")
         sys.exit(1)
    target_dem_tensor = load_target_dem(target_dem_path, device, model_dtype)
    target_dem_shape = target_dem_tensor.shape[2:] # H, W

    # --- 加载固定模型输入 ---
    fixed_inputs_dict = load_fixed_inputs(opt_config, target_dem_shape, device, model_dtype)
    initial_state_tensor = fixed_inputs_dict.pop('initial_topography') # 提取初始状态
    fixed_params_dict = fixed_inputs_dict # 剩余的是固定参数

    # --- 准备待优化参数的初始猜测 ---
    params_to_optimize_config = opt_config.get('parameters_to_optimize')
    if not params_to_optimize_config:
         logging.error("未在优化配置中找到 'parameters_to_optimize' 部分。")
         sys.exit(1)
    # 将 OmegaConf 列表/字典转换为标准 Python 类型
    params_to_optimize_config_dict = OmegaConf.to_container(params_to_optimize_config, resolve=True) if 'OmegaConf' in locals() and isinstance(params_to_optimize_config, OmegaConf) else params_to_optimize_config

    initial_params_guess = prepare_initial_param_guess(opt_config, target_dem_shape, device, model_dtype)


    # --- 获取目标时间 ---
    t_target_value = opt_config.get('t_target')
    if t_target_value is None:
         t_target_value = config.get('physics_params', {}).get('total_time') # 尝试从物理参数获取
         if t_target_value is None:
              logging.error("未在优化配置或物理参数中指定目标时间 't_target'/'total_time'。")
              sys.exit(1)
    logging.info(f"使用目标时间: {t_target_value}")

    # --- 运行优化 ---
    # 确保优化配置包含保存路径
    opt_params_config = config.get('optimization_params', {}) # 获取优化参数子配置
    # 将 OmegaConf 转为字典以进行修改
    if 'OmegaConf' in locals() and isinstance(opt_params_config, OmegaConf):
         opt_params_config = OmegaConf.to_container(opt_params_config, resolve=True)

    opt_params_config['save_path'] = os.path.join(opt_output_dir, opt_params_config.get('output_filename', 'optimized_params.pth'))
    # 更新主配置中的优化参数（如果需要 optimize_parameters 访问更新后的路径）
    config['optimization_params'] = opt_params_config


    optimized_params, history = optimize_parameters(
        model=model,
        observation_data=target_dem_tensor,
        params_to_optimize_config=params_to_optimize_config_dict, # 传递标准字典
        config=config, # 传递完整配置（包含更新后的 optimization_params）
        initial_state=initial_state_tensor,
        fixed_params=fixed_params_dict,
        t_target=t_target_value
    )

    # --- 后处理和保存 ---
    logging.info("优化过程完成。")
    if history['loss']:
         logging.info(f"最终损失: {history['final_loss']:.6e}")
    else:
         logging.warning("优化历史记录为空。")

    # 结果已由 optimize_parameters 保存
    # 可以添加额外的分析或可视化
    for param_name, optimized_tensor in optimized_params.items():
        logging.info(f"优化后的参数 '{param_name}' 均值: {optimized_tensor.mean().item():.6e}")
        # 可以选择保存单个参数场为 .npy 文件
        try:
            np.save(os.path.join(opt_output_dir, f'optimized_{param_name}.npy'), optimized_tensor.cpu().numpy())
            logging.info(f"已将优化后的参数 '{param_name}' 保存为 npy 文件。")
        except Exception as e:
            logging.error(f"保存优化后的参数 '{param_name}' 时出错: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用训练好的 PINN 优化物理参数。")
    parser.add_argument('--config', type=str, required=True, help='优化配置文件的路径。')
    args = parser.parse_args()
    main(args)