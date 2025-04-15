# PINN_Framework/scripts/generate_data.py
"""
使用 xsimlab 和 fastscape 生成模拟数据。
支持多分辨率和空间变化的参数。
"""

import argparse
import logging
import os
import sys
import numpy as np
import torch
import xsimlab as xs
import fastscape # 导入 fastscape 包
import math
from typing import Dict, Any, Tuple, List, Union

# 将项目根目录添加到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # scripts 目录的上级是项目根目录
if project_root not in sys.path:
    sys.path.append(project_root)

# 从新框架导入工具函数
try:
    # 假设 utils 在 src 目录下
    from src.utils import load_config, setup_logging, save_data_sample
except ImportError:
    print("错误：无法从 src.utils 导入必要的函数。请确保你在 PINN_Framework 目录下运行此脚本，或者项目结构正确。")
    sys.exit(1)

# 导入 fastscape 模型 (例如 basic_model)
try:
    from fastscape.models import basic_model
except ImportError:
    print("错误：无法导入 fastscape.models.basic_model。请确保 fastscape 已正确安装在环境中。")
    sys.exit(1)


# --- 辅助函数：生成空间参数场 ---
def generate_spatial_field(shape: Tuple[int, int], min_val: float, max_val: float, pattern: str = 'random') -> np.ndarray:
    """根据指定模式生成空间参数场。"""
    logging.debug(f"生成空间场: shape={shape}, min={min_val}, max={max_val}, pattern='{pattern}'")
    if pattern == 'random':
        field = np.random.uniform(min_val, max_val, shape)
    elif pattern == 'fault':
        # (断层生成逻辑保持不变)
        field = np.ones(shape) * min_val
        fault_pos_rel = np.random.uniform(0.3, 0.7)
        fault_angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        center_y, center_x = shape[0] / 2, shape[1] / 2
        y_indices, x_indices = np.indices(shape)
        y_rel, x_rel = y_indices - center_y, x_indices - center_x
        rotated_y = y_rel * np.cos(fault_angle) - x_rel * np.sin(fault_angle)
        split_value = (shape[0] * fault_pos_rel) - center_y
        field[rotated_y > split_value] = max_val
        logging.debug(f"生成断层模式: angle={fault_angle:.2f}, pos={fault_pos_rel:.2f}")
    elif pattern == 'constant':
         field = np.full(shape, min_val)
         logging.debug(f"生成常量场，值为: {min_val}")
    else:
        logging.warning(f"未知的空间模式: '{pattern}'。使用随机均匀分布。")
        field = np.random.uniform(min_val, max_val, shape)
    return field.astype(np.float32)

# --- 辅助函数：采样标量参数 ---
def sample_scalar_parameters(param_ranges: Dict[str, Union[List[float], float]]) -> Dict[str, float]:
    """根据范围采样标量参数。"""
    sampled_params = {}
    # 定义默认范围（如果配置中未提供）
    default_scalar_ranges = {
        'uplift__rate': [1e-4, 1e-3],
        'spl__k_coef': [1e-6, 1e-5],
        'diffusion__diffusivity': [0.1, 1.0],
        'spl__area_exp': [0.4, 0.6],
        'spl__slope_exp': [1.0, 1.0]
    }
    for key, default_range in default_scalar_ranges.items():
        current_range = param_ranges.get(key, default_range)
        if isinstance(current_range, list) and len(current_range) == 2:
             # Handle case where min and max are the same (fixed value)
             if abs(current_range[0] - current_range[1]) < 1e-9:
                  sampled_params[key] = float(current_range[0])
             else:
                  sampled_params[key] = np.random.uniform(current_range[0], current_range[1])
        elif isinstance(current_range, (int, float)):
             sampled_params[key] = float(current_range)
        else:
             logging.warning(f"标量参数 '{key}' 的范围格式无效: {current_range}。使用默认范围 {default_range}。")
             sampled_params[key] = np.random.uniform(default_range[0], default_range[1])
    logging.debug(f"采样的标量参数 (初始): {sampled_params}")
    return sampled_params

# --- 辅助函数：生成单个模拟样本 ---
def generate_single_sample(sample_index: int, config: Dict) -> bool:
    """根据提供的配置生成并保存单个模拟样本。"""
    # 使用 OmegaConf 访问配置，提供默认值
    try:
        from omegaconf import OmegaConf
        if not isinstance(config, OmegaConf): # Ensure it's OmegaConf object for easy access
             config = OmegaConf.create(config)
    except ImportError:
        logging.warning("OmegaConf 未安装，使用标准字典访问，可能缺少默认值处理。")

    data_gen_config = config.get('data_generation', {})
    sim_config = data_gen_config.get('simulation_params', {})
    parameter_type = data_gen_config.get('parameter_type', 'scalar').lower()
    scalar_param_ranges = data_gen_config.get('parameter_ranges', {})
    spatial_param_config = data_gen_config.get('spatial_parameter_config', {})
    output_dir = data_gen_config.get('output_dir') # 从特定分辨率的配置中获取

    if not output_dir:
        logging.error(f"样本 {sample_index+1}: 配置中缺少 'output_dir'。无法保存样本。")
        return False

    # 获取此样本/分辨率的模拟参数
    grid_shape = tuple(sim_config.get('grid_shape'))
    grid_length = sim_config.get('grid_length') # [length_y, length_x]
    time_step = sim_config.get('time_step')
    run_time_total = sim_config.get('run_time')

    if not grid_shape or len(grid_shape) != 2 or not grid_length or len(grid_length) != 2 or not time_step or not run_time_total:
        logging.error(f"样本 {sample_index+1}: 模拟参数不完整或格式错误 (shape={grid_shape}, length={grid_length}, time_step={time_step}, run_time={run_time_total})。跳过。")
        return False

    logging.info(f"--- 正在生成样本 {sample_index+1}，形状 {grid_shape} ---")
    try:
        # 1. 生成/采样参数
        final_params = sample_scalar_parameters(scalar_param_ranges)
        if parameter_type == 'spatial':
            # 定义哪些参数可以是空间的
            spatial_capable_params = ['uplift__rate', 'spl__k_coef'] # 仅 U 和 K_f
            spatial_params = {}
            for key in spatial_capable_params:
                 if key in spatial_param_config:
                      config_for_key = spatial_param_config[key]
                      pattern = config_for_key.get('pattern', 'constant')
                      min_val = float(config_for_key.get('min', 0.0))
                      max_val = float(config_for_key.get('max', 1.0))
                      spatial_params[key] = generate_spatial_field(grid_shape, min_val, max_val, pattern)
                 else:
                      logging.debug(f"未找到空间参数 '{key}' 的配置。将保持标量。")
            final_params.update(spatial_params) # 如果生成了空间场，则覆盖标量值

        logging.debug(f"样本 {sample_index+1} 的最终参数: { {k: type(v).__name__ for k, v in final_params.items()} }")

        # 2. 设置 xsimlab 模拟
        sim_times = np.arange(0, run_time_total + time_step, time_step)
        output_times = [0, run_time_total] # 仅输出初始和最终状态
        model_to_use = basic_model # 使用 fastscape 的基础模型

        input_vars_dict = {
            'grid__shape': list(grid_shape),
            'grid__length': list(grid_length), # Ensure list
            'boundary__status': sim_config.get('boundary_status', 'fixed_value'),
        }
        # 添加生成的参数，为空间数组指定维度
        for key, value in final_params.items():
            if isinstance(value, np.ndarray) and value.ndim == 2:
                # 检查参数是否必须是标量
                if key in ['spl__area_exp', 'spl__slope_exp']:
                     logging.error(f"参数 '{key}' 生成为空间场，但模型要求为标量。使用平均值。")
                     input_vars_dict[key] = float(value.mean())
                else:
                     input_vars_dict[key] = (('y', 'x'), value) # 指定维度
            else: # 标量
                input_vars_dict[key] = float(value) # 确保是浮点数

        output_vars_dict = {'topography__elevation': 'out'} # 仅输出高程

        in_ds = xs.create_setup(
            model=model_to_use,
            clocks={'time': sim_times, 'out': output_times},
            master_clock='time',
            input_vars=input_vars_dict,
            output_vars=output_vars_dict
        )

        # 3. 运行模拟
        logging.info(f"样本 {sample_index+1}: 运行 xsimlab 模拟...")
        out_ds = in_ds.xsimlab.run(model=model_to_use)
        logging.info(f"样本 {sample_index+1}: 模拟完成。")

        # 4. 提取和格式化数据
        if 'topography__elevation' not in out_ds or len(out_ds['out']) < 2:
             logging.error(f"样本 {sample_index+1}: 输出数据集中缺少 'topography__elevation' 或输出时间点不足。")
             return False

        # 确保按预期时间点提取
        try:
            initial_topo_xr = out_ds['topography__elevation'].sel(out=0)
            final_topo_xr = out_ds['topography__elevation'].sel(out=run_time_total)
        except KeyError:
             logging.warning(f"样本 {sample_index+1}: 无法按精确时间点提取高程。使用最近邻方法。")
             initial_topo_xr = out_ds['topography__elevation'].sel(out=0, method='nearest')
             final_topo_xr = out_ds['topography__elevation'].sel(out=run_time_total, method='nearest')

        # 转换为张量，添加通道维度 [1, H, W]
        initial_topo_tensor = torch.from_numpy(initial_topo_xr.values.astype(np.float32)).unsqueeze(0)
        final_topo_tensor = torch.from_numpy(final_topo_xr.values.astype(np.float32)).unsqueeze(0)

        # 准备保存的字典
        sample_output = {
            'initial_topo': initial_topo_tensor,
            'final_topo': final_topo_tensor,
            'run_time': torch.tensor(run_time_total, dtype=torch.float32)
        }
        # 添加参数（映射键名，保存为 numpy 或 tensor）
        param_mapping = {
            'uplift__rate': 'uplift_rate', 'spl__k_coef': 'k_f',
            'diffusion__diffusivity': 'k_d', 'spl__area_exp': 'm', 'spl__slope_exp': 'n'
        }
        for xsim_key, data_key in param_mapping.items():
            if xsim_key in final_params:
                value = final_params[xsim_key]
                # 保存空间场为 numpy 数组，标量为 0 维张量
                sample_output[data_key] = value if isinstance(value, np.ndarray) else torch.tensor(value, dtype=torch.float32)

        # 5. 保存样本
        filename = f"sample_{sample_index:05d}.pt"
        filepath = os.path.join(output_dir, filename)
        save_data_sample(sample_output, filepath) # 使用 utils 中的保存函数
        logging.info(f"样本 {sample_index+1}: 已成功保存到 {filepath}")
        return True # 指示成功

    except Exception as e:
        logging.error(f"生成或保存样本 {sample_index+1} 失败: {e}", exc_info=True)
        return False # 指示失败

# --- 辅助函数：生成特定分辨率的数据集 ---
def generate_dataset_for_resolution(config: Dict):
    """为特定配置（分辨率）生成数据集。"""
    # 使用 OmegaConf 访问配置，提供默认值
    try:
        from omegaconf import OmegaConf
        if not isinstance(config, OmegaConf): config = OmegaConf.create(config)
    except ImportError: pass # 继续使用字典访问

    data_gen_config = config.get('data_generation', {})
    num_samples = data_gen_config.get('num_samples', 10)
    output_dir = data_gen_config.get('output_dir')
    sim_params_conf = data_gen_config.get('simulation_params', {})
    grid_shape = tuple(sim_params_conf.get('grid_shape', [0,0]))

    if not output_dir or grid_shape == (0,0):
        logging.error("generate_dataset_for_resolution: 配置中缺少 output_dir 或 grid_shape。")
        return

    logging.info(f"开始生成数据集: {num_samples} 个样本，形状 {grid_shape}，保存到 {output_dir}")
    os.makedirs(output_dir, exist_ok=True) # 确保目录存在

    generated_count = 0
    for i in range(num_samples):
        if generate_single_sample(i, config): # 传递当前分辨率的配置
            generated_count += 1

    logging.info(f"数据集生成完成，形状 {grid_shape}。成功生成 {generated_count}/{num_samples} 个样本。")


# --- 主函数 ---
def main(args):
    """主函数，处理配置加载、日志设置和多尺度数据生成。"""
    # --- 设置 ---
    try:
        config = load_config(args.config) # 使用 utils 中的加载函数
    except Exception as e:
        print(f"错误：无法加载配置文件 {args.config}: {e}")
        sys.exit(1)

    # 使用 OmegaConf 访问配置，提供默认值
    try:
        from omegaconf import OmegaConf
        if not isinstance(config, OmegaConf): config = OmegaConf.create(config)
    except ImportError: pass # 继续使用字典访问

    log_config = config.get('logging', {})
    log_dir = log_config.get('log_dir', 'logs/data_generation')
    log_filename = log_config.get('log_filename', 'generate_data.log')
    log_file_path = os.path.join(log_dir, log_filename) if log_dir and log_filename else None
    log_level = log_config.get('log_level', 'INFO')
    setup_logging(log_level=log_level, log_file=log_file_path, log_to_console=True)

    base_data_gen_config = config.get('data_generation', {})
    base_output_dir = base_data_gen_config.get('base_output_dir', 'data/processed') # 使用 base_output_dir

    # --- 多尺度生成设置 ---
    resolutions = base_data_gen_config.get('resolutions', [(64, 64)]) # 默认为单尺度
    domain_size_x = base_data_gen_config.get('domain_size_x', 10000.0)
    domain_size_y = base_data_gen_config.get('domain_size_y', 10000.0)

    logging.info(f"开始多尺度数据生成，分辨率: {resolutions}")
    logging.info(f"基础输出目录: {base_output_dir}")
    logging.info(f"物理域尺寸: {domain_size_x} x {domain_size_y}")

    # --- 迭代处理每个分辨率 ---
    for height, width in resolutions:
        logging.info(f"--- 处理分辨率: {height}x{width} ---")

        # 创建此分辨率的特定配置副本
        try:
            from omegaconf import OmegaConf
            res_config = OmegaConf.create(OmegaConf.to_container(config, resolve=False)) # 创建可修改副本
        except ImportError:
             import copy
             res_config = copy.deepcopy(config) # Fallback

        # 确保 data_generation 部分存在且可修改
        if 'data_generation' not in res_config: res_config['data_generation'] = {}
        res_data_gen_config = res_config['data_generation']
        if 'simulation_params' not in res_data_gen_config: res_data_gen_config['simulation_params'] = {}
        sim_params = res_data_gen_config['simulation_params']

        # 更新模拟参数
        sim_params['grid_shape'] = (height, width)
        sim_params['grid_length'] = [domain_size_y, domain_size_x] # 保持物理尺寸

        # 定义并创建此分辨率的输出目录
        res_output_dir = os.path.join(base_output_dir, f"resolution_{height}x{width}")
        res_data_gen_config['output_dir'] = res_output_dir # 设置特定输出目录

        # 生成此分辨率的数据集
        generate_dataset_for_resolution(res_config) # 传递修改后的配置

    logging.info("多尺度数据生成完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Fastscape 生成训练数据。")
    parser.add_argument('--config', type=str, required=True, help='配置文件的路径。')
    args = parser.parse_args()
    main(args)