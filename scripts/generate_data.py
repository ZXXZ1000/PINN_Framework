# PINN_Framework/scripts/generate_data.py
"""
使用 xsimlab 和 fastscape 生成模拟数据。
支持多分辨率和空间变化的参数。
"""

# 先导入基本模块
import os
import sys

# 设置OpenMP环境变量以解决多个OpenMP运行时库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import logging
import numpy as np
import torch
import xsimlab as xs
import fastscape # 导入 fastscape 包
from typing import Dict, Tuple, List, Union
import random # 新增：用于随机选择
import multiprocessing # 新增：用于并行化
from functools import partial # 新增：用于并行化传递参数
from scipy.ndimage import gaussian_filter # 新增：用于平滑噪声和高斯峰
from omegaconf import DictConfig, ListConfig # 新增：用于处理配置对象

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


# 尝试导入 noise 库并设置标志
try:
   import noise
   HAS_NOISE_LIB = True
   logging.info("成功导入 'noise' 库，分形噪声方法可用。")
except ImportError:
   HAS_NOISE_LIB = False
   logging.warning("无法导入 'noise' 库。'fractal' 方法将不可用，并回退到 'smooth_noise'。请运行 'pip install noise' 安装。")

# --- 辅助函数：生成空间参数场 ---
def generate_spatial_field(shape: Tuple[int, int], min_val: float, max_val: float, pattern: Union[str, List[str]] = 'random', **kwargs) -> np.ndarray:
    """根据指定模式或模式列表生成空间参数场。

    Args:
        shape: 场地的形状 (height, width).
        min_val: 场地的最小值.
        max_val: 场地的最大值.
        pattern: 'random', 'fault', 'constant', 'gaussian_peak', 'gradient', 'sine', 'smooth_noise', 'composite', 或包含这些模式的列表。
        **kwargs: 特定模式的额外参数 (例如, sigma, frequency, angle, num_components).

    Returns:
        生成的空间参数场 (numpy array).
    """
    # 如果 pattern 是列表，随机选择一个
    print(f"DEBUG: 处理模式参数: {pattern}, 类型: {type(pattern)}")

    # 处理不同类型的模式参数
    # 首先检查是否为 ListConfig，如果是，转换为 Python 列表
    if isinstance(pattern, ListConfig):
        print(f"DEBUG: pattern 是 ListConfig，转换为 list: {list(pattern)}")
        pattern = list(pattern) # 转换为标准列表以便后续处理

    if isinstance(pattern, (list, tuple)):
        pattern_list = list(pattern) # 确保是列表
        print(f"DEBUG: 处理列表/元组模式: {pattern_list}")
        if all(isinstance(p, str) for p in pattern_list):
            chosen_pattern = random.choice(pattern_list)
            print(f"DEBUG: 从列表中随机选择模式: '{chosen_pattern}'")
            logging.debug(f"从列表 {pattern_list} 中随机选择模式: '{chosen_pattern}'")
        else:
            logging.warning(f"pattern 列表元素类型异常: {pattern_list}，将使用 'random'。")
            chosen_pattern = 'random'
    elif isinstance(pattern, str):
        chosen_pattern = pattern
        print(f"DEBUG: 使用字符串模式: '{chosen_pattern}'")
    else:
        # 处理其他无法识别的类型
        pattern_type = type(pattern).__name__
        print(f"DEBUG: 模式类型不是 ListConfig、列表、元组或字符串，而是 {pattern_type}")
        logging.warning(f"pattern 类型异常: {pattern} (类型: {pattern_type})，将使用 'random'。")
        chosen_pattern = 'random'

    logging.debug(f"生成空间场: shape={shape}, min={min_val}, max={max_val}, pattern='{chosen_pattern}', kwargs={kwargs}")
    h, w = shape
    y_indices, x_indices = np.indices(shape)

    if chosen_pattern == 'random':
        field = np.random.uniform(min_val, max_val, shape)
    elif chosen_pattern == 'fault':
        # (断层生成逻辑保持不变)
        field = np.ones(shape) * min_val
        fault_pos_rel = np.random.uniform(0.3, 0.7)
        fault_angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        center_y, center_x = h / 2, w / 2
        y_rel, x_rel = y_indices - center_y, x_indices - center_x
        rotated_y = y_rel * np.cos(fault_angle) - x_rel * np.sin(fault_angle)
        split_value = (h * fault_pos_rel) - center_y
        field[rotated_y > split_value] = max_val
        logging.debug(f"生成断层模式: angle={fault_angle:.2f}, pos={fault_pos_rel:.2f}")
        # Apply smoothing if configured for fault mode
        fault_smoothing_config = kwargs.get('fault_smoothing', {})
        if fault_smoothing_config.get('apply', False) and chosen_pattern == 'fault':
            sigma_config = fault_smoothing_config.get('sigma', 1.0) # Default sigma if not specified
            actual_sigma = _get_float_sigma(sigma_config, default_value=1.0)
            if actual_sigma > 0:
                logging.debug(f"应用断层模式平滑: sigma_config={sigma_config}, actual_sigma={actual_sigma:.4f}")
                field = gaussian_filter(field, sigma=actual_sigma)
            else:
                logging.debug(f"跳过断层模式平滑: sigma_config={sigma_config}, calculated sigma <= 0")
    elif chosen_pattern == 'constant':
         field = np.full(shape, np.random.uniform(min_val, max_val)) # 值在范围内随机选定
         logging.debug(f"生成常量场，值为: {field[0,0]:.4f}")
    elif chosen_pattern == 'gaussian_peak':
        sigma = kwargs.get('sigma', min(h, w) / 8)
        center_y = np.random.uniform(0.1 * h, 0.9 * h)
        center_x = np.random.uniform(0.1 * w, 0.9 * w)
        peak_height = max_val - min_val
        field = min_val + peak_height * np.exp(-((y_indices - center_y)**2 + (x_indices - center_x)**2) / (2 * sigma**2))
        logging.debug(f"生成高斯峰: center=({center_y:.1f}, {center_x:.1f}), sigma={sigma:.1f}")
    elif chosen_pattern == 'gradient':
        angle = kwargs.get('angle', np.random.uniform(0, 2 * np.pi))
        # Rotate coordinates
        center_x, center_y = w / 2, h / 2
        x_rot = (x_indices - center_x) * np.cos(angle) + (y_indices - center_y) * np.sin(angle)
        # Normalize rotated coordinate to [0, 1]
        min_rot, max_rot = np.min(x_rot), np.max(x_rot)
        if max_rot > min_rot:
            normalized_gradient = (x_rot - min_rot) / (max_rot - min_rot)
        else:
            normalized_gradient = np.zeros(shape)
        field = min_val + (max_val - min_val) * normalized_gradient
        logging.debug(f"生成梯度模式: angle={angle:.2f}")
    elif chosen_pattern == 'sine':
        freq_y = kwargs.get('freq_y', np.random.uniform(1, 4))
        freq_x = kwargs.get('freq_x', np.random.uniform(1, 4))
        phase_y = np.random.uniform(0, 2 * np.pi)
        phase_x = np.random.uniform(0, 2 * np.pi)
        y = np.linspace(0, 2 * np.pi * freq_y, h)
        x = np.linspace(0, 2 * np.pi * freq_x, w)
        X, Y = np.meshgrid(x, y)
        normalized_field = (np.sin(Y + phase_y) + np.sin(X + phase_x)) / 2 # Normalize to [-1, 1]
        field = min_val + (max_val - min_val) * (normalized_field + 1) / 2 # Scale to [min_val, max_val]
        logging.debug(f"生成正弦模式: freq=({freq_y:.1f}, {freq_x:.1f})")
    elif chosen_pattern == 'smooth_noise':
        sigma_config = kwargs.get('sigma', max(1, min(h, w) // 16))
        actual_sigma = _get_float_sigma(sigma_config, default_value=max(1, min(h, w) // 16))
        noise = np.random.randn(*shape)
        if actual_sigma > 0:
             logging.debug(f"生成平滑噪声: sigma_config={sigma_config}, actual_sigma={actual_sigma:.4f}")
             filtered_noise = gaussian_filter(noise, sigma=actual_sigma)
        else:
             logging.warning(f"平滑噪声的 sigma 计算结果非正 ({actual_sigma})，使用原始噪声。sigma_config={sigma_config}")
             filtered_noise = noise # Fallback to unfiltered noise if sigma is invalid
        # Normalize filtered noise (approximately)
        norm_noise = (filtered_noise - np.mean(filtered_noise)) / (np.std(filtered_noise) + 1e-8)
        # Clip and scale to [0, 1]
        scaled_noise = np.clip((norm_noise + 3) / 6, 0, 1) # Assuming approx 3 std dev range
        field = min_val + (max_val - min_val) * scaled_noise
        # logging.debug(f"生成平滑噪声: sigma={sigma}") # Replaced by logging inside the if block
    elif chosen_pattern == 'composite':
        num_components = kwargs.get('num_components', random.randint(2, 3))
        weights = np.random.rand(num_components)
        weights /= np.sum(weights)
        field = np.zeros(shape)
        available_patterns = ['random', 'fault', 'gaussian_peak', 'gradient', 'sine', 'smooth_noise'] # Exclude 'constant' and 'composite'
        chosen_components = random.sample(available_patterns, k=num_components)
        logging.debug(f"生成复合模式: components={chosen_components}, weights={weights}")
        for i, comp_pattern in enumerate(chosen_components):
            # Generate component with potentially different random internal params
            comp_field = generate_spatial_field(shape, 0, 1, comp_pattern, **kwargs) # Generate in [0,1] range first
            field += weights[i] * comp_field
        # Scale the final composite field to [min_val, max_val]
        min_f, max_f = np.min(field), np.max(field)
        if max_f > min_f:
             field = min_val + (max_val - min_val) * (field - min_f) / (max_f - min_f)
        else:
             field = np.full(shape, (min_val + max_val) / 2)
        # Apply smoothing if configured for composite mode
        composite_smoothing_config = kwargs.get('composite_smoothing', {})
        if composite_smoothing_config.get('apply', False) and chosen_pattern == 'composite':
            sigma_config = composite_smoothing_config.get('sigma', 1.0) # Default sigma if not specified
            actual_sigma = _get_float_sigma(sigma_config, default_value=1.0)
            if actual_sigma > 0:
                logging.debug(f"应用复合模式平滑: sigma_config={sigma_config}, actual_sigma={actual_sigma:.4f}")
                field = gaussian_filter(field, sigma=actual_sigma)
            else:
                 logging.debug(f"跳过复合模式平滑: sigma_config={sigma_config}, calculated sigma <= 0")
    else:
        logging.warning(f"未知的空间模式: '{chosen_pattern}'。使用随机均匀分布。")
        field = np.random.uniform(min_val, max_val, shape)
    # Apply final gentle smoothing if configured
    final_smoothing_config = kwargs.get('final_smoothing', {})
    if final_smoothing_config.get('apply', False):
         sigma_config = final_smoothing_config.get('sigma', 0.5) # Default gentle sigma
         actual_sigma = _get_float_sigma(sigma_config, default_value=0.5)
         if actual_sigma > 0:
             logging.debug(f"应用最终平滑: sigma_config={sigma_config}, actual_sigma={actual_sigma:.4f}")
             field = gaussian_filter(field, sigma=actual_sigma)
         else:
             logging.debug(f"跳过最终平滑: sigma_config={sigma_config}, calculated sigma <= 0")
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

    # 打印调试信息
    print(f"DEBUG: 参数范围配置: {param_ranges}")

    for key, default_range in default_scalar_ranges.items():
        current_range_config = param_ranges.get(key, default_range)

        # 打印调试信息
        print(f"DEBUG: 处理参数 '{key}', 原始范围配置: {current_range_config}, 类型: {type(current_range_config)}")

        # 首先检查是否为 ListConfig，如果是，转换为 Python 列表
        if isinstance(current_range_config, ListConfig):
            print(f"DEBUG: '{key}' 的范围是 ListConfig，转换为 list: {list(current_range_config)}")
            current_range = list(current_range_config) # 转换为标准列表
        else:
            current_range = current_range_config # 保持原样

        print(f"DEBUG: 处理参数 '{key}', 转换后范围: {current_range}, 类型: {type(current_range)}")

        # 支持字典格式（如{"min": x, "max": y}），以及常规的list/tuple/float/int
        if isinstance(current_range, dict):
            min_val = current_range.get("min", default_range[0])
            max_val = current_range.get("max", default_range[1])
            try:
                min_val = float(min_val)
                max_val = float(max_val)
                print(f"DEBUG: 从字典提取的范围: [{min_val}, {max_val}]")
            except Exception as e:
                logging.warning(f"标量参数 '{key}' 的字典范围值无法转换为float: {current_range} ({e})，使用默认范围 {default_range}。")
                min_val, max_val = default_range
            if abs(min_val - max_val) < 1e-9:
                sampled_params[key] = float(min_val)
            else:
                sampled_params[key] = np.random.uniform(min_val, max_val)
        elif isinstance(current_range, (list, tuple)) and len(current_range) == 2:
            try:
                # 确保列表元素是浮点数
                min_val = float(current_range[0])
                max_val = float(current_range[1])
                print(f"DEBUG: 从列表/元组提取的范围: [{min_val}, {max_val}]")
                if abs(min_val - max_val) < 1e-9:
                    sampled_params[key] = float(min_val)
                else:
                    sampled_params[key] = np.random.uniform(min_val, max_val)
            except Exception as e:
                logging.warning(f"标量参数 '{key}' 的范围列表/元组无法转换为float: {current_range} ({e})，使用默认范围 {default_range}。")
                min_val, max_val = default_range
                if abs(min_val - max_val) < 1e-9:
                    sampled_params[key] = float(min_val)
                else:
                    sampled_params[key] = np.random.uniform(min_val, max_val)
        elif isinstance(current_range, (int, float)):
            sampled_params[key] = float(current_range)
            print(f"DEBUG: 使用固定值: {sampled_params[key]}")
        else:
            logging.warning(f"标量参数 '{key}' 的范围格式无效: {current_range} (类型: {type(current_range)})。使用默认范围 {default_range}。")
            min_val, max_val = default_range
            if abs(min_val - max_val) < 1e-9:
                sampled_params[key] = float(min_val)
            else:
                sampled_params[key] = np.random.uniform(min_val, max_val)

    print(f"DEBUG: 最终采样的参数: {sampled_params}")
    logging.debug(f"采样的标量参数 (最终): {sampled_params}")
    return sampled_params

# --- 辅助函数：处理 Sigma 配置 ---
def _get_float_sigma(sigma_config, default_value: float, context: str = "sigma") -> float:
    """从配置中获取 sigma 值并确保其为正浮点数。
    支持数字、字符串、列表/元组（取第一个元素）、字典/DictConfig（采样范围）。
    Args:
        sigma_config: 输入的配置值。
        default_value: 当无法解析或结果无效时的默认浮点值。
        context: 用于日志记录的上下文名称 (例如, 'sigma', 'smoothness', 'valley_width')。
    Returns:
        有效的正浮点数 sigma 值。
    """
    actual_sigma = default_value # 默认值
    # --- 新增日志 ---
    logging.debug(f"[_get_float_sigma] Context: '{context}', Input config: {sigma_config} (Type: {type(sigma_config)}), Default value: {default_value}")
    # --- 结束新增 ---

    if isinstance(sigma_config, (int, float)):
        actual_sigma = float(sigma_config)
        logging.debug(f"{context.capitalize()} 是数字: {actual_sigma}")
    elif isinstance(sigma_config, str):
        try:
            actual_sigma = float(sigma_config)
            logging.debug(f"{context.capitalize()} 是字符串，成功转换为 float: {actual_sigma}")
        except ValueError:
            logging.warning(f"无法将 {context} 字符串 '{sigma_config}' 转换为 float，使用默认值 {default_value}。")
            actual_sigma = default_value
    elif isinstance(sigma_config, (list, tuple)):
        if len(sigma_config) > 0:
            try:
                # 尝试取第一个元素作为 sigma 值
                first_element = sigma_config[0]
                actual_sigma = float(first_element)
                logging.debug(f"{context.capitalize()} 是列表/元组，使用第一个元素: {first_element} -> {actual_sigma}")
                if len(sigma_config) > 1:
                     logging.info(f"{context.capitalize()} 配置为列表/元组 {sigma_config}，仅使用第一个元素 {actual_sigma}。如需范围采样，请使用字典格式 {{'min': ..., 'max': ...}}。")
            except (ValueError, TypeError):
                logging.warning(f"无法将 {context} 列表/元组的第一个元素 '{sigma_config[0]}' 转换为 float，使用默认值 {default_value}。")
                actual_sigma = default_value
        else:
            logging.warning(f"{context.capitalize()} 配置为空列表/元组，使用默认值 {default_value}。")
            actual_sigma = default_value
    elif isinstance(sigma_config, (dict, DictConfig)):
        is_range = False
        min_key, max_key = None, None
        # OmegaConf DictConfig 需要特殊处理键检查
        if isinstance(sigma_config, DictConfig):
            if 'min' in sigma_config and 'max' in sigma_config:
                is_range = True
                min_key, max_key = 'min', 'max'
        elif isinstance(sigma_config, dict): # 标准字典
             if 'min' in sigma_config and 'max' in sigma_config:
                is_range = True
                min_key, max_key = 'min', 'max'

        if is_range:
            try:
                # OmegaConf DictConfig 支持属性访问
                if isinstance(sigma_config, DictConfig):
                    min_val = float(sigma_config.min)
                    max_val = float(sigma_config.max)
                else: # 标准字典
                    min_val = float(sigma_config['min'])
                    max_val = float(sigma_config['max'])

                if abs(min_val - max_val) < 1e-9:
                    actual_sigma = min_val
                    logging.debug(f"从范围字典 {sigma_config} 获取固定 {context}: {actual_sigma}")
                elif min_val < max_val:
                    actual_sigma = np.random.uniform(min_val, max_val)
                    logging.debug(f"从范围字典 {sigma_config} 采样 {context}: {actual_sigma:.4f}")
                else:
                    logging.warning(f"{context.capitalize()} 范围字典 {sigma_config} 中 min ({min_val}) 不小于 max ({max_val})，使用默认值 {default_value}。")
                    actual_sigma = default_value
            except (ValueError, TypeError, KeyError) as e:
                logging.warning(f"处理 {context} 范围字典 {sigma_config} 时出错: {e}。使用默认值 {default_value}。")
                actual_sigma = default_value
        else:
            logging.warning(f"{context.capitalize()} 配置为字典 {sigma_config} 但缺少 'min'/'max' 键，无法解释为范围。使用默认值 {default_value}。")
            actual_sigma = default_value
    else:
         # 其他无法处理的类型
         logging.warning(f"未知的 {context} 配置类型: {type(sigma_config)} ({sigma_config})。使用默认值 {default_value}。")
         actual_sigma = default_value

    # 确保 sigma > 0
    if actual_sigma <= 0:
        logging.warning(f"计算得到的 {context} 值 ({actual_sigma}) 非正，强制使用默认值 {default_value}。")
        actual_sigma = default_value

    # --- 新增日志 ---
    logging.debug(f"[_get_float_sigma] Context: '{context}', Final calculated value before positivity check: {actual_sigma}")
    # --- 结束新增 ---

    # 确保 sigma > 0
    if actual_sigma <= 0:
        logging.warning(f"[_get_float_sigma] Context: '{context}', Calculated value ({actual_sigma}) is not positive. Forcing to default value {default_value}.")
        actual_sigma = default_value

    # --- 新增日志 ---
    logging.debug(f"[_get_float_sigma] Context: '{context}', Final returned value: {actual_sigma}")
    # --- 结束新增 ---
    return actual_sigma

# --- 辅助函数：生成多样化初始地形 ---
def generate_initial_topography(shape: Tuple[int, int], method: Union[str, List[str]] = 'flat', amplitude: float = 1.0, **kwargs) -> np.ndarray:
    """生成多样化初始地形，支持单一方法或从方法列表中随机选择。

    Args:
        shape: 地形形状 (height, width).
        method: 'flat', 'random', 'sine', 'sine_mix', 'smooth_noise', 'gaussian_hill',
                'fractal', 'plateau', 'valley', 'composite', 或包含这些方法的列表。
        amplitude: 地形特征的主要振幅或高度。
        **kwargs: 特定方法的额外参数 (例如, frequency, sigma, octaves, persistence, lacunarity, num_components).

    Returns:
        生成的初始地形 (numpy array).
    """
    # 移除 seed 参数，因为种子在 worker 函数中基于 sample_index 设置
    # if seed is not None:
    #     np.random.seed(seed)

    # --- 新增 ListConfig 处理 ---
    if isinstance(method, ListConfig):
        logging.debug(f"Initial topography method is ListConfig, converting to list: {list(method)}")
        method = list(method)
    # --- 结束新增 ---

    # 如果 method 是列表，随机选择一个
    print(f"DEBUG: 处理初始地形方法参数: {method}, 类型: {type(method)}")

    # 处理不同类型的方法参数
    if isinstance(method, (list, tuple)):
        # 将列表转换为标准Python列表
        method_list = list(method)
        print(f"DEBUG: 将方法转换为列表: {method_list}")

        if all(isinstance(m, str) for m in method_list):
            chosen_method = random.choice(method_list)
            print(f"DEBUG: 从列表中随机选择初始地形方法: '{chosen_method}'")
            logging.debug(f"从列表 {method_list} 中随机选择初始地形方法: '{chosen_method}'")
        else:
            logging.warning(f"初始地形方法列表元素类型异常: {method_list}，将使用 'flat'。")
            chosen_method = 'flat'
    elif isinstance(method, str):
        chosen_method = method
        print(f"DEBUG: 使用字符串初始地形方法: '{chosen_method}'")
    else:
        # 处理其他类型
        method_type = type(method).__name__
        print(f"DEBUG: 方法类型不是列表或字符串，而是 {method_type}")
        logging.warning(f"初始地形方法类型异常: {method} (类型: {method_type})，将使用 'flat'。")
        chosen_method = 'flat'

    h, w = shape
    y_indices, x_indices = np.indices(shape)
    logging.debug(f"生成初始地形: shape={shape}, method='{chosen_method}', amplitude={amplitude}, kwargs={kwargs}")

    if chosen_method == 'flat':
        topo = np.zeros((h, w), dtype=np.float32)
    elif chosen_method == 'random':
        topo = amplitude * np.random.uniform(-1, 1, (h, w))
    elif chosen_method == 'sine':
        frequency = kwargs.get('frequency', np.random.uniform(1, 4))
        phase_x = np.random.uniform(0, 2 * np.pi)
        phase_y = np.random.uniform(0, 2 * np.pi)
        y = np.linspace(0, 2 * np.pi * frequency, h)
        x = np.linspace(0, 2 * np.pi * frequency, w)
        X, Y = np.meshgrid(x, y)
        topo = amplitude * (np.sin(X + phase_x) + np.sin(Y + phase_y)) / 2 # Normalize amplitude
    elif chosen_method == 'sine_mix':
        frequency = kwargs.get('frequency', np.random.uniform(1, 3))
        f1, f2, f3 = frequency, frequency * np.random.uniform(1.5, 2.5), frequency * np.random.uniform(2.5, 3.5)
        a1, a2, a3 = 1.0, np.random.uniform(0.3, 0.7), np.random.uniform(0.1, 0.4)
        p1, p2, p3 = [np.random.uniform(0, 2*np.pi) for _ in range(3)]
        y = np.linspace(0, 2 * np.pi, h)
        x = np.linspace(0, 2 * np.pi, w)
        X, Y = np.meshgrid(x, y)
        term1 = a1 * np.sin(f1 * X + p1)
        term2 = a2 * np.sin(f2 * Y + p2)
        term3 = a3 * np.sin(f3 * X + f3 * Y + p3)
        raw_topo = term1 + term2 + term3
        topo = amplitude * raw_topo / (a1 + a2 + a3) # Normalize amplitude approx
    elif chosen_method == 'smooth_noise':
        sigma_config = kwargs.get('sigma', max(1, min(h, w) // 16))
        actual_sigma = _get_float_sigma(sigma_config, default_value=max(1, min(h, w) // 16))
        noise = np.random.randn(h, w)
        topo = amplitude * gaussian_filter(noise, sigma=actual_sigma)
        logging.debug(f"生成平滑噪声: sigma_config={sigma_config}, actual_sigma={actual_sigma:.4f}")
    elif chosen_method == 'gaussian_hill':
        sigma_config = kwargs.get('sigma', min(h, w) / np.random.uniform(6, 12))
        # 使用辅助函数处理 sigma，并传入 context
        actual_sigma = _get_float_sigma(sigma_config, default_value=min(h, w) / 8, context="gaussian_hill_sigma")
        center_y = np.random.uniform(0.1 * h, 0.9 * h)
        center_x = np.random.uniform(0.1 * w, 0.9 * w)
        if actual_sigma > 1e-6: # 避免除以零或非常小的值
            topo = amplitude * np.exp(-((y_indices - center_y)**2 + (x_indices - center_x)**2) / (2 * actual_sigma**2))
            logging.debug(f"生成高斯山: center=({center_y:.1f}, {center_x:.1f}), sigma_config={sigma_config}, actual_sigma={actual_sigma:.4f}")
        else:
            logging.warning(f"高斯山的 actual_sigma ({actual_sigma}) 过小，生成平坦地形代替。")
            topo = np.zeros(shape, dtype=np.float32)
    elif chosen_method == 'fractal':
        # --- 修改：使用 HAS_NOISE_LIB 判断 ---
        if HAS_NOISE_LIB:
            scale = kwargs.get('scale', min(h, w) / np.random.uniform(4, 10))
            octaves = kwargs.get('octaves', random.randint(4, 8))
            persistence = kwargs.get('persistence', np.random.uniform(0.4, 0.6))
            lacunarity = kwargs.get('lacunarity', np.random.uniform(1.8, 2.2))
            topo = np.zeros(shape)
            # 使用全局导入的 noise
            for i in range(h):
                for j in range(w):
                    topo[i][j] = noise.pnoise2(i/scale,
                                               j/scale,
                                               octaves=octaves,
                                               persistence=persistence,
                                               lacunarity=lacunarity,
                                               base=random.randint(0, 100))
            # Scale noise to roughly [-1, 1] then apply amplitude
            min_noise, max_noise = np.min(topo), np.max(topo)
            if max_noise > min_noise:
                topo = 2 * (topo - min_noise) / (max_noise - min_noise) - 1
            topo *= amplitude
            logging.debug(f"生成分形噪声: scale={scale:.1f}, octaves={octaves}, persistence={persistence:.2f}, lacunarity={lacunarity:.2f}")
        else:
            logging.warning("'fractal' 方法需要 'noise' 库，但未找到。回退到 'smooth_noise'。")
            # 确保传递正确的 kwargs 给回退调用
            fallback_kwargs = kwargs.copy()
            # 移除 fractal 特有的参数，避免影响 smooth_noise
            fallback_kwargs.pop('scale', None)
            fallback_kwargs.pop('octaves', None)
            fallback_kwargs.pop('persistence', None)
            fallback_kwargs.pop('lacunarity', None)
            topo = generate_initial_topography(shape, method='smooth_noise', amplitude=amplitude, **fallback_kwargs)
        # --- 结束修改 ---
    elif chosen_method == 'plateau':
        p_h = np.random.uniform(0.2, 0.5) * h # Plateau height extent
        p_w = np.random.uniform(0.2, 0.5) * w # Plateau width extent
        c_y = np.random.uniform(p_h/2, h - p_h/2)
        c_x = np.random.uniform(p_w/2, w - p_w/2)
        smoothness_config = kwargs.get('smoothness', min(h, w) / 32)
        # 使用辅助函数处理 smoothness，并传入 context
        actual_smoothness = _get_float_sigma(smoothness_config, default_value=min(h, w) / 32, context="plateau_smoothness")
        topo = np.zeros(shape)
        mask = (np.abs(y_indices - c_y) < p_h/2) & (np.abs(x_indices - c_x) < p_w/2)
        topo[mask] = amplitude
        if actual_smoothness > 0:
            topo = gaussian_filter(topo, sigma=actual_smoothness)
            logging.debug(f"生成高原: center=({c_y:.1f}, {c_x:.1f}), size=({p_h:.1f}, {p_w:.1f}), smoothness_config={smoothness_config}, actual_smoothness={actual_smoothness:.4f}")
        else:
             logging.debug(f"生成高原: center=({c_y:.1f}, {c_x:.1f}), size=({p_h:.1f}, {p_w:.1f}), 未应用平滑 (smoothness <= 0)")
    elif chosen_method == 'valley': # 修正缩进
        valley_width_config = kwargs.get('valley_width', min(h, w) / np.random.uniform(5, 10))
        # 使用辅助函数处理 valley_width，并传入 context
        actual_valley_width = _get_float_sigma(valley_width_config, default_value=min(h, w) / 8, context="valley_width")
        angle = np.random.uniform(0, np.pi)
        depth = amplitude # Use amplitude as depth
        center_x, center_y = w / 2, h / 2
        x_rot = (x_indices - center_x) * np.cos(angle) + (y_indices - center_y) * np.sin(angle)
        # Gaussian profile for the valley
        if actual_valley_width > 1e-6: # 避免除以零或非常小的值
            topo = -depth * np.exp(-(x_rot**2) / (2 * actual_valley_width**2))
            logging.debug(f"生成河谷: angle={angle:.2f}, width_config={valley_width_config}, actual_width={actual_valley_width:.4f}, depth={depth:.2f}")
        else:
            logging.warning(f"河谷的 actual_valley_width ({actual_valley_width}) 过小，生成平坦地形代替。")
            topo = np.zeros(shape, dtype=np.float32)
        # Add some base noise/variation
        # Note: The recursive call here might still pass the original 'valley_width' in kwargs if it exists.
        # Consider cleaning kwargs before recursive calls if this becomes an issue.
        base_variation_sigma_config = kwargs.get('sigma', min(h,w)/8) # Use sigma from kwargs if available for base variation
        # 使用辅助函数处理基础噪声的 sigma，并传入 context
        actual_base_sigma = _get_float_sigma(base_variation_sigma_config, default_value=min(h,w)/16, context="valley_base_noise_sigma")
        # 确保传递处理后的 sigma
        base_variation = generate_initial_topography(shape, method='smooth_noise', amplitude=amplitude * 0.1, sigma=actual_base_sigma)
        topo += base_variation
        # logging.debug(f"生成河谷: angle={angle:.2f}, width={valley_width:.1f}, depth={depth:.2f}") # Replaced by logging inside the if block
    elif chosen_method == 'composite': # 修正缩进
        num_components = kwargs.get('num_components', random.randint(2, 4))
        weights = np.random.rand(num_components)
        weights /= np.sum(weights) # 修正缩进
        topo = np.zeros(shape)
        available_methods = ['random', 'sine', 'smooth_noise', 'gaussian_hill', 'fractal', 'plateau', 'valley'] # Exclude 'flat' and 'composite'
        chosen_components = random.sample(available_methods, k=num_components)
        logging.debug(f"生成复合地形: components={chosen_components}, weights={weights}")
        for i, comp_method in enumerate(chosen_components):
            # Generate component with potentially different random internal params and scaled amplitude
            comp_amplitude = amplitude * weights[i] * num_components # Scale amplitude for each component
            comp_topo = generate_initial_topography(shape, method=comp_method, amplitude=comp_amplitude, **kwargs)
            topo += comp_topo
        # Optional: Add a final smoothing pass
        final_smoothness_config = kwargs.get('final_smoothness', 0)
        # 使用辅助函数处理最终平滑度，并传入 context
        actual_final_smoothness = _get_float_sigma(final_smoothness_config, default_value=0, context="composite_final_smoothness")
        # --- 新增检查和日志 ---
        if actual_final_smoothness <= 0:
            logging.debug(f"复合地形的最终平滑度计算结果非正 ({actual_final_smoothness:.4f})，跳过最终平滑。Config: {final_smoothness_config}")
        else:
            topo = gaussian_filter(topo, sigma=actual_final_smoothness)
            logging.debug(f"复合地形应用最终平滑: smoothness_config={final_smoothness_config}, actual_smoothness={actual_final_smoothness:.4f}")
        # --- 结束新增 ---
    else: # 修正缩进
        logging.warning(f"未知的初始地形方法: '{chosen_method}'。使用 'flat'。")
        topo = np.zeros((h, w), dtype=np.float32)
    # Apply final general smoothing if configured and method is not 'flat'
    final_smoothing_config = kwargs.get('smoothing', {}) # Use 'smoothing' key from config # 修正缩进
    if final_smoothing_config.get('apply', False) and chosen_method != 'flat': # 修正缩进
        sigma_config = final_smoothing_config.get('sigma', 1.0) # Default sigma
        # 使用辅助函数处理最终通用平滑的 sigma，并传入 context
        actual_sigma = _get_float_sigma(sigma_config, default_value=1.0, context="final_general_smoothing_sigma")
        if actual_sigma > 0: # Only apply if sigma is valid positive
             logging.debug(f"应用最终通用初始地形平滑: method='{chosen_method}', sigma_config={sigma_config}, actual_sigma={actual_sigma:.4f}")
             topo = gaussian_filter(topo, sigma=actual_sigma)
        else:
             logging.debug(f"跳过最终通用初始地形平滑: method='{chosen_method}', sigma_config={sigma_config}, calculated sigma <= 0")
    return topo.astype(np.float32) # 修正缩进

# --- 辅助函数：生成单个模拟样本 (修改为适合并行调用) ---
def generate_single_sample_worker(sample_index: int, config: Dict) -> bool:
    """工作函数，用于并行生成单个样本。"""
    # 为每个进程/样本设置不同的随机种子，以增加多样性
    # 可以结合基础种子和样本索引
    base_seed = config.get('data_generation', {}).get('random_seed', None)
    if base_seed is not None:
        process_seed = base_seed + sample_index
        np.random.seed(process_seed)
        random.seed(process_seed + 1) # For python's random module
        # torch seeding might be needed if using torch random ops directly here
        # torch.manual_seed(process_seed + 2)
        logging.debug(f"样本 {sample_index+1}: 设置随机种子 {process_seed}")
    else:
        # 如果没有基础种子，依赖系统为每个进程提供的不同状态
        logging.debug(f"样本 {sample_index+1}: 未设置特定随机种子")

    # 使用 OmegaConf 访问配置，提供默认值
    try:
        from omegaconf import OmegaConf
        if not isinstance(config, OmegaConf): # Ensure it's OmegaConf object for easy access
             config = OmegaConf.create(config)
    except ImportError:
        logging.warning("OmegaConf 未安装，使用标准字典访问，可能缺少默认值处理。")

    data_gen_config = config.get('data_generation', {}) # Use .get for safety
    sim_config = data_gen_config.get('simulation_params', {})
    parameter_type = data_gen_config.get('parameter_type', 'scalar').lower()
    scalar_param_ranges = data_gen_config.get('parameter_ranges', {})
    spatial_param_config = data_gen_config.get('spatial_parameter_config', {})
    output_dir = data_gen_config.get('output_dir') # 从特定分辨率的配置中获取

    # 读取初始地形生成方式配置 (支持列表)
    initial_topo_config = data_gen_config.get('initial_topography', {})
    initial_topo_method = initial_topo_config.get('method', 'flat') # Can be str or list
    # 获取 amplitude 配置，可能是数字、字符串或字典/DictConfig
    amplitude_config = initial_topo_config.get('amplitude', 1.0)
    initial_topo_amplitude = 1.0 # Default value

    # 尝试处理 OmegaConf DictConfig 或标准字典
    is_dict_like = False
    try:
        # 优先检查 OmegaConf 类型，因为它可能覆盖 isinstance(dict)
        from omegaconf import DictConfig
        if isinstance(amplitude_config, DictConfig):
            is_dict_like = True
            if 'min' in amplitude_config and 'max' in amplitude_config:
                min_val = float(amplitude_config.min) # 使用点访问
                max_val = float(amplitude_config.max)
                if abs(min_val - max_val) < 1e-9:
                    initial_topo_amplitude = min_val
                else:
                    initial_topo_amplitude = np.random.uniform(min_val, max_val)
                logging.debug(f"从 DictConfig 范围 [{min_val}, {max_val}] 采样初始地形振幅: {initial_topo_amplitude:.4f}")
            else:
                 logging.warning(f"DictConfig {amplitude_config} 缺少 'min' 或 'max' 键。使用默认振幅 1.0。")
                 initial_topo_amplitude = 1.0 # Reset to default if keys missing
        elif isinstance(amplitude_config, dict): # 如果不是 DictConfig，再检查标准 dict
             is_dict_like = True
             if 'min' in amplitude_config and 'max' in amplitude_config:
                 min_val = float(amplitude_config['min'])
                 max_val = float(amplitude_config['max'])
                 if abs(min_val - max_val) < 1e-9:
                     initial_topo_amplitude = min_val
                 else:
                     initial_topo_amplitude = np.random.uniform(min_val, max_val)
                 logging.debug(f"从 dict 范围 [{min_val}, {max_val}] 采样初始地形振幅: {initial_topo_amplitude:.4f}")
             else:
                 logging.warning(f"字典 {amplitude_config} 缺少 'min' 或 'max' 键。使用默认振幅 1.0。")
                 initial_topo_amplitude = 1.0 # Reset to default if keys missing

    except ImportError:
        # OmegaConf 未安装，只检查标准 dict
        if isinstance(amplitude_config, dict):
            is_dict_like = True
            if 'min' in amplitude_config and 'max' in amplitude_config:
                try:
                    min_val = float(amplitude_config['min'])
                    max_val = float(amplitude_config['max'])
                    if abs(min_val - max_val) < 1e-9:
                        initial_topo_amplitude = min_val
                    else:
                        initial_topo_amplitude = np.random.uniform(min_val, max_val)
                    logging.debug(f"从 dict 范围 [{min_val}, {max_val}] 采样初始地形振幅: {initial_topo_amplitude:.4f}")
                except (ValueError, TypeError) as e:
                    logging.warning(f"无法从字典 {amplitude_config} 解析初始地形振幅范围，错误: {e}。使用默认值 1.0。")
                    initial_topo_amplitude = 1.0
            else:
                logging.warning(f"字典 {amplitude_config} 缺少 'min' 或 'max' 键。使用默认振幅 1.0。")
                initial_topo_amplitude = 1.0
    except (ValueError, TypeError, AttributeError) as e:
         # 处理 DictConfig 或 dict 转换/访问错误
         logging.warning(f"解析字典/DictConfig {amplitude_config} 时出错: {e}。使用默认振幅 1.0。")
         initial_topo_amplitude = 1.0

    # 如果不是字典类型，尝试处理数字或字符串
    if not is_dict_like:
        if isinstance(amplitude_config, (int, float, str)):
            try:
                initial_topo_amplitude = float(amplitude_config)
                logging.debug(f"使用固定或可转换的初始地形振幅: {initial_topo_amplitude:.4f}")
            except (ValueError, TypeError) as e:
                logging.warning(f"无法将 {amplitude_config} (类型: {type(amplitude_config)}) 转换为初始地形振幅，错误: {e}。使用默认值 1.0。")
                initial_topo_amplitude = 1.0
        else:
            logging.warning(f"未知的初始地形振幅类型: {type(amplitude_config)}，值: {amplitude_config}。使用默认值 1.0。")
            initial_topo_amplitude = 1.0
    # Pass relevant kwargs from config to the generation function
    initial_topo_kwargs = {
        k: v for k, v in initial_topo_config.items()
        if k not in ['method', 'amplitude']
    }
    # initial_topo_seed is handled above per-process

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

    logging.info(f"--- [Worker {os.getpid()}] 正在生成样本 {sample_index+1}，形状 {grid_shape} ---")
    try:
        # 1. 生成/采样参数
        final_params = sample_scalar_parameters(scalar_param_ranges)
        if parameter_type == 'spatial':
            spatial_capable_params = ['uplift__rate', 'spl__k_coef']
            spatial_params = {}
            for key in spatial_capable_params:
                 if key in spatial_param_config:
                      config_for_key = spatial_param_config[key]
                      pattern = config_for_key.get('pattern', 'constant') # Can be str or list
                      min_val = float(config_for_key.get('min', 0.0))
                      max_val = float(config_for_key.get('max', 1.0))
                      # Extract potential kwargs for the pattern generation
                      pattern_kwargs = {
                          k: v for k, v in config_for_key.items()
                          if k not in ['pattern', 'min', 'max']
                      }
                      # 直接将pattern传递给generate_spatial_field函数，该函数已经能够处理列表或字符串模式
                      print(f"[Worker {os.getpid()} Sample {sample_index+1} DBG] 生成空间场: key={key}, min={min_val}, max={max_val}, pattern='{pattern}', kwargs={pattern_kwargs}")
                      try:
                          # Debug print grid_shape before generating spatial field
                          print(f"[Worker {os.getpid()} Sample {sample_index+1}] 空间场生成前的 grid_shape: {grid_shape}, 类型: {type(grid_shape)}")
                          spatial_params[key] = generate_spatial_field(grid_shape, min_val, max_val, pattern, **pattern_kwargs)
                          # Debug print the shape of the generated field
                          field_shape = spatial_params[key].shape
                          print(f"[Worker {os.getpid()} Sample {sample_index+1} DBG] 为 '{key}' 生成了空间场。形状: {field_shape}")
                      except Exception as e:
                          print(f"[Worker {os.getpid()} Sample {sample_index+1}] 错误: 生成 '{key}' 的空间场时出错: {e}")
                          # Create a default constant field as fallback
                          spatial_params[key] = np.full(grid_shape, (min_val + max_val) / 2, dtype=np.float32)
                          print(f"[Worker {os.getpid()} Sample {sample_index+1}] 已创建默认常量场作为回退。")
                 else:
                      logging.debug(f"未找到空间参数 '{key}' 的配置。将保持标量。")
            final_params.update(spatial_params)

        param_types = {k: type(v).__name__ for k, v in final_params.items()}
        logging.debug(f"样本 {sample_index+1} 的最终参数: {param_types}")

        # 2. 生成初始地形 (在模拟前生成)
        initial_topo_np = generate_initial_topography(grid_shape, method=initial_topo_method, amplitude=initial_topo_amplitude, **initial_topo_kwargs)
        initial_topo_tensor = torch.from_numpy(initial_topo_np).unsqueeze(0)

        # 3. 设置 xsimlab 模拟
        # 读取输出步数配置
        output_steps = data_gen_config.get('output_steps', 101)  # 默认101，可在yaml中配置
        steps = output_steps  # 设置输出步数

        # 先删除 init_topography 进程，然后直接设置初始地形
        model_to_use = basic_model.drop_processes('init_topography')

        # 创建坐标数组用于DataArray
        y_coords = np.arange(grid_shape[0])
        x_coords = np.arange(grid_shape[1])

        # 创建初始地形的DataArray
        import xarray as xr
        initial_topo_da = xr.DataArray(initial_topo_np, coords={'y': y_coords, 'x': x_coords}, dims=('y', 'x'))

        # 准备参数DataArray
        param_arrays = {}
        for key, value in final_params.items():
            if isinstance(value, np.ndarray) and value.ndim == 2:
                if key in ['spl__area_exp', 'spl__slope_exp']:
                    logging.warning(f"参数 '{key}' 生成为空间场，但模型要求为标量。使用平均值。")
                    param_arrays[key] = float(value.mean())
                else:
                    # 创建DataArray格式的空间参数
                    param_arrays[key] = xr.DataArray(value, coords={'y': y_coords, 'x': x_coords}, dims=('y', 'x'))
            else:
                param_arrays[key] = float(value)

        # 准备输入变量字典
        input_vars_dict = {
            'grid__shape': list(grid_shape),
            'grid__length': list(grid_length),
            'boundary__status': sim_config.get('boundary_status', 'fixed_value'),
            'topography__elevation': initial_topo_da,  # 使用DataArray格式的初始地形
        }

        # 添加参数到输入字典
        for key, value in param_arrays.items():
            input_vars_dict[key] = value

        # 设置输出变量 - 增加更多输出
        output_vars_dict = {
            'topography__elevation': 'out'
        }

        # 创建模拟设置 - 使用用户示例中的时钟设置方式
        in_ds = xs.create_setup(
            model=model_to_use,
            clocks={
                'time': np.linspace(0., run_time_total, 100+1),
                'out': np.linspace(0., run_time_total, steps)
            },
            master_clock='time',
            input_vars=input_vars_dict,
            output_vars=output_vars_dict
        )

        # 4. 运行模拟 - 使用链式 API 方式运行
        logging.info(f"样本 {sample_index+1}: 运行 xsimlab 模拟...")
        # 直接运行模型，不使用with语句
        out_ds = in_ds.xsimlab.run(model=model_to_use)
        logging.info(f"样本 {sample_index+1}: 模拟完成。")

        # 5. 提取和格式化数据
        if 'topography__elevation' not in out_ds or len(out_ds['out']) < 1:
             logging.error(f"样本 {sample_index+1}: 输出数据集中缺少 'topography__elevation' 或输出时间点不足。")
             return False

        # 提取最终地形 - 使用最后一个时间点
        final_topo_xr = out_ds['topography__elevation'].isel(out=-1)
        final_topo_tensor = torch.from_numpy(final_topo_xr.values.astype(np.float32)).unsqueeze(0)

        # 准备保存的字典
        sample_output = {
            'initial_topo': initial_topo_tensor, # 使用我们生成的初始地形
            'final_topo': final_topo_tensor,
            'run_time': torch.tensor(run_time_total, dtype=torch.float32)
        }
        # 添加参数
        param_mapping = {
            'uplift__rate': 'uplift_rate', 'spl__k_coef': 'k_f',
            'diffusion__diffusivity': 'k_d', 'spl__area_exp': 'm', 'spl__slope_exp': 'n'
        }
        for xsim_key, data_key in param_mapping.items():
            if xsim_key in final_params:
                value = final_params[xsim_key]
                sample_output[data_key] = value if isinstance(value, np.ndarray) else torch.tensor(value, dtype=torch.float32)

        # 6. 保存样本
        filename = f"sample_{sample_index:05d}.pt"
        filepath = os.path.join(output_dir, filename)
        save_data_sample(sample_output, filepath)
        logging.info(f"样本 {sample_index+1}: 已成功保存到 {filepath}")
        return True

    except Exception as e:
        logging.error(f"生成或保存样本 {sample_index+1} 失败: {e}", exc_info=True)
        # 在并行环境中，打印堆栈跟踪可能很有用
        import traceback
        traceback.print_exc()
        return False

# --- 辅助函数：生成特定分辨率的数据集 (修改为并行) ---
def generate_dataset_for_resolution(config: Dict):
    """为特定配置（分辨率）生成数据集，支持并行化。"""
    try:
        from omegaconf import OmegaConf
        if not isinstance(config, OmegaConf): config = OmegaConf.create(config)
    except ImportError: pass

    data_gen_config = config.get('data_generation', {})
    num_samples = data_gen_config.get('num_samples', 10)
    output_dir = data_gen_config.get('output_dir')
    sim_params_conf = data_gen_config.get('simulation_params', {})
    grid_shape = tuple(sim_params_conf.get('grid_shape', [0,0]))
    # 新增：读取并行工作进程数
    num_workers = data_gen_config.get('num_workers', 1)
    if num_workers == 0:
        num_workers = os.cpu_count()
        logging.info(f"num_workers 设置为 0，自动检测到 {num_workers} 个 CPU 核心。")
    elif num_workers < 0:
        num_workers = 1
        logging.warning("num_workers 不能为负数，已重置为 1。")
    else:
        num_workers = int(num_workers)

    if not output_dir or grid_shape == (0,0):
        logging.error("generate_dataset_for_resolution: 配置中缺少 output_dir 或 grid_shape。")
        return

    logging.info(f"开始生成数据集: {num_samples} 个样本，形状 {grid_shape}，保存到 {output_dir}")
    logging.info(f"使用 {num_workers} 个工作进程进行并行生成。")
    os.makedirs(output_dir, exist_ok=True)

    # 准备传递给工作进程的参数
    # 使用 partial 冻结 config 参数
    worker_func = partial(generate_single_sample_worker, config=config)
    sample_indices = range(num_samples)

    results = []
    if num_workers > 1:
        # 使用多进程池
        try:
            # 在 Windows 上，'fork' 可能不可用或不安全，'spawn' 是更安全的选择
            # context = multiprocessing.get_context('spawn')
            # with context.Pool(processes=num_workers) as pool:
            # 尝试默认上下文，如果不行再考虑 spawn
            with multiprocessing.Pool(processes=num_workers) as pool:
                # map 或 starmap 可以自动处理任务分发和结果收集
                # map 只接受一个可迭代参数，所以我们使用 partial
                results = pool.map(worker_func, sample_indices)
        except Exception as e:
            logging.error(f"启动或运行多进程池失败: {e}", exc_info=True)
            logging.warning("回退到单进程模式。")
            # 回退到单进程
            results = [worker_func(i) for i in sample_indices]
    else:
        # 单进程执行
        logging.info("以单进程模式运行生成。")
        results = [worker_func(i) for i in sample_indices]

    # 统计成功/失败的样本数
    successful_count = sum(1 for r in results if r is True)
    failed_count = num_samples - successful_count

    logging.info(f"数据集生成完成，形状 {grid_shape}。成功生成 {successful_count}/{num_samples} 个样本。")
    if failed_count > 0:
        logging.warning(f"{failed_count} 个样本生成失败。请检查上面的错误日志。")


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
    # 内置配置文件路径，不再需要通过命令行参数指定
    config_path = os.path.join(project_root, 'configs', 'data_gen_config.yaml')
    print(f"使用内置配置文件路径: {config_path}")

    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)

    # 创建一个类似于命令行参数的对象
    class Args:
        def __init__(self):
            self.config = config_path

    args = Args()
    main(args)

    # 保留原始命令行参数处理，以便在需要时可以覆盖内置路径
    # parser = argparse.ArgumentParser(description="使用 Fastscape 生成训练数据。")
    # parser.add_argument('--config', type=str, default=config_path, help='配置文件的路径。')
    # args = parser.parse_args()
    # main(args)
