# PINN_Framework/src/utils.py
"""
通用辅助函数模块。
"""

import os
import sys
import logging
import random
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf # Using OmegaConf for config loading
import time
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional, Tuple, Any, Union, List, Callable

def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None, log_to_console: bool = True) -> logging.Logger:
    """
    设置日志系统，支持文件和控制台输出。

    Args:
        log_level: 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        log_file: 日志文件路径。如果为 None，则不记录到文件。
        log_to_console: 是否输出到控制台。

    Returns:
        配置好的根日志记录器。
    """
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    # Ensure level is valid, default to INFO if not
    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(level)

    # 清除现有处理器以避免重复日志
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 控制台处理器
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        root_logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            # 使用 RotatingFileHandler 进行日志轮换
            file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
            file_handler.setFormatter(log_format)
            root_logger.addHandler(file_handler)
        except Exception as e:
            root_logger.error(f"无法设置文件日志处理器到 {log_file}: {e}")

    root_logger.info(f"日志系统初始化完成。级别: {log_level.upper()}")
    return root_logger

def get_device(device_config: str = 'auto') -> torch.device:
    """根据配置和可用性获取 torch 设备。"""
    if device_config == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_config

    if device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA 指定但不可用。回退到 CPU。")
        device = "cpu"

    selected_device = torch.device(device)
    logging.info(f"使用设备: {selected_device}")
    return selected_device

def set_seed(seed: Optional[int]):
    """设置随机种子以保证可复现性。"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) # 适用于多 GPU
            # 确保确定性算法（可能影响性能）
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
        logging.info(f"随机种子设置为 {seed}")

def save_data_sample(data_dict: Dict, filepath: str):
    """将数据样本字典保存到 .pt 文件。"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # 兼容旧版本 PyTorch，尝试使用 weights_only 参数
        try:
            torch.save(data_dict, filepath, weights_only=True)
        except TypeError:
            # 旧版本 PyTorch 不支持 weights_only 参数
            torch.save(data_dict, filepath)
    except Exception as e:
        logging.error(f"保存文件 {filepath} 时出错: {e}")

def load_config(config_path: str) -> Union[Dict, OmegaConf]:
    """使用 OmegaConf 加载 YAML 配置文件。"""
    try:
        conf = OmegaConf.load(config_path)
        logging.info(f"已使用 OmegaConf 从 {config_path} 加载配置")
        # 返回 OmegaConf 对象，允许插值和访问属性
        return OmegaConf.to_container(conf, resolve=False)
    except FileNotFoundError:
        logging.error(f"配置文件未找到: {config_path}")
        raise
    except Exception as e:
        logging.error(f"加载配置文件 {config_path} 时出错: {e}")
        raise

def save_config(config: Union[Dict, OmegaConf], filepath: str):
    """将配置字典或 OmegaConf 对象保存到 YAML 文件。"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # 如果是 OmegaConf 对象，直接使用 OmegaConf.save 保存
        if OmegaConf.is_config(config): # 更可靠的检查方式
            # resolve=True 解析插值后保存
            OmegaConf.save(config, filepath, resolve=True)
        elif isinstance(config, dict):
            # 如果是普通字典，使用 yaml.dump
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        else:
            raise TypeError(f"Unsupported config type for saving: {type(config)}")
        logging.info(f"配置已保存到 {filepath}")
    except Exception as e:
        logging.error(f"保存配置文件 {filepath} 时出错: {e}")

def _try_expand_tensor(tensor: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
    """尝试将张量扩展到目标形状，如果失败则引发错误。"""
    try:
        return tensor.expand(target_shape)
    except RuntimeError:
        raise ValueError(f"无法将形状为 {tensor.shape} 的张量广播到目标形状 {target_shape}")

def prepare_parameter(param_value: Any, target_shape: Optional[Tuple[int, ...]] = None,
                      batch_size: Optional[int] = None, device: Optional[torch.device] = None,
                      dtype: Optional[torch.dtype] = None, param_name: str = "unknown") -> Optional[torch.Tensor]:
    """
    统一处理不同形式的参数值（标量、张量、数组），确保输出一致的形状和类型。
    主要用于处理物理参数，使其与模型输入/计算兼容。

    Args:
        param_value: 参数值。
        target_shape: 目标空间形状 (H, W) 或完整形状 (B, C, H, W)。
        batch_size: 批次大小。
        device: 目标设备。
        dtype: 目标数据类型。
        param_name: 参数名称（用于日志）。

    Returns:
        处理后的参数张量，或在无法处理时返回 None。
    """
    # 确定设备和类型
    ref_tensor = None
    if isinstance(param_value, torch.Tensor): ref_tensor = param_value
    if device is None: device = ref_tensor.device if ref_tensor is not None else torch.device('cpu')
    if dtype is None: dtype = ref_tensor.dtype if ref_tensor is not None else torch.float32

    # 处理 None 值
    if param_value is None:
        logging.warning(f"参数 '{param_name}' 为 None。")
        # 如果提供了目标形状，可以返回零张量
        if target_shape is not None and batch_size is not None:
             full_shape = (batch_size, 1, *target_shape) if len(target_shape) == 2 else target_shape
             logging.warning(f"返回形状为 {full_shape} 的零张量。")
             return torch.zeros(full_shape, device=device, dtype=dtype)
        return None # 否则返回 None

    # 处理标量
    if isinstance(param_value, (int, float)):
        value = float(param_value)
        if target_shape is None or batch_size is None:
            return torch.tensor(value, device=device, dtype=dtype)
        else:
            full_shape = (batch_size, 1, *target_shape) if len(target_shape) == 2 else target_shape
            return torch.full(full_shape, value, device=device, dtype=dtype)

    # 处理张量
    elif isinstance(param_value, torch.Tensor):
        param_tensor = param_value.to(device=device, dtype=dtype)
        if target_shape is None or batch_size is None:
            return param_tensor # 不需要调整形状

        full_target_shape = (batch_size, 1, *target_shape) if len(target_shape) == 2 else target_shape

        # 尝试广播/调整形状
        if param_tensor.shape == full_target_shape: return param_tensor
        elif param_tensor.numel() == 1: return param_tensor.expand(full_target_shape)
        elif param_tensor.ndim == 1 and param_tensor.shape[0] == batch_size: return param_tensor.view(batch_size, 1, 1, 1).expand(full_target_shape)
        elif param_tensor.ndim == 2 and param_tensor.shape == target_shape: return param_tensor.unsqueeze(0).unsqueeze(0).expand(full_target_shape)
        elif param_tensor.ndim == 3 and param_tensor.shape[0] == batch_size and param_tensor.shape[1:] == target_shape: return param_tensor.unsqueeze(1)
        else: return _try_expand_tensor(param_tensor, full_target_shape) # 最后尝试直接广播

    # 处理 NumPy 数组
    elif isinstance(param_value, np.ndarray):
        return prepare_parameter(torch.from_numpy(param_value), target_shape, batch_size, device, dtype, param_name)

    else:
        raise TypeError(f"参数 '{param_name}' 的类型 '{type(param_value)}' 不受支持。")


def standardize_coordinate_system(coords: Union[Dict, Tuple, List],
                                  domain_x: Tuple[float, float] = (0.0, 1.0),
                                  domain_y: Tuple[float, float] = (0.0, 1.0),
                                  normalize: bool = False,
                                  device: Optional[torch.device] = None,
                                  dtype: Optional[torch.dtype] = None) -> Dict[str, torch.Tensor]:
    """
    标准化坐标系，确保坐标表示一致。

    Args:
        coords: 坐标字典 {'x': x, 'y': y, ...} 或元组/列表 (x, y, ...)。
        domain_x: x 轴的物理域 (min, max)。
        domain_y: y 轴的物理域 (min, max)。
        normalize: 是否将物理坐标归一化到 [0, 1] 范围。
        device: 目标设备。
        dtype: 目标数据类型。

    Returns:
        标准化的坐标字典 {'x': tensor, 'y': tensor, ...}。
    """
    if isinstance(coords, dict):
        x, y = coords.get('x'), coords.get('y')
        extra_keys = {k: v for k, v in coords.items() if k not in ['x', 'y']}
        if x is None or y is None: raise ValueError("坐标字典必须包含 'x' 和 'y' 键")
    elif isinstance(coords, (list, tuple)) and len(coords) >= 2:
        x, y = coords[0], coords[1]
        extra_keys = {} # 假设元组/列表只有 x, y
    else:
        raise TypeError("坐标必须是字典或至少包含两个元素的元组/列表")

    # 确定设备和类型
    ref_tensor = x if isinstance(x, torch.Tensor) else (y if isinstance(y, torch.Tensor) else None)
    if device is None: device = ref_tensor.device if ref_tensor is not None else torch.device('cpu')
    if dtype is None: dtype = ref_tensor.dtype if ref_tensor is not None else torch.float32

    # 转换为张量
    x = torch.as_tensor(x, device=device, dtype=dtype)
    y = torch.as_tensor(y, device=device, dtype=dtype)

    # 归一化
    if normalize:
        x_min, x_max = domain_x
        y_min, y_max = domain_y
        # 防止除以零
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0
        x_norm = (x - x_min) / x_range
        y_norm = (y - y_min) / y_range
        result = {'x': x_norm, 'y': y_norm}
    else:
        result = {'x': x, 'y': y}

    # 添加其他键
    for k, v in extra_keys.items():
        result[k] = torch.as_tensor(v, device=device, dtype=dtype) if not isinstance(v, torch.Tensor) else v.to(device=device, dtype=dtype)

    return result