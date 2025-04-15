# PINN_Framework/src/optimizer_utils.py
"""
参数优化工具，用于从观测数据推断物理参数。
使用 PyTorch 优化器进行基于梯度的优化。
"""

import torch
import torch.optim as optim
import logging
from tqdm import tqdm
import time
import numpy as np
import torch.nn.functional as F
import os
from typing import Dict, Optional, Tuple, Any, Callable, Union

# 尝试从 .physics 导入必要的函数
try:
    # 确保导入的是新框架中的 physics 模块
    from .physics import calculate_laplacian
except ImportError:
    logging.error("无法从 .physics 导入 calculate_laplacian。请确保 physics.py 文件存在且路径正确。")
    def calculate_laplacian(*args, **kwargs):
        raise NotImplementedError("calculate_laplacian 未能从 .physics 导入")

# 尝试从 .models 导入模型类 (用于类型提示和测试)
try:
    # 确保导入的是新框架中的 models 模块
    from .models import AdaptiveFastscapePINN, TimeDerivativePINN
except ImportError:
    logging.warning("无法从 .models 导入模型类。类型提示和测试可能受影响。")
    # 定义占位符以允许模块加载
    class AdaptiveFastscapePINN: pass
    class TimeDerivativePINN: pass


# --- Differentiable Interpolation (Optional Utility) ---

def interpolate_params_torch(
    params_tensor: torch.Tensor,
    param_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
    method: str = 'bilinear',
    sigma: float = 0.1 # Only used for RBF
) -> torch.Tensor:
    """
    使用 PyTorch 进行参数场的可微插值。

    Args:
        params_tensor: 需要插值的参数张量 (可以是展平的或网格形状)。
        param_shape: 参数网格的原始形状 (H_param, W_param)。
        target_shape: 目标网格形状 (H_target, W_target)。
        method: 插值方法 ('bilinear' 或 'rbf')。
        sigma: RBF 插值的带宽。

    Returns:
        插值后的张量，形状为 target_shape。
    """
    device = params_tensor.device
    dtype = params_tensor.dtype # Use input tensor's dtype

    params_flat = params_tensor.flatten()
    param_h, param_w = param_shape
    target_h, target_w = target_shape

    if method == 'rbf':
        # RBF 插值逻辑 (与原文件相同)
        x_src = torch.linspace(0, 1, param_w, device=device, dtype=dtype)
        y_src = torch.linspace(0, 1, param_h, device=device, dtype=dtype)
        grid_y_src, grid_x_src = torch.meshgrid(y_src, x_src, indexing='ij')
        points_src = torch.stack([grid_x_src.flatten(), grid_y_src.flatten()], dim=1)

        x_tgt = torch.linspace(0, 1, target_w, device=device, dtype=dtype)
        y_tgt = torch.linspace(0, 1, target_h, device=device, dtype=dtype)
        grid_y_tgt, grid_x_tgt = torch.meshgrid(y_tgt, x_tgt, indexing='ij')
        points_tgt = torch.stack([grid_x_tgt.flatten(), grid_y_tgt.flatten()], dim=1)

        diff = points_tgt.unsqueeze(1) - points_src.unsqueeze(0)
        dist_sq = torch.sum(diff**2, dim=2)
        weights = torch.exp(-dist_sq / (2 * sigma**2))
        weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-10) # Normalize weights
        values_tgt = torch.matmul(weights, params_flat.unsqueeze(1)).squeeze(1)
        return values_tgt.reshape(target_shape) # Reshape to target grid

    elif method == 'bilinear':
        # 双线性插值逻辑 (与原文件相同)
        param_grid = params_flat.reshape(1, 1, param_h, param_w)
        x_tgt_gs = torch.linspace(-1, 1, target_w, device=device, dtype=dtype)
        y_tgt_gs = torch.linspace(-1, 1, target_h, device=device, dtype=dtype)
        grid_y_gs, grid_x_gs = torch.meshgrid(y_tgt_gs, x_tgt_gs, indexing='ij')
        grid_sample_coords = torch.stack([grid_x_gs, grid_y_gs], dim=2).unsqueeze(0)

        values_tgt_grid = F.grid_sample(
            param_grid, grid_sample_coords, mode='bilinear',
            padding_mode='border', align_corners=False # 通常 align_corners=False 更好
        )
        return values_tgt_grid.squeeze() # 返回 [H_target, W_target]
    else:
        raise ValueError(f"未知的插值方法: {method}")


# --- Parameter Optimizer Class ---

class ParameterOptimizer:
    """
    使用训练好的 PINN 模型和观测数据优化指定的物理参数。
    专为基于网格的优化设计，使用 PyTorch 优化器。
    """
    def __init__(self,
                 model: TimeDerivativePINN, # 接受 TimeDerivativePINN 兼容的模型
                 observation_data: torch.Tensor,
                 initial_state: Optional[torch.Tensor] = None,
                 fixed_params: Optional[Dict[str, Any]] = None,
                 t_target: Union[float, torch.Tensor] = 1.0):
        """
        初始化参数优化器。

        Args:
            model: 训练好的 PINN 模型 (例如 AdaptiveFastscapePINN)。
            observation_data: 观测到的最终地形 [B, 1, H, W]。
            initial_state: 初始地形 [B, 1, H, W]。如果为 None，则使用零初始化。
            fixed_params: 固定的物理参数字典 (例如 {'K': val})。值可以是标量或张量。
            t_target: 观测数据对应的时间点。
        """
        self.model = model
        self.observation = observation_data
        self.device = observation_data.device
        self.dtype = observation_data.dtype

        # 提取形状信息前检查维度
        if observation_data.ndim != 4:
             raise ValueError(f"Observation data must be 4D [B, C, H, W], but got shape {observation_data.shape}")
        self.batch_size, _, self.height, self.width = observation_data.shape
        logging.info(f"ParameterOptimizer 初始化，网格形状: B={self.batch_size}, H={self.height}, W={self.width}")

        self.model.to(self.device)
        self.model.eval() # 确保模型处于评估模式

        # 处理初始状态
        if initial_state is None:
            logging.info("ParameterOptimizer: 未提供初始状态，使用零初始化。")
            self.initial_state = torch.zeros_like(observation_data, device=self.device, dtype=self.dtype)
        else:
            self.initial_state = initial_state.to(device=self.device, dtype=self.dtype)

        # 存储固定参数，确保它们是张量
        self.fixed_params = {}
        if fixed_params:
            for k, v in fixed_params.items():
                 # 使用辅助函数确保参数是张量并具有兼容的形状
                 self.fixed_params[k] = self._ensure_tensor_param(v, k, "fixed_params")

        # 处理目标时间
        self.t_target = self._ensure_tensor_param(t_target, "t_target", "target time")

        logging.info(f"ParameterOptimizer 初始化完成")

    def _ensure_tensor_param(self, value: Any, name: str, context: str) -> torch.Tensor:
        """辅助函数，将标量/张量输入转换为设备和类型正确的张量。"""
        target_shape = (self.batch_size, 1, self.height, self.width) # 空间场的期望形状
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=self.device, dtype=self.dtype)
            # 允许从标量、批次标量或无批次的空间场进行广播
            if tensor.shape == target_shape: return tensor
            elif tensor.numel() == 1: return tensor.expand(target_shape)
            elif tensor.ndim == 1 and tensor.shape[0] == self.batch_size: return tensor.view(-1, 1, 1, 1).expand(target_shape)
            elif tensor.ndim == 2 and tensor.shape == (self.height, self.width): return tensor.unsqueeze(0).unsqueeze(0).expand(target_shape)
            elif tensor.ndim == 3 and tensor.shape == (self.batch_size, self.height, self.width): return tensor.unsqueeze(1)
            else: raise ValueError(f"无法将 {name} ({context}) 的形状 {tensor.shape} 广播到目标形状 {target_shape}")
        elif isinstance(value, (int, float)):
            # 如果是 t_target，保持标量张量
            if name == 't_target':
                 return torch.tensor(float(value), device=self.device, dtype=self.dtype)
            # 否则，假定是空间场，扩展到目标形状
            else:
                 return torch.full(target_shape, float(value), device=self.device, dtype=self.dtype)
        else:
            raise TypeError(f"{context} 中 {name} 的类型不受支持: {type(value)}")

    def _ensure_initial_param_shape(self, initial_value: Any, param_name: str) -> torch.Tensor:
        """确保初始优化参数张量具有正确的形状 [B, 1, H, W] 并需要梯度。"""
        target_shape = (self.batch_size, 1, self.height, self.width)
        if initial_value is None:
            logging.info(f"未提供 '{param_name}' 的初始值。使用 ones 初始化。")
            param_tensor = torch.ones(target_shape, device=self.device, dtype=self.dtype)
        elif isinstance(initial_value, (int, float)):
            param_tensor = torch.full(target_shape, float(initial_value), device=self.device, dtype=self.dtype)
        elif isinstance(initial_value, torch.Tensor):
            param_tensor = initial_value.clone().to(device=self.device, dtype=self.dtype)
            if param_tensor.shape != target_shape:
                logging.warning(f"'{param_name}' 的初始形状 {param_tensor.shape} 与目标 {target_shape} 不匹配。尝试调整。")
                # 尝试扩展或插值
                if param_tensor.numel() == 1:
                    param_tensor = param_tensor.expand(target_shape)
                elif param_tensor.ndim == 2 and param_tensor.shape == (self.height, self.width):
                    param_tensor = param_tensor.unsqueeze(0).unsqueeze(0).expand(target_shape)
                elif param_tensor.ndim == 3 and param_tensor.shape == (self.batch_size, self.height, self.width):
                     param_tensor = param_tensor.unsqueeze(1)
                else: # 尝试插值作为最后的手段
                    try:
                         if param_tensor.ndim == 2: param_tensor = param_tensor.unsqueeze(0).unsqueeze(0)
                         elif param_tensor.ndim == 3: param_tensor = param_tensor.unsqueeze(1)
                         # Ensure 4D input for interpolate
                         if param_tensor.ndim != 4: raise ValueError("Interpolation requires 4D input")
                         param_tensor = F.interpolate(param_tensor, size=(self.height, self.width), mode='bilinear', align_corners=False)
                         if param_tensor.shape[0] != self.batch_size: param_tensor = param_tensor.expand(self.batch_size, -1, -1, -1)
                         logging.info(f"已将 '{param_name}' 的初始值插值到 {param_tensor.shape}。")
                    except Exception as e:
                         raise ValueError(f"无法将 '{param_name}' 的初始形状 {initial_value.shape} 调整到目标 {target_shape}: {e}")
        else:
            raise TypeError(f"'{param_name}' 的 initial_value 类型不受支持: {type(initial_value)}")

        # 确保需要梯度
        if not param_tensor.requires_grad:
            param_tensor = param_tensor.detach().requires_grad_(True)
        return param_tensor

    def create_objective_function(self,
                                  params_to_optimize: Dict[str, torch.Tensor],
                                  spatial_smoothness_weight: float = 0.0,
                                  bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None
                                  ) -> Callable[[], Tuple[torch.Tensor, Dict[str, float]]]:
        """
        为 PyTorch 优化器创建优化目标函数。

        Args:
            params_to_optimize: 需要梯度的待优化参数字典 {'param_name': tensor}。
            spatial_smoothness_weight: 空间平滑度正则化权重 (拉普拉斯惩罚)。
            bounds: 参数边界字典 {'param_name': (min, max)}。None 表示无边界。

        Returns:
            可调用的目标函数，返回 (total_loss, loss_components_dict)。
        """
        param_names = list(params_to_optimize.keys())
        logging.info(f"创建优化目标函数，优化参数: {param_names}")
        if spatial_smoothness_weight > 0: logging.info(f"空间平滑度权重: {spatial_smoothness_weight}")
        if bounds: logging.info(f"应用边界: {bounds}")

        def objective_function() -> Tuple[torch.Tensor, Dict[str, float]]:
            # 应用边界约束（用于前向传播）
            current_params_constrained = {}
            if bounds:
                for name, param in params_to_optimize.items():
                    if name in bounds:
                        min_val, max_val = bounds[name]
                        # 使用 torch.clamp 处理 None 边界
                        current_params_constrained[name] = torch.clamp(param, min=min_val, max=max_val)
                    else:
                        current_params_constrained[name] = param
            else:
                current_params_constrained = params_to_optimize

            # 合并固定参数和当前优化的参数
            all_params = {**self.fixed_params, **current_params_constrained}

            # 准备模型输入
            model_input = {
                'initial_state': self.initial_state,
                'params': all_params,
                't_target': self.t_target
            }

            # 使用 PINN 模型预测最终状态
            predicted_state = self.model(model_input, mode='predict_state')
            # 如果模型返回字典，提取 'state'
            if isinstance(predicted_state, dict):
                 if 'state' in predicted_state: predicted_state = predicted_state['state']
                 else: raise ValueError("模型在 'predict_state' 模式下未返回 'state' 键。")

            # 计算数据保真度损失 (MSE)
            data_loss = F.mse_loss(predicted_state, self.observation)
            loss_components = {'data_loss': data_loss.item()}
            total_loss = data_loss

            # 添加空间平滑度正则化 (拉普拉斯惩罚)
            if spatial_smoothness_weight > 0:
                smoothness_loss_total = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                for name, param in params_to_optimize.items(): # 使用原始参数计算梯度
                    # 假设 dx=1, dy=1 进行惩罚，直接惩罚曲率
                    laplacian = calculate_laplacian(param, dx=1.0, dy=1.0)
                    # 惩罚拉普拉斯算子的平方范数
                    smoothness_loss = torch.mean(laplacian**2) * spatial_smoothness_weight
                    smoothness_loss_total = smoothness_loss_total + smoothness_loss
                    loss_components[f'{name}_smoothness_loss'] = smoothness_loss.item()
                total_loss = total_loss + smoothness_loss_total
                loss_components['smoothness_loss'] = smoothness_loss_total.item() # 记录总平滑度损失

            loss_components['total_loss'] = total_loss.item()
            return total_loss, loss_components

        return objective_function

# --- PyTorch-based Optimization Function ---

def optimize_parameters(
    model: TimeDerivativePINN,
    observation_data: torch.Tensor,
    params_to_optimize_config: Dict[str, Dict],
    config: Dict,
    initial_state: Optional[torch.Tensor] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
    t_target: Union[float, torch.Tensor] = 1.0
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """
    使用 PyTorch 优化器运行参数优化流程。

    Args:
        model: 训练好的 PINN 模型。
        observation_data: 观测地形数据 [B, 1, H, W]。
        params_to_optimize_config: 配置要优化的参数
                                   {'param_name': {'initial_value': val, 'bounds': (min, max)}, ...}。
        config: 主配置文件，包含 'optimization_params'。
        initial_state: 初始地形状态 [B, 1, H, W]。
        fixed_params: 固定的物理参数字典。
        t_target: 观测数据对应的时间点。

    Returns:
        Tuple: (optimized_params_dict, history)
               - optimized_params_dict: 包含优化后参数张量的字典 (detached)。
               - history: 包含优化历史（损失、迭代次数、时间）的字典。
    """
    opt_config = config.get('optimization_params', {})
    optimizer_name = opt_config.get('optimizer', 'Adam').lower()
    lr = opt_config.get('learning_rate', 0.01)
    iterations = opt_config.get('max_iterations', 1000)
    spatial_smoothness = opt_config.get('spatial_smoothness_weight', 0.0)
    log_interval = opt_config.get('log_interval', 50)
    convergence_patience = opt_config.get('convergence_patience', 20)
    loss_tolerance = opt_config.get('loss_tolerance', 1e-7)

    # --- 初始化 ParameterOptimizer ---
    param_optimizer_instance = ParameterOptimizer(model, observation_data, initial_state, fixed_params, t_target)
    device = param_optimizer_instance.device

    # --- 初始化待优化参数 ---
    params_to_optimize = {}
    param_bounds = {}
    for name, p_config in params_to_optimize_config.items():
        initial_value = p_config.get('initial_value')
        bounds = p_config.get('bounds') # e.g., (0.0, 0.005) or (None, 0.005) or (0.0, None)
        params_to_optimize[name] = param_optimizer_instance._ensure_initial_param_shape(initial_value, name)
        if bounds: param_bounds[name] = bounds

    # --- 创建优化目标函数 ---
    objective_fn = param_optimizer_instance.create_objective_function(
        params_to_optimize, spatial_smoothness, param_bounds
    )

    # --- 创建 PyTorch 优化器 ---
    params_list = list(params_to_optimize.values())
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(params_list, lr=lr, betas=opt_config.get('betas', (0.9, 0.999)))
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(params_list, lr=lr, weight_decay=opt_config.get('weight_decay', 0.01))
    elif optimizer_name == 'lbfgs':
        optimizer = torch.optim.LBFGS(params_list, lr=lr, max_iter=opt_config.get('lbfgs_max_iter', 20), line_search_fn="strong_wolfe")
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_name}")

    logging.info(f"开始优化: {optimizer_name}, 迭代次数: {iterations}, 学习率: {lr}")
    history = {'loss': [], 'iterations': 0, 'time': 0.0, 'loss_components': []}
    start_time = time.time()

    # --- 优化循环 ---
    for i in tqdm(range(iterations), desc="优化参数"):
        # 定义闭包（LBFGS 需要）
        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss, loss_components = objective_fn()
            if torch.is_tensor(loss) and loss.requires_grad:
                loss.backward()
            elif not torch.is_tensor(loss):
                 logging.error("目标函数未返回张量损失。")
                 return torch.tensor(float('nan'), device=device) # Signal error
            # 存储损失组件以供外部访问
            closure.loss_components = loss_components
            return loss

        # --- Optimizer Step ---
        if optimizer_name == 'lbfgs':
            loss = optimizer.step(closure)
            loss_components = getattr(closure, 'loss_components', {}) # 获取最后一次闭包调用的组件
        else:
            optimizer.zero_grad()
            loss, loss_components = objective_fn() # 计算损失和梯度
            if torch.is_tensor(loss) and torch.isfinite(loss) and loss.requires_grad:
                loss.backward()
                optimizer.step()
            elif not torch.is_tensor(loss) or not torch.isfinite(loss):
                 logging.warning(f"迭代 {i}: 损失无效 ({loss})。跳过优化步骤。")
                 if not torch.isfinite(loss):
                      logging.error("优化因损失无效而停止。")
                      break # 停止优化

        # --- 应用边界约束 (优化步骤之后) ---
        if param_bounds:
            with torch.no_grad():
                for name, param in params_to_optimize.items():
                    if name in param_bounds:
                        min_val, max_val = param_bounds[name]
                        # clamp_ 需要 inplace 操作
                        param.clamp_(min=min_val, max=max_val)

        # --- 记录历史 ---
        current_loss = loss.item() if torch.is_tensor(loss) else float('nan')
        history['loss'].append(current_loss)
        history['loss_components'].append(loss_components) # 记录当前步骤的损失组件

        if i % log_interval == 0:
            log_str = ", ".join([f"{k}: {v:.3e}" for k, v in loss_components.items()])
            logging.info(f"Iter {i}/{iterations}, Loss: {current_loss:.6e} ({log_str})")

        # --- 收敛检查 (Adam/AdamW) ---
        if optimizer_name != 'lbfgs':
            if i > convergence_patience:
                 loss_hist = history['loss'][-convergence_patience:]
                 # 检查历史记录是否足够长且包含有效值
                 valid_hist = [l for l in loss_hist if np.isfinite(l)]
                 if len(valid_hist) > convergence_patience // 2 : # 需要足够多的有效点
                      if len(valid_hist) > 1 and np.abs(valid_hist[-1] - np.mean(valid_hist[:-1])) < loss_tolerance:
                           logging.info(f"在迭代 {i} 时因损失变化小而收敛。")
                           break
            if not np.isfinite(current_loss):
                 logging.error(f"在迭代 {i} 时损失无效。停止优化。")
                 break

    # --- 结束 ---
    end_time = time.time()
    total_time = end_time - start_time
    history['iterations'] = len(history['loss'])
    history['time'] = total_time
    history['final_loss'] = history['loss'][-1] if history['loss'] and np.isfinite(history['loss'][-1]) else float('nan')

    logging.info(f"优化在 {total_time:.2f} 秒内完成 {history['iterations']} 次迭代。")
    logging.info(f"最终损失: {history['final_loss']:.6e}")

    # Detach final parameters
    optimized_detached = {k: v.detach().clone() for k, v in params_to_optimize.items()}

    # --- 保存结果 ---
    save_path = opt_config.get('save_path')
    if save_path:
        try:
            save_dir = os.path.dirname(save_path)
            if save_dir: os.makedirs(save_dir, exist_ok=True)
            # 准备要保存的历史记录（移除张量）
            history_to_save = history.copy()
            # 确保 loss_components 列表中的字典不包含张量
            history_to_save['loss_components'] = [{k: v if isinstance(v, (int, float)) else str(v) for k, v in lc.items()} for lc in history['loss_components']]

            save_dict = {
                'optimized_params': {k: v.cpu() for k, v in optimized_detached.items()}, # 保存到 CPU
                'history': history_to_save,
                'config': config # 保存使用的完整配置
            }
            torch.save(save_dict, save_path)
            logging.info(f"优化结果已保存到: {save_path}")
        except Exception as e:
            logging.error(f"保存优化结果失败: {e}", exc_info=True)

    return optimized_detached, history


# --- Main test block ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    print("Testing PyTorch-based Parameter Optimizer...")

    # --- Dummy Setup ---
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")

        # Dummy Model (Needs predict_state and TimeDerivativePINN compatibility)
        class OptimizerTestDummyModel(TimeDerivativePINN): # Inherit for compatibility check
            def __init__(self, H=16, W=16):
                super().__init__() # Call base class init
                self.H, self.W = H, W
                # 增加复杂度以更好地测试优化
                self.layer1 = torch.nn.Conv2d(1 + 1, 8, kernel_size=3, padding=1)
                self.act1 = torch.nn.ReLU()
                self.layer2 = torch.nn.Conv2d(8, 1, kernel_size=3, padding=1)

            def forward(self, x, mode):
                if mode == 'predict_state':
                    initial_topo = x['initial_state']
                    params = x['params']
                    t_target = x['t_target']
                    # 使用 U，如果不存在则默认为零
                    U_grid = params.get('U', torch.zeros_like(initial_topo))

                    # 确保 t_target 可以广播
                    if isinstance(t_target, torch.Tensor) and t_target.ndim == 0:
                         t_target = t_target.view(1, 1, 1, 1) # Expand scalar time
                    elif isinstance(t_target, torch.Tensor) and t_target.ndim == 1:
                         t_target = t_target.view(-1, 1, 1, 1) # Expand batch time

                    # 模拟更复杂的模型响应
                    combined_input = torch.cat([initial_topo, U_grid], dim=1)
                    features = self.act1(self.layer1(combined_input))
                    processed = self.layer2(features)
                    # 模拟时间演化和参数影响
                    pred_state = initial_topo * torch.exp(-t_target * 0.001) + processed * torch.tanh(t_target * 0.1) + U_grid * t_target * 0.5

                    # 模拟导数输出（例如，与 U 和状态相关）
                    pred_deriv = U_grid * 0.5 - initial_topo * 0.001 * torch.exp(-t_target * 0.001) + processed * 0.01 / (torch.cosh(t_target * 0.1)**2)


                    # 根据输出模式返回
                    output = {}
                    if self.output_state: output['state'] = pred_state
                    if self.output_derivative: output['derivative'] = pred_deriv
                    if not output: raise ValueError("模型未配置输出任何内容")
                    if len(output) == 1: return next(iter(output.values()))
                    return output

                elif mode == 'predict_coords':
                     # 为 predict_coords 提供虚拟实现
                     coords = x
                     output = {}
                     if self.output_state: output['state'] = torch.zeros_like(coords['x'])
                     if self.output_derivative: output['derivative'] = torch.zeros_like(coords['x'])
                     if not output: raise ValueError("模型未配置输出任何内容")
                     if len(output) == 1: return next(iter(output.values()))
                     return output
                return None # 不支持的模式

        dummy_model = OptimizerTestDummyModel(H=16, W=16).to(device)

        # Dummy Target Data
        H, W = 16, 16
        # 创建一个更结构化的真实抬升场
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=device), torch.linspace(-1, 1, W, device=device), indexing='ij')
        true_uplift_torch = (0.001 * (1 + torch.sin(np.pi * xx) * torch.cos(np.pi * yy))).unsqueeze(0).unsqueeze(0)

        initial_topo_torch = torch.rand(1, 1, H, W, device=device) * 10
        t_target_val = 1000.0
        # 固定参数 K 和 D
        fixed_params_test = {'K': torch.tensor(1e-5, device=device), 'D': torch.tensor(0.01, device=device)}
        true_params_for_target = {'U': true_uplift_torch, **fixed_params_test}

        with torch.no_grad():
             dummy_model.set_output_mode(state=True, derivative=False) # 仅需要状态来生成目标
             dummy_target = dummy_model(x={'initial_state': initial_topo_torch, 'params': true_params_for_target, 't_target': t_target_val}, mode='predict_state')
        logging.info(f"生成虚拟目标数据，形状: {dummy_target.shape}")

        # Parameters to Optimize Config
        params_to_opt_config = {
            'U': {
                'initial_value': torch.zeros_like(true_uplift_torch) + true_uplift_torch.mean()*0.8, # 初始猜测接近均值
                'bounds': (0.0, 0.005) # 抬升率边界
            }
        }

        # Main Config for Optimization
        dummy_main_config = {
            'optimization_params': {
                'optimizer': 'AdamW', 'learning_rate': 1e-3, 'max_iterations': 300, # 增加迭代次数
                'spatial_smoothness_weight': 5e-2, 'log_interval': 25, 'weight_decay': 1e-3,
                'convergence_patience': 30, 'loss_tolerance': 1e-8, # 更严格的容忍度
                'save_path': 'results/dummy_optimize_test/optimized_params.pth'
            },
            'physics': { 'dx': 1.0, 'dy': 1.0 } # 用于平滑度损失
        }

        # --- Run Optimization ---
        print("\n运行虚拟优化 (PyTorch)...")
        dummy_model.eval() # 确保模型处于评估模式
        optimized_params, history = optimize_parameters(
            model=dummy_model,
            observation_data=dummy_target,
            params_to_optimize_config=params_to_opt_config,
            config=dummy_main_config,
            initial_state=initial_topo_torch,
            fixed_params=fixed_params_test, # 传递固定的 K, D
            t_target=t_target_val
        )
        print("优化完成。")

        # --- Analyze Results ---
        if optimized_params and 'U' in optimized_params:
            optimized_U = optimized_params['U'].cpu()
            initial_U = params_to_opt_config['U']['initial_value'].cpu()
            true_U_cpu = true_uplift_torch.cpu()
            print(f"初始 U 均值: {initial_U.mean().item():.6f}")
            print(f"优化后 U 均值: {optimized_U.mean().item():.6f} (真实均值: {true_U_cpu.mean().item():.6f})")
            print(f"最终损失: {history['final_loss']:.6e}")
            # 计算优化后的 U 与真实 U 的 MSE
            mse_u = torch.mean((optimized_U - true_U_cpu)**2).item()
            print(f"优化后 U 与真实 U 的 MSE: {mse_u:.6e}")

            # 可选：可视化比较
            try:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                vmin = min(initial_U.min(), optimized_U.min(), true_U_cpu.min())
                vmax = max(initial_U.max(), optimized_U.max(), true_U_cpu.max())
                im0 = axes[0].imshow(initial_U.squeeze().numpy(), vmin=vmin, vmax=vmax, cmap='viridis')
                axes[0].set_title("初始 U")
                plt.colorbar(im0, ax=axes[0])
                im1 = axes[1].imshow(optimized_U.squeeze().numpy(), vmin=vmin, vmax=vmax, cmap='viridis')
                axes[1].set_title("优化后 U")
                plt.colorbar(im1, ax=axes[1])
                im2 = axes[2].imshow(true_U_cpu.squeeze().numpy(), vmin=vmin, vmax=vmax, cmap='viridis')
                axes[2].set_title("真实 U")
                plt.colorbar(im2, ax=axes[2])
                plt.tight_layout()
                # 保存图像而不是显示
                save_fig_path = 'results/dummy_optimize_test/uplift_comparison.png'
                os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
                plt.savefig(save_fig_path)
                print(f"比较图像已保存到: {save_fig_path}")
                plt.close(fig) # 关闭图像以防显示
            except ImportError:
                print("未安装 Matplotlib，跳过可视化。")

        else:
            print("优化未返回预期的参数 'U'。")

    except ImportError as e:
         print(f"因缺少依赖项而跳过优化器测试: {e}")
    except Exception as e:
        print(f"优化器测试期间出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n优化器工具测试完成。")