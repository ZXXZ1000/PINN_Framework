# PINN_Framework/src/losses.py
"""
损失函数模块，包含数据损失、物理残差损失（双输出）和平滑度惩罚。
"""

import torch
import torch.nn.functional as F
import logging
from typing import Dict, Optional, Tuple

# 尝试从 .physics 导入必要的函数
try:
    from .physics import calculate_dhdt_physics, calculate_slope_magnitude
except ImportError:
    logging.error("无法从 .physics 导入必要的函数 (calculate_dhdt_physics, calculate_slope_magnitude)。请确保 physics.py 文件存在且路径正确。")
    # 定义占位符函数以允许模块加载，但在运行时会失败
    def calculate_dhdt_physics(*args, **kwargs):
        raise NotImplementedError("calculate_dhdt_physics 未能从 .physics 导入")
    def calculate_slope_magnitude(*args, **kwargs):
        raise NotImplementedError("calculate_slope_magnitude 未能从 .physics 导入")


# --- Data Fidelity Loss ---

def compute_data_loss(predicted_topo: torch.Tensor, target_topo: torch.Tensor) -> torch.Tensor:
    """
    计算预测地形和目标地形之间的数据保真度损失 (MSE)。
    假设 predicted_topo 和 target_topo 对应相同的时间实例，
    并且具有相同的网格形状 [B, C, H, W]。
    """
    if predicted_topo.shape != target_topo.shape:
         logging.warning(f"compute_data_loss 中的形状不匹配: pred={predicted_topo.shape}, target={target_topo.shape}。正在调整目标大小。")
         # 调整目标大小以匹配预测
         target_topo = F.interpolate(target_topo.float(), size=predicted_topo.shape[-2:], mode='bilinear', align_corners=False)

    # 确保 target_topo 也是 float 类型以用于 mse_loss
    return F.mse_loss(predicted_topo, target_topo.float())

# --- PDE Residual Calculation (Dual Output Model) ---

def compute_pde_residual_dual_output(outputs: Dict[str, torch.Tensor], physics_params: Dict) -> torch.Tensor:
    """
    使用模型直接输出的状态和导数计算PDE残差。
    Residual = predicted_dh_dt - physics_dh_dt
             = predicted_dh_dt - (U - K_f*A^m*S^n + K_d*Laplacian(h))

    Args:
        outputs (dict): 模型输出字典，必须包含 'state' 和 'derivative' 张量。
                        'state' shape: [B, C_out, H, W]
                        'derivative' shape: [B, C_out, H, W]
        physics_params (dict): 物理参数字典 (U, K_f, m, n, K_d, dx, dy, precip, da_params)。

    Returns:
        torch.Tensor: PDE残差均方误差。
    """
    if 'state' not in outputs or 'derivative' not in outputs:
        raise ValueError("模型输出字典必须包含 'state' 和 'derivative' 键。")

    h_pred = outputs['state']
    dh_dt_pred = outputs['derivative']

    if h_pred.shape != dh_dt_pred.shape:
        raise ValueError(f"状态和导数预测的形状不匹配: state={h_pred.shape}, derivative={dh_dt_pred.shape}")

    # 假设输入总是网格模式 (B, C, H, W)
    if h_pred.ndim != 4:
         logging.warning(f"compute_pde_residual_dual_output 期望 4D 网格输入，但得到形状 {h_pred.shape}。结果可能不可靠。")
         # 尝试继续，但可能在物理计算中失败

    try:
        # 提取必要的物理参数
        U = physics_params.get('U', 0.0)
        K_f = physics_params.get('K_f', 1e-5)
        m = physics_params.get('m', 0.5)
        n = physics_params.get('n', 1.0)
        K_d = physics_params.get('K_d', 0.01)
        dx = physics_params.get('dx', 1.0)
        dy = physics_params.get('dy', 1.0)
        precip = physics_params.get('precip', 1.0)
        da_params = physics_params.get('da_params', {}) # 从 physics_params 获取 da_params

        # --- 计算物理倾向项 dh/dt_physics ---
        # (复用 prepare_param 逻辑，或者直接在 calculate_dhdt_physics 中处理)
        def prepare_param(param_val, target_shape, device, dtype):
             if isinstance(param_val, torch.Tensor):
                 param_val = param_val.to(device=device, dtype=dtype)
                 if param_val.shape == target_shape: return param_val
                 elif param_val.numel() == 1: return param_val.expand(target_shape)
                 elif param_val.ndim == 1 and param_val.shape[0] == target_shape[0]: return param_val.view(-1, 1, 1, 1).expand(target_shape)
                 else:
                     try: return param_val.expand(target_shape)
                     except RuntimeError: raise ValueError(f"无法广播参数形状 {param_val.shape} 到目标 {target_shape}")
             else: return torch.full(target_shape, float(param_val), device=device, dtype=dtype)

        device = h_pred.device
        dtype = h_pred.dtype
        target_shape = h_pred.shape
        U_grid = prepare_param(U, target_shape, device, dtype)
        K_f_val = prepare_param(K_f, target_shape, device, dtype) # 处理可能的空间 K_f
        K_d_val = prepare_param(K_d, target_shape, device, dtype) # 处理可能的空间 K_d

        dhdt_physics = calculate_dhdt_physics(
            h=h_pred,
            U=U_grid,
            K_f=K_f_val, # 使用处理后的 K_f
            m=float(m),  # 确保 m, n 是浮点数
            n=float(n),
            K_d=K_d_val, # 使用处理后的 K_d
            dx=dx,
            dy=dy,
            precip=precip,
            da_params=da_params # 传递汇水面积参数
        )

        # --- 计算残差 ---
        pde_residual = dh_dt_pred - dhdt_physics

        # 返回均方误差损失
        return F.mse_loss(pde_residual, torch.zeros_like(pde_residual))

    except Exception as e:
        logging.error(f"计算双输出 PDE 残差时出错: {e}", exc_info=True)
        # 返回一个零损失，但保留梯度连接（如果可能）
        zero_loss = (outputs['state'].sum() + outputs['derivative'].sum()) * 0.0
        return zero_loss

# --- Smoothness Penalty ---

def compute_smoothness_penalty(predicted_topo: torch.Tensor, dx: float = 1.0, dy: float = 1.0) -> torch.Tensor:
    """
    计算基于预测地形梯度幅度的平滑度惩罚。
    假设 predicted_topo 是网格形式 (B, C, H, W)。
    """
    if predicted_topo.ndim != 4 or predicted_topo.shape[1] != 1:
         logging.warning(f"Smoothness penalty 期望输入形状 (B, 1, H, W)，但得到 {predicted_topo.shape}。跳过惩罚。")
         return predicted_topo.sum() * 0.0 # 返回带梯度的零张量

    try:
        # 使用 physics 模块中的函数计算坡度幅度
        slope_mag = calculate_slope_magnitude(predicted_topo, dx, dy)
        # 惩罚平均坡度幅度
        smoothness_loss = torch.mean(slope_mag)
        return smoothness_loss
    except Exception as e:
        logging.error(f"计算平滑度惩罚时出错: {e}", exc_info=True)
        return predicted_topo.sum() * 0.0 # 返回带梯度的零张量

# --- Total Loss Calculation (Simplified) ---

def compute_total_loss(
    data_pred: Optional[torch.Tensor],
    target_topo: Optional[torch.Tensor],
    physics_loss_value: Optional[torch.Tensor],
    smoothness_pred: Optional[torch.Tensor], # Prediction used for smoothness (can be same as data_pred)
    physics_params: Dict, # Still needed for dx, dy for smoothness
    loss_weights: Dict[str, float]
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算总加权损失。

    Args:
        data_pred: 用于数据损失的模型预测 (例如，最终状态网格)。
        target_topo: 数据损失的目标数据。
        physics_loss_value: 预先计算的物理损失 (PDE 残差)。
        smoothness_pred: 用于平滑度损失的模型预测。
        physics_params: 包含物理参数的字典 (例如，dx, dy)。
        loss_weights: 映射损失分量名称 ('data', 'physics', 'smoothness') 到权重的字典。

    Returns:
        tuple: (total_loss, weighted_losses_dict)
    """
    loss_components = {}
    weighted_losses = {}
    total_loss = None

    # 确定设备和参考张量以保持梯度连接
    ref_tensor_for_grad = None
    if data_pred is not None: ref_tensor_for_grad = data_pred
    elif target_topo is not None: ref_tensor_for_grad = target_topo
    elif physics_loss_value is not None: ref_tensor_for_grad = physics_loss_value
    elif smoothness_pred is not None: ref_tensor_for_grad = smoothness_pred

    if ref_tensor_for_grad is not None:
        device = ref_tensor_for_grad.device
    else:
        device = torch.device('cpu') # Fallback

    # 辅助函数创建带梯度的零张量
    def _zero_with_grad(ref_tensor):
        if ref_tensor is not None and isinstance(ref_tensor, torch.Tensor) and ref_tensor.requires_grad:
            return (ref_tensor.sum() * 0.0).to(device)
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)

    # 1. 数据拟合损失
    data_weight = loss_weights.get('data', 0.0)
    if data_weight > 0 and data_pred is not None and target_topo is not None:
        try:
            loss_components['data'] = compute_data_loss(data_pred, target_topo)
        except Exception as e:
             logging.error(f"计算数据损失时出错: {e}", exc_info=True)
             loss_components['data'] = _zero_with_grad(ref_tensor_for_grad)
    else:
        loss_components['data'] = _zero_with_grad(ref_tensor_for_grad)

    # 2. 物理残差损失 (使用预计算的值)
    physics_weight = loss_weights.get('physics', 0.0)
    if physics_weight > 0 and physics_loss_value is not None and isinstance(physics_loss_value, torch.Tensor) and torch.isfinite(physics_loss_value):
        loss_components['physics'] = physics_loss_value
    else:
        loss_components['physics'] = _zero_with_grad(ref_tensor_for_grad)
        if physics_weight > 0 and (physics_loss_value is None or not isinstance(physics_loss_value, torch.Tensor) or not torch.isfinite(physics_loss_value)):
             logging.warning(f"收到无效或非有限的 physics_loss_value ({physics_loss_value})。将物理损失分量设为零。")

    # 3. 平滑正则化损失
    smoothness_weight = loss_weights.get('smoothness', 0.0)
    if smoothness_weight > 0 and smoothness_pred is not None:
         try:
            dx = physics_params.get('dx', 1.0)
            dy = physics_params.get('dy', 1.0)
            loss_components['smoothness'] = compute_smoothness_penalty(smoothness_pred, dx, dy)
         except Exception as e:
             logging.error(f"计算平滑度惩罚时出错: {e}", exc_info=True)
             loss_components['smoothness'] = _zero_with_grad(ref_tensor_for_grad)
    else:
        loss_components['smoothness'] = _zero_with_grad(ref_tensor_for_grad)

    # 累加加权损失
    for name, value in loss_components.items():
        weight = loss_weights.get(name, 0.0)
        if isinstance(value, torch.Tensor) and weight > 0 and torch.isfinite(value):
            weighted_value = weight * value
            if total_loss is None:
                total_loss = weighted_value
            else:
                total_loss = total_loss + weighted_value # 确保张量相加
            weighted_losses[f"{name}_loss"] = weighted_value.item()
        elif isinstance(value, torch.Tensor) and not torch.isfinite(value):
             logging.warning(f"损失分量 '{name}' 遇到非有限值。跳过。")
             weighted_losses[f"{name}_loss"] = float('nan')
        else:
             weighted_losses[f"{name}_loss"] = 0.0 # 权重为零或值无效

    # 如果没有有效损失项，使用零梯度张量
    if total_loss is None:
        total_loss = _zero_with_grad(ref_tensor_for_grad)

    # 最终检查 total_loss
    if not isinstance(total_loss, torch.Tensor) or not torch.isfinite(total_loss):
        logging.error(f"总损失在累加过程中变为非有限或无效: {total_loss}。各加权损失: {weighted_losses}")
        total_loss = _zero_with_grad(ref_tensor_for_grad) # 使用带梯度的零张量作为后备

    weighted_losses['total_loss'] = total_loss.item() if isinstance(total_loss, torch.Tensor) and torch.isfinite(total_loss) else float('nan')

    return total_loss, weighted_losses
