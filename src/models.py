# PINN_Framework/src/models.py
import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
import math # For sqrt(2) in AdaptiveFastscapePINN tiling (if needed)
from .utils import standardize_coordinate_system, prepare_parameter # Assuming these will be in utils.py

# --- Base Class for Dual Output ---
class TimeDerivativePINN(nn.Module):
    """能同时输出状态及其时间导数的PINN基类"""

    def __init__(self):
        super().__init__()
        self.output_state = True
        self.output_derivative = True
        # Add output mode tracker for debugging
        self._mode_changes = []

    def get_output_mode(self):
        """获取当前输出模式"""
        modes = []
        if self.output_state:
            modes.append('state')
        if self.output_derivative:
            modes.append('derivative')
        return modes

    def set_output_mode(self, state=True, derivative=True):
        """设置输出模式（状态和/或导数）

        Args:
            state (bool): 是否输出状态
            derivative (bool): 是否输出时间导数

        Raises:
            ValueError: 如果state和derivative均为False
        """
        if not state and not derivative:
            raise ValueError("至少需要一个输出模式为True（state或derivative）")

        # Track mode changes for debugging
        old_modes = self.get_output_mode()
        self.output_state = state
        self.output_derivative = derivative
        new_modes = self.get_output_mode()

        # Log mode changes
        if old_modes != new_modes:
            self._mode_changes.append((old_modes, new_modes))
            logging.debug(f"TimeDerivativePINN output mode changed: {old_modes} -> {new_modes}")

    def check_output_format(self, outputs, required_outputs=None):
        """检查输出格式是否符合预期 (Optional: Can be simplified or removed if not strictly needed)"""
        # Implementation from original code can be kept or simplified
        if required_outputs is None:
            # 只检查模式配置和输出类型匹配
            if isinstance(outputs, dict):
                if self.output_state and 'state' not in outputs:
                    return False
                if self.output_derivative and 'derivative' not in outputs:
                    return False
                return True
            else: # Single tensor output
                return self.output_state and not self.output_derivative
        else:
            # 检查是否有所有需要的输出
            if isinstance(outputs, dict):
                return all(output_type in outputs for output_type in required_outputs)
            else: # Single tensor output
                return len(required_outputs) == 1 and required_outputs[0] == 'state'


    def forward(self, *args, **kwargs):
        """前向传播，需要在子类中实现"""
        raise NotImplementedError("子类必须实现forward方法")

    def predict_derivative_fd(self, x, delta_t=1e-3, mode='predict_coords'):
        """使用有限差分近似计算时间导数（用于测试）"""
        # Implementation from original code can be kept for testing purposes
        original_state = self.output_state
        original_derivative = self.output_derivative
        self.set_output_mode(state=True, derivative=False)

        try:
            if mode == 'predict_coords':
                if not isinstance(x, dict) or 't' not in x:
                    raise ValueError("predict_derivative_fd在'predict_coords'模式下需要't'键")
                t = x['t']
                x_forward = {**x, 't': t + delta_t / 2}
                x_backward = {**x, 't': t - delta_t / 2}
            elif mode == 'predict_state':
                 if not isinstance(x, dict) or 't_target' not in x:
                     raise ValueError("predict_derivative_fd在'predict_state'模式下需要't_target'键")
                 t_target = x['t_target']
                 x_forward = {**x, 't_target': t_target + delta_t / 2}
                 x_backward = {**x, 't_target': t_target - delta_t / 2}
            else:
                raise ValueError(f"predict_derivative_fd不支持的模式: {mode}")

            with torch.no_grad():
                pred_forward = self.forward(x_forward, mode=mode)
                pred_backward = self.forward(x_backward, mode=mode)

            if isinstance(pred_forward, dict): pred_forward = pred_forward['state']
            if isinstance(pred_backward, dict): pred_backward = pred_backward['state']

            derivative_fd = (pred_forward - pred_backward) / delta_t

        finally:
            # Ensure original mode is restored even if errors occur
            self.set_output_mode(state=original_state, derivative=original_derivative)

        return derivative_fd


# --- AdaptiveFastscapePINN (Dual Output, Refactored) ---
class AdaptiveFastscapePINN(TimeDerivativePINN):
    """
    支持任意尺寸参数矩阵和多分辨率处理的物理信息神经网络 (主线模型)。
    同时输出状态 (地形 h) 和时间导数 (dh/dt)。
    内部整合了坐标处理 MLP 的逻辑。
    """
    def __init__(self,
                 output_dim=1,
                 hidden_dim=256,
                 num_layers=8, # Total layers for coordinate MLP part
                 base_resolution=64,
                 max_resolution=1024,
                 activation_fn=nn.Tanh, # Use activation function class
                 coordinate_input_dim=5, # (x, y, t, k, u)
                 domain_x: list = [0.0, 1.0],
                 domain_y: list = [0.0, 1.0]):
        super().__init__()
        self.output_dim = output_dim
        self.base_resolution = base_resolution
        self.max_resolution = max_resolution
        self.coordinate_input_dim = coordinate_input_dim

        # Initialize physics parameters with defaults
        self.physics_params = {
            'dx': 1.0,
            'dy': 1.0,
            'precip': 1.0,
            'da_params': {
                'omega': 0.3,            # Reduced relaxation factor for better stability
                'solver_max_iters': 5000, # Increased max iterations
                'solver_tol': 1e-5,      # Adjusted tolerance
                'eps': 1e-10,            # Small epsilon for numerical stability
                'verbose': False,         # Verbose output for debugging
                'stabilize': True,        # Enable stabilization
                'use_fallback': True      # Enable fallback to simpler method if convergence fails
            }
        }

        # Store domain boundaries
        # Handle OmegaConf ListConfig objects and other sequence types
        # First convert to Python lists if possible
        try:
            # Check if it's a sequence type (list, tuple, or OmegaConf ListConfig)
            from collections.abc import Sequence # Use abc for broader type check
            from omegaconf import ListConfig # Import ListConfig for specific check
            if not isinstance(domain_x, Sequence) or isinstance(domain_x, str) or len(domain_x) != 2:
                 raise TypeError("domain_x must be a sequence of length 2")
            if not isinstance(domain_y, Sequence) or isinstance(domain_y, str) or len(domain_y) != 2:
                 raise TypeError("domain_y must be a sequence of length 2")

            # Try to access items and convert to float
            domain_x_min = float(domain_x[0])
            domain_x_max = float(domain_x[1])
            domain_y_min = float(domain_y[0])
            domain_y_max = float(domain_y[1])

            # Create new Python lists
            domain_x = [domain_x_min, domain_x_max]
            domain_y = [domain_y_min, domain_y_max]
        except (IndexError, TypeError, ValueError) as e:
            raise ValueError(f"domain_x and domain_y must be sequence-like objects with 2 float-convertible elements. Error: {e}")

        # Check values are valid (min < max)
        if domain_x[0] >= domain_x[1] or domain_y[0] >= domain_y[1]:
            raise ValueError(f"Domain boundaries must have min < max. Got domain_x={domain_x}, domain_y={domain_y}")

        self.domain_x = domain_x
        self.domain_y = domain_y
        self.epsilon = 1e-9 # For safe division during normalization

        # --- Coordinate MLP Feature Extractor (Integrated) ---
        activation = activation_fn() # Instantiate activation function
        coord_layers = []
        coord_layers.append(nn.Linear(coordinate_input_dim, hidden_dim))
        coord_layers.append(activation)
        for _ in range(num_layers - 2): # num_layers includes input and output, feature layers are num_layers-1
            coord_layers.append(nn.Linear(hidden_dim, hidden_dim))
            coord_layers.append(activation)
        # The last layer outputs features of size hidden_dim
        coord_layers.append(nn.Linear(hidden_dim, hidden_dim))
        coord_layers.append(activation) # Activation after the last feature layer
        self.coordinate_feature_extractor = nn.Sequential(*coord_layers)

        # --- Output Heads ---
        self.state_head = nn.Linear(hidden_dim, output_dim)
        self.derivative_head = nn.Linear(hidden_dim, output_dim)

        # --- CNN Encoder-Decoder (for grid processing) ---
        # Input channels: 1 (terrain) + 2 (params U, K) = 3
        cnn_input_channels = 3
        cnn_activation = nn.LeakyReLU(0.2) # Use LeakyReLU for CNN
        self.encoder = nn.Sequential(
            nn.Conv2d(cnn_input_channels, 32, 3, padding=1), cnn_activation,
            nn.Conv2d(32, 64, 3, padding=1), cnn_activation, nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), cnn_activation
        )
        # State Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), cnn_activation,
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), cnn_activation,
            nn.Conv2d(32, output_dim, 3, padding=1)
        )
        # Derivative Decoder
        self.derivative_decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), cnn_activation,
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), cnn_activation,
            nn.Conv2d(32, output_dim, 3, padding=1)
        )
        # Downsampler for multi-resolution processing
        self.downsampler = nn.Upsample(size=(base_resolution, base_resolution), mode='bilinear', align_corners=False)

        self._init_weights() # Initialize weights after defining all layers

    def _init_weights(self):
        """Initialize network weights."""
        # Initialize Coordinate MLP Feature Extractor
        for m in self.coordinate_feature_extractor.modules():
            if isinstance(m, nn.Linear):
                # Use Xavier Uniform for Linear layers
                gain = 1.0
                # Try to get gain from activation
                activation_module = None
                # Find the activation associated with this linear layer (usually the next module)
                try:
                    # This is heuristic, might need adjustment based on actual Sequential structure
                    idx = list(self.coordinate_feature_extractor.children()).index(m)
                    if idx + 1 < len(self.coordinate_feature_extractor):
                        next_module = self.coordinate_feature_extractor[idx+1]
                        if isinstance(next_module, (nn.Tanh, nn.ReLU, nn.LeakyReLU, nn.Sigmoid)): # Add other activations if needed
                             activation_module = next_module
                except ValueError:
                    pass # Layer not found directly

                if activation_module and hasattr(nn.init, 'calculate_gain'):
                     try:
                          gain = nn.init.calculate_gain(activation_module.__class__.__name__.lower())
                     except ValueError:
                          gain = 1.0 # Default gain if activation not recognized

                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize Output Heads
        nn.init.xavier_uniform_(self.state_head.weight, gain=1.0)
        if self.state_head.bias is not None: nn.init.constant_(self.state_head.bias, 0)
        nn.init.xavier_uniform_(self.derivative_head.weight, gain=1.0)
        if self.derivative_head.bias is not None: nn.init.constant_(self.derivative_head.bias, 0)

        # Initialize CNN Layers (Encoder/Decoders) - Kaiming He for ReLU/LeakyReLU
        def init_cnn(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu') # Use leaky_relu as default for CNN
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.encoder.apply(init_cnn)
        self.decoder.apply(init_cnn)
        self.derivative_decoder.apply(init_cnn)


    def _prepare_coord_input(self, coords_dict):
        """准备并验证坐标输入以进行预测 (Integrated from MLP_PINN logic)。"""
        expected_keys = []
        if self.coordinate_input_dim >= 1: expected_keys.append('x')
        if self.coordinate_input_dim >= 2: expected_keys.append('y')
        if self.coordinate_input_dim >= 3: expected_keys.append('t')
        # Assume extra dimensions are k, u
        if self.coordinate_input_dim == 5:
            expected_keys.extend(['k', 'u'])
        elif self.coordinate_input_dim > 3:
            for i in range(3, self.coordinate_input_dim):
                 expected_keys.append(f'param{i-2}')
            logging.warning(f"AdaptiveFastscapePINN coordinate_input_dim={self.coordinate_input_dim} > 3. Assuming extra inputs are 'param1', 'param2', ...")

        tensors_to_cat = []
        ref_tensor_for_attrs = None
        # Find a reference tensor first
        for key in expected_keys:
            if key in coords_dict and isinstance(coords_dict[key], torch.Tensor):
                ref_tensor_for_attrs = coords_dict[key]
                break
        if ref_tensor_for_attrs is None:
             # Fallback: check any value in the dict
             for v in coords_dict.values():
                  if isinstance(v, torch.Tensor):
                       ref_tensor_for_attrs = v
                       break
        if ref_tensor_for_attrs is None:
             raise ValueError("无法确定形状/设备/类型，因为输入 coords_dict 中没有张量。")


        for key in expected_keys:
            if key in coords_dict:
                 tensor = coords_dict[key]
                 # Ensure tensor is on the correct device and dtype
                 tensor = tensor.to(device=ref_tensor_for_attrs.device, dtype=ref_tensor_for_attrs.dtype)
                 tensors_to_cat.append(tensor)
            else:
                # If optional param (k, u, paramX) and not provided, use zeros
                if key in ['k', 'u'] or key.startswith('param'):
                     logging.debug(f"Parameter '{key}' not found in coords_dict, using zeros.")
                     tensors_to_cat.append(torch.zeros_like(ref_tensor_for_attrs))
                else:
                     raise ValueError(f"缺少必需的坐标键 '{key}' (coordinate_input_dim={self.coordinate_input_dim})")

        if not tensors_to_cat:
             raise ValueError("未找到用于坐标 MLP 输入的张量。")

        # Ensure all tensors have compatible shapes for concatenation (e.g., same number of points)
        num_points = tensors_to_cat[0].shape[0]
        for i, t in enumerate(tensors_to_cat):
            if t.shape[0] != num_points:
                raise ValueError(f"坐标张量 '{expected_keys[i]}' 的点数 ({t.shape[0]}) 与第一个张量 ({num_points}) 不匹配。")
            # Ensure tensors are [N, 1] before concatenating
            if t.ndim == 1:
                tensors_to_cat[i] = t.unsqueeze(-1)
            elif t.ndim != 2 or t.shape[1] != 1:
                 # Allow [N, C] for parameters k, u if they were sampled that way
                 if key in ['k', 'u'] and t.ndim == 2 and t.shape[1] > 1:
                      pass # Accept [N, C] for params
                 else:
                      raise ValueError(f"坐标张量 '{expected_keys[i]}' 的形状应为 [N] 或 [N, 1]，但得到 {t.shape}")


        model_input = torch.cat(tensors_to_cat, dim=-1) # Concatenate along the last dimension

        # Validate final input shape
        if model_input.shape[-1] != self.coordinate_input_dim:
             raise ValueError(f"坐标 MLP 输入维度不匹配。预期 {self.coordinate_input_dim}, 得到 {model_input.shape[-1]}")

        return model_input

    def _sample_at_coords(self, param_grid, x_coords_norm, y_coords_norm):
        """在参数网格上采样局部值 (使用归一化坐标 [0, 1])，支持批处理。"""
        if param_grid is None:
            # Determine output shape based on coords shape
            if x_coords_norm.ndim == 3: # Batched coords [B, N, 1]
                batch_size, num_points, _ = x_coords_norm.shape
                return torch.zeros(batch_size, num_points, 1, device=x_coords_norm.device, dtype=x_coords_norm.dtype)
            else: # Unbatched coords [N, 1]
                num_points, _ = x_coords_norm.shape
                return torch.zeros(num_points, 1, device=x_coords_norm.device, dtype=x_coords_norm.dtype)

        device = x_coords_norm.device
        dtype = x_coords_norm.dtype # Use coordinate dtype for sampling grid

        # Ensure param_grid is tensor and on correct device
        if not isinstance(param_grid, torch.Tensor):
             param_grid = torch.tensor(param_grid, device=device, dtype=dtype) # Use coord dtype
        else:
             param_grid = param_grid.to(device=device, dtype=dtype) # Use coord dtype

        # Ensure param_grid is [B, C, H, W]
        if param_grid.ndim == 2: param_grid = param_grid.unsqueeze(0).unsqueeze(0) # H,W -> 1,1,H,W
        elif param_grid.ndim == 3: param_grid = param_grid.unsqueeze(1) # B,H,W -> B,1,H,W
        elif param_grid.ndim != 4: raise ValueError(f"param_grid 形状无效: {param_grid.shape}")

        grid_batch_size = param_grid.shape[0]
        num_channels = param_grid.shape[1]

        # Ensure coords are [B, N, 1] or [N, 1]
        is_batched_coords = x_coords_norm.ndim == 3
        if is_batched_coords:
            coord_batch_size = x_coords_norm.shape[0]
            num_points = x_coords_norm.shape[1]
            if coord_batch_size != grid_batch_size and grid_batch_size != 1 and coord_batch_size != 1:
                 raise ValueError(f"坐标批次大小 ({coord_batch_size}) 与参数网格批次大小 ({grid_batch_size}) 不兼容")
        else: # Unbatched coords
            if x_coords_norm.ndim == 1: x_coords_norm = x_coords_norm.unsqueeze(-1)
            if y_coords_norm.ndim == 1: y_coords_norm = y_coords_norm.unsqueeze(-1)
            num_points = x_coords_norm.shape[0]
            coord_batch_size = 1 # Treat as batch size 1
            # Add batch dim for grid_sample: [N, 1] -> [1, N, 1]
            x_coords_norm = x_coords_norm.unsqueeze(0)
            y_coords_norm = y_coords_norm.unsqueeze(0)

        # Convert coordinates [0, 1] to [-1, 1] for grid_sample
        x_sample = 2.0 * torch.clamp(x_coords_norm, 0, 1) - 1.0
        y_sample = 2.0 * torch.clamp(y_coords_norm, 0, 1) - 1.0

        # Prepare sampling grid [B, N, 1, 2] or [B, 1, N, 2]? grid_sample expects [B, H_out, W_out, 2]
        # Let H_out=N, W_out=1. So shape [B, N, 1, 2]
        grid = torch.cat([x_sample, y_sample], dim=-1) # Shape [B, N, 2]
        grid = grid.unsqueeze(2) # Shape [B, N, 1, 2]

        # Handle batch broadcasting between grid and coords
        final_batch_size = max(grid_batch_size, coord_batch_size)
        if param_grid.shape[0] == 1 and final_batch_size > 1:
            param_grid = param_grid.expand(final_batch_size, -1, -1, -1)
        if grid.shape[0] == 1 and final_batch_size > 1:
            grid = grid.expand(final_batch_size, -1, -1, -1)

        # Sample [B, C, N, 1]
        sampled = F.grid_sample(param_grid, grid, mode='bilinear', padding_mode='border', align_corners=False)

        # Reshape to [B, N, C]
        sampled = sampled.squeeze(-1).permute(0, 2, 1) # [B, C, N] -> [B, N, C]

        # If input coords were not batched, remove the batch dimension
        if not is_batched_coords:
            sampled = sampled.squeeze(0) # [1, N, C] -> [N, C]

        return sampled # Shape [N, C] or [B, N, C]

    def _encode_time(self, t_target, batch_size, device, dtype):
        """将时间编码为特征向量 (增强版)"""
        if isinstance(t_target, (int, float)):
            t_tensor = torch.full((batch_size, 1), float(t_target), device=device, dtype=dtype)
        elif isinstance(t_target, (list, tuple)):
            # 处理列表/元组输入
            try:
                t_tensor = torch.tensor(t_target, device=device, dtype=dtype)
                if t_tensor.ndim == 1:  # [B]
                    if t_tensor.shape[0] != batch_size:
                        # 如果长度不匹配批次大小，广播或截断
                        if t_tensor.shape[0] == 1:
                            t_tensor = t_tensor.expand(batch_size)
                        else:
                            logging.warning(f"t_target 长度 {t_tensor.shape[0]} 与 batch_size {batch_size} 不匹配。使用前 {batch_size} 个元素或重复。")
                            t_tensor = t_tensor[:batch_size] if t_tensor.shape[0] > batch_size else t_tensor.repeat(batch_size // t_tensor.shape[0] + 1)[:batch_size]
                    t_tensor = t_tensor.view(batch_size, 1)
                elif t_tensor.ndim == 0:  # 标量
                    t_tensor = t_tensor.expand(batch_size, 1)
            except (ValueError, TypeError) as e:
                raise TypeError(f"无法将 t_target 转换为张量: {e}")
        elif isinstance(t_target, torch.Tensor):
            # 处理张量输入
            t_target = t_target.to(device=device, dtype=dtype)
            if t_target.numel() == 1:  # 单个元素张量
                t_tensor = t_target.expand(batch_size, 1)
            elif t_target.ndim == 1:  # 向量 [B]
                if t_target.shape[0] != batch_size:
                    # 如果长度不匹配批次大小，广播或截断
                    if t_target.shape[0] == 1:
                        t_tensor = t_target.expand(batch_size)
                    else:
                        logging.warning(f"t_target 长度 {t_target.shape[0]} 与 batch_size {batch_size} 不匹配。使用前 {batch_size} 个元素或重复。")
                        t_tensor = t_target[:batch_size] if t_target.shape[0] > batch_size else t_target.repeat(batch_size // t_target.shape[0] + 1)[:batch_size]
                else:
                    t_tensor = t_target
                t_tensor = t_tensor.view(batch_size, 1)
            elif t_target.ndim == 2:  # 已经是 [B, C]
                if t_target.shape[0] != batch_size:
                    # 处理批次大小不匹配
                    if t_target.shape[0] == 1:
                        t_tensor = t_target.expand(batch_size, -1)
                    else:
                        logging.warning(f"t_target 批次大小 {t_target.shape[0]} 与输入 batch_size {batch_size} 不匹配。使用前 {batch_size} 个批次或重复。")
                        t_tensor = t_target[:batch_size] if t_target.shape[0] > batch_size else t_target.repeat(batch_size // t_target.shape[0] + 1, 1)[:batch_size]
                else:
                    t_tensor = t_target
                # 如果有多个通道，取平均值
                if t_tensor.shape[1] > 1:
                    t_tensor = t_tensor.mean(dim=1, keepdim=True)
            else:  # 更高维度
                raise ValueError(f"t_target 维度过多: {t_target.ndim}。预期为 0、1 或 2。")
        else:
            raise TypeError(f"不支持的目标时间类型: {type(t_target)}")

        # Simple scaling as a basic time encoding
        return t_tensor * 0.01

    def _fuse_time_features(self, spatial_features, time_features):
        """融合时间特征到空间特征 (简化调制)"""
        batch_size = spatial_features.shape[0]
        # Ensure time_features is [B, 1, 1, 1]
        time_channel = time_features.view(batch_size, 1, 1, 1).to(spatial_features.dtype)
        # Modulate spatial features
        return spatial_features * (1.0 + time_channel) # Additive modulation factor

    def _process_with_cnn(self, initial_state, k_field, u_field, t_target):
        """使用CNN处理（通常是小尺寸或基础分辨率）"""
        device = initial_state.device
        dtype = initial_state.dtype

        # Handle different input shapes
        # Ensure we have at least 3 dimensions [B, H, W] or [B, C, H, W]
        if initial_state.ndim < 3:
            raise ValueError(f"initial_state must have at least 3 dimensions, got shape {initial_state.shape}")

        # Add batch dimension if missing
        if initial_state.ndim == 2:  # [H, W] -> [1, 1, H, W]
            initial_state = initial_state.unsqueeze(0).unsqueeze(0)
        elif initial_state.ndim == 3:  # [B, H, W] -> [B, 1, H, W] or [1, C, H, W] -> [1, C, H, W]
            # Check if first dimension is likely batch or channel
            if initial_state.shape[0] > 16:  # Heuristic: if first dim > 16, likely [C, H, W] not [B, H, W]
                initial_state = initial_state.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
            else:
                initial_state = initial_state.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

        # Now initial_state should be [B, C, H, W]
        batch_size = initial_state.shape[0]

        # Process k_field and u_field similarly
        # For k_field
        if isinstance(k_field, (int, float)):
            k_field = torch.full_like(initial_state, float(k_field))
        elif isinstance(k_field, torch.Tensor):
            if k_field.ndim == 0:  # scalar tensor
                k_field = torch.full_like(initial_state, k_field.item())
            elif k_field.ndim == 1:  # [B] -> [B, 1, 1, 1]
                k_field = k_field.view(-1, 1, 1, 1).expand(-1, 1, initial_state.shape[2], initial_state.shape[3])
            elif k_field.ndim == 2:  # [H, W] -> [1, 1, H, W]
                k_field = k_field.unsqueeze(0).unsqueeze(0)
            elif k_field.ndim == 3:  # [B, H, W] -> [B, 1, H, W] or [C, H, W] -> [1, C, H, W]
                if k_field.shape[0] == batch_size:
                    k_field = k_field.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
                else:
                    k_field = k_field.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
        else:
            k_field = torch.full_like(initial_state, 0.0)  # Default to zeros

        # For u_field
        if isinstance(u_field, (int, float)):
            u_field = torch.full_like(initial_state, float(u_field))
        elif isinstance(u_field, torch.Tensor):
            if u_field.ndim == 0:  # scalar tensor
                u_field = torch.full_like(initial_state, u_field.item())
            elif u_field.ndim == 1:  # [B] -> [B, 1, 1, 1]
                u_field = u_field.view(-1, 1, 1, 1).expand(-1, 1, initial_state.shape[2], initial_state.shape[3])
            elif u_field.ndim == 2:  # [H, W] -> [1, 1, H, W]
                u_field = u_field.unsqueeze(0).unsqueeze(0)
            elif u_field.ndim == 3:  # [B, H, W] -> [B, 1, H, W] or [C, H, W] -> [1, C, H, W]
                if u_field.shape[0] == batch_size:
                    u_field = u_field.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
                else:
                    u_field = u_field.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
        else:
            u_field = torch.full_like(initial_state, 0.0)  # Default to zeros

        # 打印输入形状以进行调试
        logging.debug(f"initial_state shape: {initial_state.shape}")
        logging.debug(f"k_field shape: {k_field.shape}")
        logging.debug(f"u_field shape: {u_field.shape}")

        # 确保所有输入都有相同的形状
        if initial_state.shape != k_field.shape or initial_state.shape != u_field.shape:
            # 获取目标形状（高度和宽度）
            # 安全地获取形状维度
            if initial_state.ndim >= 4:
                _, _, h, w = initial_state.shape
            elif initial_state.ndim == 3:
                _, h, w = initial_state.shape
            else:
                h, w = initial_state.shape[-2:] if initial_state.ndim >= 2 else (1, 1)

            # Reshape k_field and u_field if needed
            if k_field.shape[-2:] != (h, w):
                k_field = F.interpolate(k_field, size=(h, w), mode='nearest')
            if u_field.shape[-2:] != (h, w):
                u_field = F.interpolate(u_field, size=(h, w), mode='nearest')

            # Ensure channel dimension is 1
            if k_field.shape[1] != 1:
                k_field = k_field.mean(dim=1, keepdim=True)
            if u_field.shape[1] != 1:
                u_field = u_field.mean(dim=1, keepdim=True)

            # Ensure batch dimension matches
            if k_field.shape[0] != batch_size:
                k_field = k_field.expand(batch_size, -1, -1, -1)
            if u_field.shape[0] != batch_size:
                u_field = u_field.expand(batch_size, -1, -1, -1)

        # Concatenate input channels
        cnn_input = torch.cat([initial_state, k_field, u_field], dim=1) # [B, 3, H, W]

        # Time encoding
        t_encoded = self._encode_time(t_target, batch_size, device, dtype)

        # CNN processing
        features = self.encoder(cnn_input)
        fused_features = self._fuse_time_features(features, t_encoded)

        # Return dictionary with decoded state and derivative
        outputs = {}
        if self.output_state:
            outputs['state'] = self.decoder(fused_features)
        if self.output_derivative:
            outputs['derivative'] = self.derivative_decoder(fused_features)
        return outputs

    def _process_multi_resolution(self, initial_state, k_field, u_field, t_target, original_shape):
        """多分辨率处理中等尺寸输入"""
        # Downsample inputs
        initial_state_down = self.downsampler(initial_state)
        k_field_down = self.downsampler(k_field)
        u_field_down = self.downsampler(u_field)

        # Process at base resolution (returns dict)
        output_dict_down = self._process_with_cnn(initial_state_down, k_field_down, u_field_down, t_target)

        # Upsample results back to original resolution
        output_dict_up = {}
        upsampler = nn.Upsample(size=original_shape, mode='bilinear', align_corners=False)
        for key, tensor_down in output_dict_down.items():
            output_dict_up[key] = upsampler(tensor_down)

        return output_dict_up

    def _process_tiled(self, initial_state, k_field, u_field, t_target, original_shape, tile_size=None, overlap=0.1):
        """分块处理超大尺寸输入 (带重叠)"""
        if tile_size is None: tile_size = self.base_resolution
        height, width = original_shape
        batch_size = initial_state.shape[0]
        device = initial_state.device
        dtype = initial_state.dtype

        # Calculate overlap pixels and stride
        overlap_pixels = int(tile_size * overlap)
        stride = tile_size - overlap_pixels
        if stride <= 0:
             logging.warning(f"Tile stride ({stride}) is non-positive (tile_size={tile_size}, overlap={overlap}). Setting stride to 1.")
             stride = 1

        # Determine output keys based on model settings
        output_keys = []
        if self.output_state: output_keys.append('state')
        if self.output_derivative: output_keys.append('derivative')
        if not output_keys: raise ValueError("Tiled processing requires at least one output ('state' or 'derivative')")

        # Create result dictionary and count tensor
        result_dict = {key: torch.zeros((batch_size, self.output_dim, height, width), device=device, dtype=dtype) for key in output_keys}
        counts = torch.zeros((batch_size, 1, height, width), device=device, dtype=dtype)

        # Use Hann window for smooth blending
        window = torch.hann_window(tile_size, periodic=False, device=device, dtype=dtype)
        window = window**0.75 # Enhance center weight
        window2d = window[:, None] * window[None, :] # H, W

        # Calculate start indices to cover the whole image
        h_starts = list(range(0, height - overlap_pixels, stride)) # Ensure last full stride is included
        w_starts = list(range(0, width - overlap_pixels, stride))
        # Add final tile start if needed to cover edges
        if height > tile_size and (height - tile_size) % stride != 0 :
             h_starts.append(height - tile_size)
        elif height <= tile_size and 0 not in h_starts:
             h_starts.append(0)
        if width > tile_size and (width - tile_size) % stride != 0:
             w_starts.append(width - tile_size)
        elif width <= tile_size and 0 not in w_starts:
             w_starts.append(0)
        h_starts = sorted(list(set(h_starts))) # Ensure unique and sorted
        w_starts = sorted(list(set(w_starts)))

        for h_start in h_starts:
            for w_start in w_starts:
                h_end = min(h_start + tile_size, height)
                w_end = min(w_start + tile_size, width)
                current_tile_h = h_end - h_start
                current_tile_w = w_end - w_start

                if current_tile_h <= 0 or current_tile_w <= 0: continue

                # Extract tile slices
                h_slice = slice(h_start, h_end)
                w_slice = slice(w_start, w_end)

                initial_tile = initial_state[:, :, h_slice, w_slice]
                k_tile = k_field[:, :, h_slice, w_slice]
                u_tile = u_field[:, :, h_slice, w_slice]

                # Pad if tile is smaller than tile_size (e.g., at edges)
                pad_h = max(0, tile_size - current_tile_h)
                pad_w = max(0, tile_size - current_tile_w)
                if pad_h > 0 or pad_w > 0:
                     # Use reflect padding
                     initial_tile = F.pad(initial_tile, (0, pad_w, 0, pad_h), mode='reflect')
                     k_tile = F.pad(k_tile, (0, pad_w, 0, pad_h), mode='reflect')
                     u_tile = F.pad(u_tile, (0, pad_w, 0, pad_h), mode='reflect')

                # Process the tile (returns dict)
                tile_output_dict = self._process_with_cnn(initial_tile, k_tile, u_tile, t_target)

                # Crop back if padded
                if pad_h > 0 or pad_w > 0:
                     for key in tile_output_dict:
                          tile_output_dict[key] = tile_output_dict[key][:, :, :current_tile_h, :current_tile_w]

                # Get the window for the current tile size
                current_window = window2d[:current_tile_h, :current_tile_w].view(1, 1, current_tile_h, current_tile_w)

                # Accumulate weighted results
                for key in output_keys:
                    if key in tile_output_dict:
                         result_dict[key][:, :, h_slice, w_slice] += tile_output_dict[key] * current_window
                    else:
                         logging.warning(f"Key '{key}' expected but not found in tile output dictionary during tiling.")

                # Accumulate weights
                counts[:, :, h_slice, w_slice] += current_window

        # Avoid division by zero
        counts = torch.clamp(counts, min=1e-8)

        # Normalize results
        final_output_dict = {key: result_dict[key] / counts for key in output_keys}

        # Return dictionary or single tensor
        if len(final_output_dict) == 1:
            return next(iter(final_output_dict.values()))
        return final_output_dict


    def _predict_state_adaptive(self, initial_state, params, t_target):
        """优化的网格状态预测，支持多分辨率处理"""
        input_shape = initial_state.shape[-2:] # H, W
        batch_size = initial_state.shape[0]
        device = initial_state.device
        dtype = initial_state.dtype

        # Extract and ensure parameter shapes using prepare_parameter from utils
        # Assuming params keys are 'K', 'U' etc.
        k_field = prepare_parameter(params.get('K'), input_shape, batch_size, device, dtype)
        u_field = prepare_parameter(params.get('U'), input_shape, batch_size, device, dtype)

        # Extract additional parameters for physics calculations if provided
        # These will be passed to the physics functions in the forward pass
        self.physics_params = {
            'dx': params.get('dx', 1.0),
            'dy': params.get('dy', 1.0),
            'precip': params.get('precip', 1.0),
            'da_params': params.get('da_params', {})
        }

        # Choose processing strategy based on input size
        if max(input_shape) <= self.base_resolution:
            logging.debug(f"Input size {input_shape} <= base_resolution {self.base_resolution}. Using direct CNN.")
            return self._process_with_cnn(initial_state, k_field, u_field, t_target)
        elif max(input_shape) <= self.max_resolution:
            logging.debug(f"Input size {input_shape} <= max_resolution {self.max_resolution}. Using multi-resolution processing.")
            return self._process_multi_resolution(initial_state, k_field, u_field, t_target, input_shape)
        else:
            logging.info(f"Input size {input_shape} > max_resolution {self.max_resolution}. Using tiled processing.")
            return self._process_tiled(initial_state, k_field, u_field, t_target, input_shape, tile_size=self.base_resolution, overlap=0.1)


    def forward(self, x, mode='predict_state'):
        """
        前向传播，支持双输出和不同模式。

        Args:
            x: 输入数据 (字典或元组，取决于模式)
               - mode='predict_coords': 字典，包含 'x', 'y', 't' (物理坐标)
                                        以及可选的参数网格 'k_grid', 'u_grid' 用于采样。
               - mode='predict_state': 字典或元组 (initial_state, params, t_target)
                                        initial_state: [B, 1, H, W] or [B, H, W]
                                        params: 字典，包含 'K', 'U' (标量, [B], [H,W], [B,H,W], or [B,1,H,W])
                                                以及可选的 'dx', 'dy', 'precip', 'da_params' 用于物理计算
                                        t_target: 标量, [B], or [B, 1]
            mode (str): 'predict_coords' 或 'predict_state'

        Returns:
            dict or torch.Tensor: 包含 'state' 和/或 'derivative' 的字典，
                                  或单个张量（如果只请求一个输出）。
        """
        outputs = {}

        if mode == 'predict_coords':
            # 1. Prepare and standardize coordinates and sample parameters
            if not isinstance(x, dict): raise TypeError("对于 'predict_coords' 模式，输入 x 必须是字典。")

            # Standardize coordinates to [0, 1] using domain info
            coords_norm = standardize_coordinate_system(
                x,
                domain_x=self.domain_x,
                domain_y=self.domain_y,
                normalize=True # Normalize to [0,1]
            )
            x_coords_norm = coords_norm['x'] # Shape [N, 1] or [B, N, 1]
            y_coords_norm = coords_norm['y'] # Shape [N, 1] or [B, N, 1]
            t_coords = coords_norm['t']      # Shape [N, 1] or [B, N, 1]

            # Sample parameters K and U at normalized coordinates
            # _sample_at_coords expects normalized coords [0, 1]
            k_value = self._sample_at_coords(x.get('k_grid'), x_coords_norm, y_coords_norm) # Shape [N, Ck] or [B, N, Ck]
            u_value = self._sample_at_coords(x.get('u_grid'), x_coords_norm, y_coords_norm) # Shape [N, Cu] or [B, N, Cu]

            # Ensure sampled params have shape [N, 1] or [B, N, 1] (assuming C=1)
            if k_value.shape[-1] != 1: k_value = k_value.mean(dim=-1, keepdim=True) # Average if multiple channels sampled
            if u_value.shape[-1] != 1: u_value = u_value.mean(dim=-1, keepdim=True)

            # Prepare input dictionary for the internal MLP part
            mlp_input_dict = {'x': x_coords_norm, 'y': y_coords_norm, 't': t_coords, 'k': k_value, 'u': u_value}
            model_input = self._prepare_coord_input(mlp_input_dict) # Concatenates to [N, D] or [B*N, D]

            # 2. Get shared features from the coordinate feature extractor
            features = self.coordinate_feature_extractor(model_input)

            # 3. Compute outputs using respective heads
            if self.output_state:
                outputs['state'] = self.state_head(features)
            if self.output_derivative:
                outputs['derivative'] = self.derivative_head(features)

            # Reshape if input was batched [B, N, 1] -> output should be [B, N, C_out]
            if x_coords_norm.ndim == 3: # Input was [B, N, 1]
                 batch_size = x_coords_norm.shape[0]
                 num_points = x_coords_norm.shape[1]
                 if 'state' in outputs:
                      outputs['state'] = outputs['state'].view(batch_size, num_points, self.output_dim)
                 if 'derivative' in outputs:
                      outputs['derivative'] = outputs['derivative'].view(batch_size, num_points, self.output_dim)


        elif mode == 'predict_state':
            # 1. Parse input
            if isinstance(x, dict):
                initial_state = x.get('initial_state')
                params = x.get('params')
                t_target = x.get('t_target')
            elif isinstance(x, (tuple, list)) and len(x) == 3:
                initial_state, params, t_target = x
            else: raise ValueError("对于 'predict_state' 模式，输入 x 必须是字典或 (initial_state, params, t_target) 元组/列表")
            if initial_state is None or params is None or t_target is None: raise ValueError("缺少 'initial_state', 'params', 或 't_target' 输入")

            # 2. Call adaptive state prediction logic
            outputs = self._predict_state_adaptive(initial_state, params, t_target)

        else:
            raise ValueError(f"未知的 forward 模式: {mode}")

        # Return dictionary or single tensor based on requested outputs
        if not outputs:
             # This case should ideally be prevented by set_output_mode check
             raise ValueError("模型未配置为输出任何内容 (state=False, derivative=False)")
        elif len(outputs) == 1:
             return next(iter(outputs.values())) # Return the single tensor
        else:
             # Ensure both state and derivative are requested if both are present
             if self.output_state and self.output_derivative:
                 return outputs
             elif self.output_state:
                 return outputs['state']
             elif self.output_derivative:
                 return outputs['derivative']
             else: # Should not happen
                 return outputs # Return dict if somehow len > 1 but only one mode active
