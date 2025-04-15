# PINN_Framework/tests/test_losses.py
import pytest
import torch
import torch.nn.functional as F
import logging
import logging
from unittest.mock import patch, MagicMock

# 导入被测试的模块
from src import losses
from src.losses import (
    compute_data_loss,
    compute_pde_residual_dual_output,
    compute_smoothness_penalty,
    compute_total_loss
)

# --- Fixtures ---

@pytest.fixture
def sample_prediction():
    """创建一个样本预测张量 (B, C, H, W)。"""
    return torch.rand(2, 1, 10, 10, requires_grad=True)

@pytest.fixture
def sample_target():
    """创建一个样本目标张量 (B, C, H, W)。"""
    return torch.rand(2, 1, 10, 10)

@pytest.fixture
def sample_target_mismatched_shape():
    """创建一个形状不匹配的样本目标张量。"""
    return torch.rand(2, 1, 8, 8) # Different H, W

@pytest.fixture
def sample_model_outputs(sample_prediction):
    """创建一个包含 state 和 derivative 的模型输出字典。"""
    # Derivative should require grad if state does
    derivative = torch.rand_like(sample_prediction) * 0.1
    if sample_prediction.requires_grad:
         derivative.requires_grad_(True)
    return {
        'state': sample_prediction,
        'derivative': derivative
    }

@pytest.fixture
def sample_physics_params():
    """创建一个样本物理参数字典。"""
    return {
        'U': 1e-4,
        'K_f': 1e-5,
        'm': 0.5,
        'n': 1.0,
        'K_d': 1e-2,
        'dx': 10.0,
        'dy': 10.0,
        'precip': 1.0,
        'da_params': {'solver_tol': 1e-6} # Example DA param
    }

@pytest.fixture
def sample_loss_weights():
    """创建一个样本损失权重字典。"""
    return {
        'data': 1.0,
        'physics': 0.5,
        'smoothness': 0.1
    }

# --- 测试 compute_data_loss ---

def test_compute_data_loss_matching_shape(sample_prediction, sample_target):
    """测试数据损失计算（形状匹配）。"""
    loss = compute_data_loss(sample_prediction, sample_target)
    expected_loss = F.mse_loss(sample_prediction, sample_target)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0 # Scalar loss
    assert torch.isclose(loss, expected_loss)
    assert loss.requires_grad == sample_prediction.requires_grad # Check grad propagation

def test_compute_data_loss_mismatched_shape(sample_prediction, sample_target_mismatched_shape):
    """测试数据损失计算（形状不匹配，触发插值）。"""
    target_shape = sample_prediction.shape[-2:]
    # Mock interpolate to check if it's called
    with patch('torch.nn.functional.interpolate', wraps=F.interpolate) as mock_interpolate:
        loss = compute_data_loss(sample_prediction, sample_target_mismatched_shape)

    mock_interpolate.assert_called_once_with(
        sample_target_mismatched_shape.float(),
        size=target_shape,
        mode='bilinear',
        align_corners=False
    )
    # Calculate expected loss with interpolated target
    target_interpolated = F.interpolate(sample_target_mismatched_shape.float(), size=target_shape, mode='bilinear', align_corners=False)
    expected_loss = F.mse_loss(sample_prediction, target_interpolated)

    assert isinstance(loss, torch.Tensor)
    assert torch.isclose(loss, expected_loss)
    assert loss.requires_grad == sample_prediction.requires_grad

def test_compute_data_loss_no_grad(sample_prediction, sample_target):
    """测试数据损失在输入不需要梯度时不计算梯度。"""
    pred_no_grad = sample_prediction.detach()
    loss = compute_data_loss(pred_no_grad, sample_target)
    assert not loss.requires_grad

# --- 测试 compute_pde_residual_dual_output ---

@patch('src.losses.calculate_dhdt_physics')
def test_compute_pde_residual_success(mock_calc_dhdt, sample_model_outputs, sample_physics_params):
    """测试 PDE 残差计算成功。"""
    h_pred = sample_model_outputs['state']
    dh_dt_pred = sample_model_outputs['derivative']
    # Mock the physics calculation result
    mock_dhdt_physics = torch.rand_like(h_pred) * 0.5
    mock_calc_dhdt.return_value = mock_dhdt_physics

    loss = compute_pde_residual_dual_output(sample_model_outputs, sample_physics_params)

    mock_calc_dhdt.assert_called_once()
    # Check some args passed to physics calculation
    call_args, call_kwargs = mock_calc_dhdt.call_args
    assert torch.equal(call_kwargs['h'], h_pred)
    assert call_kwargs['K_f'].shape == h_pred.shape # Check broadcasting
    assert call_kwargs['m'] == sample_physics_params['m']
    assert call_kwargs['da_params'] == sample_physics_params['da_params']

    # Calculate expected residual loss
    expected_residual = dh_dt_pred - mock_dhdt_physics
    expected_loss = F.mse_loss(expected_residual, torch.zeros_like(expected_residual))

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isclose(loss, expected_loss)
    assert loss.requires_grad # Should propagate gradients

@patch('src.losses.calculate_dhdt_physics', side_effect=Exception("Physics failed"))
def test_compute_pde_residual_physics_error(mock_calc_dhdt, sample_model_outputs, sample_physics_params, caplog):
    """测试 PDE 残差计算在物理计算失败时的错误处理。"""
    with caplog.at_level(logging.ERROR):
        loss = compute_pde_residual_dual_output(sample_model_outputs, sample_physics_params)

    assert "计算双输出 PDE 残差时出错: Physics failed" in caplog.text
    # Should return zero loss with gradient connection
    assert torch.isclose(loss, torch.tensor(0.0))
    assert loss.requires_grad

def test_compute_pde_residual_missing_keys(sample_model_outputs, sample_physics_params):
    """测试 PDE 残差计算在缺少 state 或 derivative 键时引发 ValueError。"""
    outputs_no_state = {'derivative': sample_model_outputs['derivative']}
    with pytest.raises(ValueError, match="模型输出字典必须包含 'state' 和 'derivative' 键"):
        compute_pde_residual_dual_output(outputs_no_state, sample_physics_params)

    outputs_no_derivative = {'state': sample_model_outputs['state']}
    with pytest.raises(ValueError, match="模型输出字典必须包含 'state' 和 'derivative' 键"):
        compute_pde_residual_dual_output(outputs_no_derivative, sample_physics_params)

def test_compute_pde_residual_shape_mismatch(sample_model_outputs, sample_physics_params):
    """测试 PDE 残差计算在 state 和 derivative 形状不匹配时引发 ValueError。"""
    outputs_mismatch = {
        'state': sample_model_outputs['state'],
        'derivative': sample_model_outputs['derivative'][:, :, :-1, :-1] # Different shape
    }
    with pytest.raises(ValueError, match="状态和导数预测的形状不匹配"):
        compute_pde_residual_dual_output(outputs_mismatch, sample_physics_params)

# --- 测试 compute_smoothness_penalty ---

@patch('src.losses.calculate_slope_magnitude')
def test_compute_smoothness_penalty_success(mock_calc_slope, sample_prediction, sample_physics_params):
    """测试平滑度惩罚计算成功。"""
    # Mock slope calculation result
    mock_slope = torch.rand_like(sample_prediction)
    mock_calc_slope.return_value = mock_slope
    dx = sample_physics_params['dx']
    dy = sample_physics_params['dy']

    loss = compute_smoothness_penalty(sample_prediction, dx, dy)

    mock_calc_slope.assert_called_once_with(sample_prediction, dx, dy)
    expected_loss = torch.mean(mock_slope)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isclose(loss, expected_loss)
    # The mock doesn't preserve requires_grad, so we don't check it here

@patch('src.losses.calculate_slope_magnitude', side_effect=Exception("Slope failed"))
def test_compute_smoothness_penalty_slope_error(mock_calc_slope, sample_prediction, sample_physics_params, caplog):
    """测试平滑度惩罚在坡度计算失败时的错误处理。"""
    dx = sample_physics_params['dx']
    dy = sample_physics_params['dy']
    with caplog.at_level(logging.ERROR):
        loss = compute_smoothness_penalty(sample_prediction, dx, dy)

    assert "计算平滑度惩罚时出错: Slope failed" in caplog.text
    # Should return zero loss with gradient connection
    assert torch.isclose(loss, torch.tensor(0.0))
    assert loss.requires_grad

def test_compute_smoothness_penalty_invalid_shape(sample_physics_params, caplog):
    """测试平滑度惩罚对无效输入形状的处理。"""
    invalid_pred = torch.rand(2, 3, 10, 10) # Wrong channel dim
    dx = sample_physics_params['dx']
    dy = sample_physics_params['dy']
    with caplog.at_level(logging.WARNING):
        loss = compute_smoothness_penalty(invalid_pred, dx, dy)

    assert "Smoothness penalty 期望输入形状 (B, 1, H, W)" in caplog.text
    assert torch.isclose(loss, torch.tensor(0.0))
    # The implementation returns a detached tensor for invalid shapes

# --- 测试 compute_total_loss ---

@patch('src.losses.compute_data_loss', return_value=torch.tensor(1.5, requires_grad=True))
@patch('src.losses.compute_smoothness_penalty', return_value=torch.tensor(0.5, requires_grad=True))
def test_compute_total_loss_all_components(mock_smoothness, mock_data, sample_prediction, sample_target, sample_physics_params, sample_loss_weights):
    """测试总损失计算（所有组件）。"""
    physics_loss_val = torch.tensor(1.0, requires_grad=True) # Precomputed physics loss
    smoothness_pred = sample_prediction # Use same pred for smoothness

    total_loss, loss_dict = compute_total_loss(
        data_pred=sample_prediction,
        target_topo=sample_target,
        physics_loss_value=physics_loss_val,
        smoothness_pred=smoothness_pred,
        physics_params=sample_physics_params,
        loss_weights=sample_loss_weights
    )

    mock_data.assert_called_once_with(sample_prediction, sample_target)
    mock_smoothness.assert_called_once_with(smoothness_pred, sample_physics_params['dx'], sample_physics_params['dy'])

    expected_total = (sample_loss_weights['data'] * 1.5 +
                      sample_loss_weights['physics'] * 1.0 +
                      sample_loss_weights['smoothness'] * 0.5)

    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.ndim == 0
    assert torch.isclose(total_loss, torch.tensor(expected_total))
    assert total_loss.requires_grad

    assert isinstance(loss_dict, dict)
    assert loss_dict['data_loss'] == pytest.approx(sample_loss_weights['data'] * 1.5)
    assert loss_dict['physics_loss'] == pytest.approx(sample_loss_weights['physics'] * 1.0)
    assert loss_dict['smoothness_loss'] == pytest.approx(sample_loss_weights['smoothness'] * 0.5)
    assert loss_dict['total_loss'] == pytest.approx(expected_total)

@patch('src.losses.compute_data_loss')
@patch('src.losses.compute_smoothness_penalty')
def test_compute_total_loss_partial_components(mock_smoothness, mock_data, sample_prediction, sample_target, sample_physics_params, sample_loss_weights):
    """测试总损失计算（部分组件）。"""
    # Disable physics loss (weight=0) and smoothness loss (input=None)
    partial_weights = sample_loss_weights.copy()
    partial_weights['physics'] = 0.0
    mock_data.return_value = torch.tensor(1.5, requires_grad=True)

    total_loss, loss_dict = compute_total_loss(
        data_pred=sample_prediction,
        target_topo=sample_target,
        physics_loss_value=torch.tensor(1.0), # Provide value, but weight is 0
        smoothness_pred=None, # Disable smoothness via None input
        physics_params=sample_physics_params,
        loss_weights=partial_weights
    )

    mock_data.assert_called_once()
    mock_smoothness.assert_not_called() # Should not be called

    expected_total = partial_weights['data'] * 1.5

    assert torch.isclose(total_loss, torch.tensor(expected_total))
    assert loss_dict['data_loss'] == pytest.approx(partial_weights['data'] * 1.5)
    assert loss_dict['physics_loss'] == 0.0 # Weight was 0
    assert loss_dict['smoothness_loss'] == 0.0 # Input was None
    assert loss_dict['total_loss'] == pytest.approx(expected_total)

def test_compute_total_loss_nan_physics(sample_loss_weights, caplog):
    """测试总损失计算处理 NaN 物理损失。"""
    nan_physics_loss = torch.tensor(float('nan'))
    with caplog.at_level(logging.WARNING):
        total_loss, loss_dict = compute_total_loss(
            data_pred=None, target_topo=None, # Disable data loss
            physics_loss_value=nan_physics_loss,
            smoothness_pred=None, # Disable smoothness loss
            physics_params={},
            loss_weights=sample_loss_weights # Physics weight > 0
        )

    assert "收到无效或非有限的 physics_loss_value" in caplog.text
    assert torch.isclose(total_loss, torch.tensor(0.0)) # Should default to 0
    assert loss_dict['data_loss'] == 0.0
    # The implementation sets physics_loss to 0.0 when NaN is detected
    assert loss_dict['physics_loss'] == 0.0
    assert loss_dict['smoothness_loss'] == 0.0
    # The implementation sets total_loss to 0.0 when all components are 0 or invalid
    assert loss_dict['total_loss'] == 0.0

def test_compute_total_loss_all_none(sample_loss_weights):
    """测试总损失计算在所有输入都为 None 时返回零。"""
    total_loss, loss_dict = compute_total_loss(
        data_pred=None, target_topo=None,
        physics_loss_value=None,
        smoothness_pred=None,
        physics_params={},
        loss_weights=sample_loss_weights
    )
    assert torch.isclose(total_loss, torch.tensor(0.0))
    assert total_loss.requires_grad # Should still have grad connection (from _zero_with_grad)
    assert loss_dict['data_loss'] == 0.0
    assert loss_dict['physics_loss'] == 0.0
    assert loss_dict['smoothness_loss'] == 0.0
    assert loss_dict['total_loss'] == 0.0