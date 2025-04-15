# PINN_Framework/tests/visualization_scripts/test_viz_model_predictions.py
"""
可视化验证脚本：检查 PINN 模型的预测、与目标的差异以及 PDE 残差。
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import yaml # 用于加载配置文件以获取模型结构等信息

# --- 将项目根目录添加到 Python 路径 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(tests_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
print(f"项目根目录已添加: {project_root}")

# --- 导入必要的模块 ---
try:
    from src.models import AdaptiveFastscapePINN # 导入主模型
    from src.physics import calculate_dhdt_physics # 导入物理计算
    from src.utils import load_config, get_device # 导入工具函数
    print("成功导入 src.models, src.physics, src.utils")
except ImportError as e:
    print(f"错误：无法导入必要的模块: {e}。请检查相关文件是否存在。")
    sys.exit(1)
except Exception as e:
    print(f"导入模块时发生未知错误: {e}")
    sys.exit(1)

# --- 配置路径 ---
OUTPUT_DIR = os.path.join(tests_dir, "visualization_outputs")
DATA_DIR = os.path.join(tests_dir, "test_data")
# 假设我们有一个包含初始、最终和参数的测试数据文件
# 例如，一个 .npz 文件
TEST_DATA_PATH = os.path.join(DATA_DIR, "sample_evolution_data.npz")

# --- 确保目录存在 ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --- 加载或生成测试数据 ---
def get_sample_evolution_data(save_path=TEST_DATA_PATH, shape=(50, 50)):
    """生成或加载包含初始/最终状态和参数的测试数据"""
    if os.path.exists(save_path):
        print(f"加载已存在的测试数据: {save_path}")
        data = np.load(save_path)
        return {
            'initial_topo': torch.tensor(data['initial_topo'], dtype=torch.float32),
            'final_topo': torch.tensor(data['final_topo'], dtype=torch.float32),
            'params': {k: torch.tensor(v, dtype=torch.float32) if isinstance(v, np.ndarray) else float(v) for k, v in data['params'].item().items()}, # 加载字典参数
            't_target': torch.tensor(data['t_target'], dtype=torch.float32)
        }
    else:
        print("生成新的测试演化数据...")
        # 1. 生成初始 DEM (例如，带噪声的斜坡)
        y, x = np.meshgrid(np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1]), indexing='ij')
        initial_dem = 100.0 * (x + y) + np.random.rand(*shape) * 5.0
        initial_tensor = torch.tensor(initial_dem, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]

        # 2. 定义物理参数和目标时间
        params = {
            'U': 0.001, 'K_f': 5e-5, 'm': 0.5, 'n': 1.0, 'K_d': 0.005,
            'dx': 10.0, 'dy': 10.0, 'precip': 1.0, 'da_params': {}
        }
        t_target_val = 5000.0 # 模拟 5000 年
        t_target = torch.tensor(t_target_val, dtype=torch.float32)

        # 3. (简化) 使用物理模型模拟一小步作为 "final_topo"
        # 注意：这只是为了生成示例数据，实际应使用独立的模拟结果或观测数据
        try:
            dhdt = calculate_dhdt_physics(initial_tensor, **params)
            # 简单的欧拉积分估算最终地形
            final_tensor = initial_tensor + dhdt * t_target_val
            final_dem = final_tensor.squeeze().numpy()
        except Exception as e:
            print(f"警告：无法运行物理模拟生成 final_topo: {e}。将使用初始地形代替。")
            final_dem = initial_dem

        # 4. 保存数据
        np.savez(
            save_path,
            initial_topo=initial_dem,
            final_topo=final_dem,
            params=params, # 保存字典
            t_target=t_target_val
        )
        print(f"测试数据已保存到: {save_path}")
        return {
            'initial_topo': initial_tensor,
            'final_topo': torch.tensor(final_dem, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            'params': {k: torch.tensor(v, dtype=torch.float32) if isinstance(v, (int, float, np.ndarray)) else v for k, v in params.items()},
            't_target': t_target
        }

# --- 加载模型 ---
def load_trained_model(checkpoint_path, config_path, device):
    """从检查点和配置文件加载模型"""
    if not os.path.exists(checkpoint_path):
        print(f"错误：检查点文件未找到: {checkpoint_path}")
        return None
    if not os.path.exists(config_path):
        print(f"错误：配置文件未找到: {config_path}")
        return None

    try:
        print(f"加载配置文件: {config_path}")
        config = load_config(config_path) # 使用 utils 加载
        model_config = config.get('model', {})
        # 移除 name 字段（如果存在），因为它不是构造函数参数
        model_config.pop('name', None)
        # 处理可能的 OmegaConf 类型
        from omegaconf import OmegaConf
        if isinstance(model_config, OmegaConf):
             model_config = OmegaConf.to_container(model_config, resolve=True)

        print("初始化模型...")
        # 移除构造函数不接受的 'dtype' 参数（如果存在）
        model_config.pop('dtype', None)
        # --- 处理激活函数字符串 ---
        if 'activation_fn' in model_config and isinstance(model_config['activation_fn'], str):
            act_str = model_config['activation_fn'].lower()
            activation_map = {
                'relu': torch.nn.ReLU,
                'silu': torch.nn.SiLU,
                'tanh': torch.nn.Tanh
                # 可以添加更多映射
            }
            if act_str in activation_map:
                print(f"将激活函数字符串 '{model_config['activation_fn']}' 转换为 {activation_map[act_str]}")
                model_config['activation_fn'] = activation_map[act_str]
            else:
                print(f"警告：未知的激活函数字符串 '{model_config['activation_fn']}'。模型初始化可能失败。")
                # 决定如何处理未知字符串，这里暂时保留让模型报错

        model = AdaptiveFastscapePINN(**model_config) # 使用配置中的参数

        print(f"加载检查点: {checkpoint_path}")
        # 直接使用 weights_only=False，因为我们的数据包含 numpy 数组
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval() # 设置为评估模式
        print("模型加载成功并设置为评估模式。")
        return model, config # 返回模型和配置
    except FileNotFoundError:
        print(f"错误：文件未找到 - {checkpoint_path} 或 {config_path}")
        return None, None
    except KeyError as e:
        print(f"错误：检查点文件中缺少键: {e}")
        return None, None
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# --- 主可视化函数 ---
def visualize_model_predictions(checkpoint_path, config_path):
    print("开始可视化模型预测...")
    device = get_device('auto') # 自动选择设备
    print(f"使用设备: {device}")

    # 加载模型
    model, config = load_trained_model(checkpoint_path, config_path, device)
    if model is None or config is None:
        print("无法加载模型或配置，退出。")
        return

    # 获取测试数据
    test_data = get_sample_evolution_data()
    if test_data is None:
        print("无法获取测试数据，退出。")
        return

    # 将测试数据移动到设备
    initial_state = test_data['initial_topo'].to(device)
    final_topo = test_data['final_topo'].to(device)
    params_tensor = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in test_data['params'].items()}
    t_target = test_data['t_target'].to(device)

    # 准备模型输入
    model_input = {
        'initial_state': initial_state,
        'params': params_tensor, # 传递包含张量的字典
        't_target': t_target
    }

    # --- 模型推理 ---
    print("运行模型推理...")
    with torch.no_grad():
        # 确保模型设置为双输出模式
        if isinstance(model, AdaptiveFastscapePINN):
            model.set_output_mode(state=True, derivative=True)
        outputs = model(model_input, mode='predict_state')
        if not isinstance(outputs, dict) or 'state' not in outputs or 'derivative' not in outputs:
            print(f"错误：模型输出格式不正确。需要包含 'state' 和 'derivative' 的字典，但得到: {type(outputs)}")
            return
        predicted_state = outputs['state']
        predicted_derivative = outputs['derivative']
    print("模型推理完成。")
    print(f"预测状态形状: {predicted_state.shape}")
    print(f"预测导数形状: {predicted_derivative.shape}")

    # --- 计算物理导数和残差 ---
    print("计算物理导数和 PDE 残差...")
    try:
        # 从加载的配置中获取物理参数，覆盖测试数据中的参数（如果需要）
        physics_params_from_config = config.get('physics', {})
        # 合并参数，优先使用 config 中的
        physics_params_for_calc = {**test_data['params'], **physics_params_from_config}
        # 确保 dx, dy 等存在
        physics_params_for_calc.setdefault('dx', 10.0)
        physics_params_for_calc.setdefault('dy', 10.0)
        physics_params_for_calc.setdefault('precip', 1.0)
        physics_params_for_calc.setdefault('da_params', {})

        # 使用模型预测的状态计算物理导数
        dhdt_physics = calculate_dhdt_physics(
            h=predicted_state, # 使用模型预测的状态
            U=physics_params_for_calc['U'],
            K_f=physics_params_for_calc['K_f'],
            m=physics_params_for_calc['m'],
            n=physics_params_for_calc['n'],
            K_d=physics_params_for_calc['K_d'],
            dx=physics_params_for_calc['dx'],
            dy=physics_params_for_calc['dy'],
            precip=physics_params_for_calc['precip'],
            da_params=physics_params_for_calc['da_params']
        )
        pde_residual = predicted_derivative - dhdt_physics
        print("物理导数和残差计算完成。")
    except Exception as e:
        print(f"计算物理导数或残差时出错: {e}")
        import traceback
        traceback.print_exc()
        # 创建零张量以便继续绘图
        dhdt_physics = torch.zeros_like(predicted_state)
        pde_residual = torch.zeros_like(predicted_state)


    # --- 准备绘图数据 (Numpy) ---
    initial_np = initial_state.squeeze().cpu().numpy()
    final_np = final_topo.squeeze().cpu().numpy()
    predicted_np = predicted_state.squeeze().cpu().numpy()
    pred_deriv_np = predicted_derivative.squeeze().cpu().numpy()
    physics_deriv_np = dhdt_physics.squeeze().cpu().numpy()
    residual_np = pde_residual.squeeze().cpu().numpy()
    diff_np = predicted_np - final_np

    # --- 绘图 ---
    print("开始绘图...")
    fig, axes = plt.subplots(2, 4, figsize=(24, 10)) # 调整布局以容纳更多图像
    fig.suptitle(f'Model Prediction Visualization (Checkpoint: {os.path.basename(checkpoint_path)})', fontsize=16)
    axes = axes.ravel() # 展平数组以便索引

    # 确定地形图的颜色范围
    vmin = min(initial_np.min(), final_np.min(), predicted_np.min())
    vmax = max(initial_np.max(), final_np.max(), predicted_np.max())

    # 绘制地形相关图像
    im0 = axes[0].imshow(initial_np, cmap='terrain', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title('Initial State')
    fig.colorbar(im0, ax=axes[0], label='Elevation')

    im1 = axes[1].imshow(final_np, cmap='terrain', origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title('Target State (Final Topo)')
    fig.colorbar(im1, ax=axes[1], label='Elevation')

    im2 = axes[2].imshow(predicted_np, cmap='terrain', origin='lower', vmin=vmin, vmax=vmax)
    axes[2].set_title('Predicted State')
    fig.colorbar(im2, ax=axes[2], label='Elevation')

    # 绘制差异图
    max_abs_diff = np.max(np.abs(diff_np))
    im3 = axes[3].imshow(diff_np, cmap='coolwarm', origin='lower', vmin=-max_abs_diff, vmax=max_abs_diff)
    axes[3].set_title('Difference (Predicted - Target)')
    fig.colorbar(im3, ax=axes[3], label='Elevation Difference')

    # 确定导数图的颜色范围
    deriv_vmin = min(pred_deriv_np.min(), physics_deriv_np.min())
    deriv_vmax = max(pred_deriv_np.max(), physics_deriv_np.max())
    max_abs_deriv = max(abs(deriv_vmin), abs(deriv_vmax))

    # 绘制导数相关图像
    im4 = axes[4].imshow(pred_deriv_np, cmap='coolwarm', origin='lower', vmin=-max_abs_deriv, vmax=max_abs_deriv)
    axes[4].set_title('Predicted dh/dt (Model)')
    fig.colorbar(im4, ax=axes[4], label='Rate')

    im5 = axes[5].imshow(physics_deriv_np, cmap='coolwarm', origin='lower', vmin=-max_abs_deriv, vmax=max_abs_deriv)
    axes[5].set_title('Calculated dh/dt (Physics)')
    fig.colorbar(im5, ax=axes[5], label='Rate')

    # 绘制残差图
    max_abs_residual = np.max(np.abs(residual_np))
    im6 = axes[6].imshow(residual_np, cmap='coolwarm', origin='lower', vmin=-max_abs_residual, vmax=max_abs_residual)
    axes[6].set_title('PDE Residual (Pred_dhdt - Phys_dhdt)')
    fig.colorbar(im6, ax=axes[6], label='Residual')

    # 隐藏最后一个子图（如果不需要）
    axes[7].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存图像
    output_filename = f"model_predictions_{os.path.splitext(os.path.basename(checkpoint_path))[0]}.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    try:
        plt.savefig(output_path)
        print(f"可视化结果已保存到: {output_path}")
    except Exception as e:
        print(f"保存图像时出错: {e}")

    # 显示图像 (可选)
    # plt.show()

    print("可视化完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化 PINN 模型预测结果和 PDE 残差。")
    parser.add_argument('--checkpoint', type=str, required=True, help='预训练模型检查点文件的路径 (.pth)。')
    parser.add_argument('--config', type=str, required=True, help='与检查点对应的训练配置文件路径 (.yaml)。')
    args = parser.parse_args()

    visualize_model_predictions(args.checkpoint, args.config)