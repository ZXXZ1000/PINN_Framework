# PINN_Framework/tests/visualization_scripts/test_viz_physics_step.py
"""
可视化验证脚本：检查单步物理演化 (dh/dt) 计算。
"""
import os
import sys
import numpy as np
# Set matplotlib backend to Agg (non-GUI backend) before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import torch
import argparse # 重新添加 argparse

# --- 将项目根目录添加到 Python 路径 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(tests_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
print(f"项目根目录已添加: {project_root}")

# --- 导入必要的物理模块 ---
try:
    from src.physics import calculate_dhdt_physics
    print("成功导入 src.physics.calculate_dhdt_physics")
except ImportError as e:
    print(f"错误：无法导入物理模块函数: {e}。请检查 src/physics.py 中的函数名和路径。")
    sys.exit(1)
except Exception as e:
    print(f"导入物理模块时发生未知错误: {e}")
    sys.exit(1)

# --- 配置路径 ---
OUTPUT_DIR = os.path.join(tests_dir, "visualization_outputs")
DATA_DIR = os.path.join(tests_dir, "test_data")
SAMPLE_DEM_PATH = os.path.join(DATA_DIR, "sample_dem_slope.npy") # 复用之前的斜坡DEM

# --- 确保目录存在 ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --- 加载或生成测试数据 (复用函数) ---
def load_dem_from_path(dem_path):
    """从指定路径加载 DEM 文件"""
    if not os.path.exists(dem_path):
        print(f"错误：DEM 文件未找到: {dem_path}")
        return None
    try:
        print(f"加载 DEM 文件: {dem_path}")
        dem = np.load(dem_path)
        # 确保 DEM 是 2D 的
        if dem.ndim != 2:
            print(f"错误：加载的 DEM 维度不为 2 (形状: {dem.shape})")
            return None
        # 转换为 PyTorch 张量并添加批次和通道维度
        return torch.tensor(dem, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
    except Exception as e:
        print(f"加载 DEM 文件时出错 ({dem_path}): {e}")
        return None

# --- 主可视化函数 ---
def visualize_physics_step(dem_path):
    """主可视化函数，使用指定的 DEM 文件"""
    print("开始可视化物理步骤 (dh/dt) 计算...")

    # 加载 DEM 数据
    dem_tensor = load_dem_from_path(dem_path)
    if dem_tensor is None:
        print("无法加载 DEM 数据，退出。")
        return
    print(f"DEM 张量形状: {dem_tensor.shape}")

    # 设置物理参数 (根据需要调整)
    physics_params = {
        'U': 0.001,      # 抬升率 (m/yr)
        'K_f': 5e-5,     # 河流侵蚀系数
        'm': 0.5,        # 汇水面积指数
        'n': 1.0,        # 坡度指数
        'K_d': 0.005,    # 山坡扩散系数
        'dx': 10.0,      # 网格间距 x (m)
        'dy': 10.0,      # 网格间距 y (m)
        'precip': 1.0,   # 降水率 (用于汇水面积计算)
        'da_params': {}  # 汇水面积计算的其他参数 (如边界条件)
    }
    print(f"使用的物理参数: {physics_params}")

    # 调用 dh/dt 计算函数
    try:
        print("调用 calculate_dhdt_physics...")
        # 确保传递所有需要的参数
        dhdt_physics = calculate_dhdt_physics(
            h=dem_tensor,
            U=physics_params['U'],
            K_f=physics_params['K_f'],
            m=physics_params['m'],
            n=physics_params['n'],
            K_d=physics_params['K_d'],
            dx=physics_params['dx'],
            dy=physics_params['dy'],
            precip=physics_params['precip'],
            da_params=physics_params['da_params']
        )
        print("函数调用完成。")
        print(f"计算得到的 dh/dt 形状: {dhdt_physics.shape}")

    except NotImplementedError:
         print("错误：calculate_dhdt_physics 函数未实现或未正确导入。")
         return
    except Exception as e:
        print(f"计算 dh/dt 时出错: {e}")
        import traceback
        traceback.print_exc()
        return

    # 准备绘图数据 (从 Tensor 转为 Numpy)
    dem_np = dem_tensor.squeeze().cpu().numpy()
    dhdt_np = dhdt_physics.squeeze().cpu().numpy()

    # --- 绘图 ---
    print("开始绘图...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Physics Step (dh/dt) Calculation Visualization', fontsize=16)

    # 绘制 DEM
    im0 = axes[0].imshow(dem_np, cmap='terrain', origin='lower')
    axes[0].set_title('Input DEM')
    axes[0].set_xlabel('X Index')
    axes[0].set_ylabel('Y Index')
    fig.colorbar(im0, ax=axes[0], label='Elevation (m)')

    # 绘制 dh/dt
    # 使用发散色图，中心为 0
    max_abs_dhdt = np.max(np.abs(dhdt_np))
    im1 = axes[1].imshow(dhdt_np, cmap='coolwarm', origin='lower',
                         vmin=-max_abs_dhdt, vmax=max_abs_dhdt)
    axes[1].set_title('Calculated dh/dt (Physics)')
    axes[1].set_xlabel('X Index')
    axes[1].set_ylabel('Y Index')
    fig.colorbar(im1, ax=axes[1], label='Elevation Change Rate (m/yr)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局

    # 保存图像
    # 使用默认输出文件名
    # 使用输入 DEM 文件名生成输出文件名
    dem_filename = os.path.basename(dem_path)
    output_filename = f"physics_step_{os.path.splitext(dem_filename)[0]}.png"
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
    parser = argparse.ArgumentParser(description="可视化单步物理演化 (dh/dt) 计算。")
    parser.add_argument('--dem', type=str, required=True, help='输入 DEM 文件的路径 (.npy)。')
    # 可以添加其他参数来覆盖默认的物理参数
    # parser.add_argument('--U', type=float, help='覆盖抬升率 U')
    # parser.add_argument('--Kf', type=float, help='覆盖河流侵蚀系数 K_f')
    # ... 其他物理参数 ...
    args = parser.parse_args()

    # TODO: 如果添加了物理参数的命令行参数，在这里更新 physics_params 字典

    visualize_physics_step(args.dem)