# PINN_Framework/tests/visualization_scripts/test_viz_drainage_area.py
"""
可视化验证脚本：检查汇水面积计算。
"""
import os
import sys
import numpy as np
# Set matplotlib backend to Agg (non-GUI backend) before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Qt issues
import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm # 移除 LogNorm
import torch
import logging

# 设置日志级别为INFO，以显示更多调试信息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import argparse # 重新添加 argparse

# --- 将项目根目录添加到 Python 路径 ---
# 当前脚本位于 tests/visualization_scripts/
script_dir = os.path.dirname(os.path.abspath(__file__))
# tests 目录
tests_dir = os.path.dirname(script_dir)
# 项目根目录 (PINN_Framework)
project_root = os.path.dirname(tests_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
print(f"项目根目录已添加: {project_root}")

# --- 导入必要的物理模块 ---
try:
    # 分别导入汇水面积和坡度计算函数
    from src.physics import calculate_drainage_area_ida_dinf_torch, calculate_slope_magnitude, stream_power_erosion
    print("成功导入 src.physics 中的必要函数")
except ImportError as e:
    print(f"错误：无法导入物理模块函数: {e}。请检查 src/physics.py 中的函数名和路径。")
    sys.exit(1)
except Exception as e:
    print(f"导入物理模块时发生未知错误: {e}")
    sys.exit(1)


# --- 配置路径 ---
OUTPUT_DIR = os.path.join(tests_dir, "visualization_outputs")
DATA_DIR = os.path.join(tests_dir, "test_data")
SAMPLE_DEM_PATH = os.path.join(DATA_DIR, "sample_dem_gaussian.npy") # 确认使用高斯峰DEM

# --- 确保目录存在 ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --- 工具函数 ---
def interpolate_data(data, scale_factor=2):
    """使用双线性插值提高数据分辨率

    Args:
        data (numpy.ndarray): 输入数据数组
        scale_factor (int): 缩放因子，默认为2

    Returns:
        numpy.ndarray: 插值后的高分辨率数据
    """
    from scipy.ndimage import zoom
    try:
        # 使用scipy的zoom函数进行插值
        return zoom(data, scale_factor, order=1) # order=1为双线性插值
    except Exception as e:
        print(f"插值数据时出错: {e}")
        return data  # 如果出错，返回原始数据

# --- 工具函数 ---
def interpolate_data(data, scale_factor=2):
    """使用双线性插值提高数据分辨率

    Args:
        data (numpy.ndarray): 输入数据数组
        scale_factor (int): 缩放因子，默认为2

    Returns:
        numpy.ndarray: 插值后的高分辨率数据
    """
    from scipy.ndimage import zoom
    try:
        # 使用scipy的zoom函数进行插值
        return zoom(data, scale_factor, order=1) # order=1为双线性插值
    except Exception as e:
        print(f"插值数据时出错: {e}")
        return data  # 如果出错，返回原始数据

# --- 生成或加载测试数据 ---
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
        print(f"DEM 加载成功，形状: {dem.shape}")
        # 转换为 PyTorch 张量并添加批次和通道维度
        return torch.tensor(dem, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
    except Exception as e:
        print(f"加载 DEM 文件时出错 ({dem_path}): {e}")
        return None

# --- 主可视化函数 ---
def visualize_drainage_area(dem_path):
    """主可视化函数，使用指定的 DEM 文件"""
    print("开始可视化汇水面积计算...")

    print(f"尝试从路径加载 DEM: {dem_path}")
    # 加载 DEM 数据
    dem_tensor = load_dem_from_path(dem_path)
    if dem_tensor is None:
        print("DEM 加载失败，退出。")
        return
    print(f"DEM 张量形状: {dem_tensor.shape}")

    # 设置物理参数 (根据需要调整)
    dx = 10.0
    dy = 10.0
    precip = 1.0 # 假设均匀降水

    # 分别调用汇水面积和坡度计算函数
    try:
        # 1. 计算汇水面积
        print("调用 calculate_drainage_area_ida_dinf_torch...")
        # 使用与 Drainage_area_cal.py 中外部 DEM 相似的参数
        ida_dinf_kwargs = {
            'omega': 0.9,
            'solver_max_iters': 2000, # 增加迭代次数
            'solver_tol': 1e-5,       # 提高精度要求
            'verbose': True           # 启用详细日志以便观察收敛情况
        }
        # 过滤掉 physics.py 中函数不支持的参数
        import inspect
        sig = inspect.signature(calculate_drainage_area_ida_dinf_torch)
        valid_kwargs = {k: v for k, v in ida_dinf_kwargs.items() if k in sig.parameters}
        print(f"使用的 IDA-Dinf 参数: {valid_kwargs}")
        drainage_area = calculate_drainage_area_ida_dinf_torch(
            h=dem_tensor,
            dx=dx,
            dy=dy,
            precip=precip,
            **valid_kwargs # 使用过滤后的参数
        )
        print(f"汇水面积计算完成。形状: {drainage_area.shape}")

        # 2. 计算坡度
        print("调用 calculate_slope_magnitude...")
        slope_mag = calculate_slope_magnitude(
            h=dem_tensor,
            dx=dx,
            dy=dy
        )
        print(f"坡度计算完成。形状: {slope_mag.shape}")

        # 3. 计算河流侵蚀势 (需要 K_f, m, n 参数)
        print("计算河流侵蚀势...")
        K_f_vis = 5e-5 # 示例值
        m_vis = 0.5    # 示例值
        n_vis = 1.0    # 示例值
        erosion_pot = stream_power_erosion(
            h=dem_tensor, # stream_power_erosion 可能不需要 h
            drainage_area=drainage_area,
            slope_magnitude=slope_mag,
            K_f=K_f_vis,
            m=m_vis,
            n=n_vis
        )
        print(f"河流侵蚀势计算完成。形状: {erosion_pot.shape}")

    except NotImplementedError:
         print("错误：汇水面积、坡度或侵蚀势计算函数未实现或未正确导入。")
         return
    except Exception as e:
        print(f"计算过程中发生异常: {e}")
        import traceback
        print("--- Traceback ---")
        traceback.print_exc() # 打印完整的错误追踪信息
        print("--- End Traceback ---")
        return

    # 准备绘图数据 (从 Tensor 转为 Numpy)
    dem_np = dem_tensor.squeeze().cpu().numpy()
    da_np = drainage_area.squeeze().cpu().numpy()
    slope_np = slope_mag.squeeze().cpu().numpy()
    erosion_np = erosion_pot.squeeze().cpu().numpy() # 添加侵蚀率Numpy转换

    # 使用插值提高分辨率（可选）
    print("对数据进行插值以提高分辨率...")
    try:
        # 对所有数据进行插值，提高分辨率
        interp_scale = 2  # 插值缩放因子
        dem_np = interpolate_data(dem_np, interp_scale)
        da_np = interpolate_data(da_np, interp_scale)
        slope_np = interpolate_data(slope_np, interp_scale)
        erosion_np = interpolate_data(erosion_np, interp_scale)
        print(f"插值后的数据形状: {dem_np.shape}")
    except Exception as e:
        print(f"插值失败，使用原始分辨率: {e}")

    # --- 绘图 ---
    print("开始绘图...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12)) # 2x3 布局，共6个子图
    fig.suptitle(f'Drainage Analysis ({os.path.basename(dem_path)})', fontsize=16)

    # --- 子图 1 (0, 0): 输入 DEM ---
    ax = axes[0, 0]
    im0 = ax.imshow(dem_np, cmap='terrain', origin='lower', interpolation='bilinear') # 添加插值
    ax.set_title('Input DEM')
    ax.set_xlabel('X Index')
    ax.set_ylabel('Y Index')
    fig.colorbar(im0, ax=ax, label='Elevation (m)', shrink=0.8)
    ax.axis('equal')

    # --- 子图 2 (0, 1): 汇水面积 (Blues + Alpha + Vmax) ---
    ax = axes[0, 1]
    da_plot = np.maximum(da_np, 0) # 强制非负
    vmax_da = np.percentile(da_plot[da_plot > 0], 99.5) if (da_plot > 0).any() else 1.0 # 设置 vmax
    norm_da = plt.Normalize(vmin=0, vmax=vmax_da)
    # 创建 alpha 通道，面积越大越不透明
    alpha_channel = np.clip(da_plot / vmax_da, 0.1, 1.0) # 基于 vmax 归一化 alpha
    # 获取颜色映射
    cmap_da = plt.cm.Blues # 改用 Blues
    colors = cmap_da(norm_da(da_plot))
    # 应用 alpha
    colors[..., -1] = alpha_channel # 设置 RGBA 中的 A
    ax.imshow(colors, origin='lower', interpolation='bilinear') # 添加插值
    ax.set_title('Drainage Area (Alpha Blend)')
    ax.set_xlabel('X Index')
    ax.set_ylabel('Y Index')
    # 添加颜色条 (需要手动创建 ScalarMappable)
    sm = plt.cm.ScalarMappable(cmap=cmap_da, norm=norm_da)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Contributing Area (grid cells)', shrink=0.8)
    ax.axis('equal')

    # --- 子图 3 (1, 0): 河流网络 (连续颜色 + 更高阈值) ---
    ax = axes[1, 0]
    river_threshold_percentile = 98 # 提高阈值
    if da_plot.size > 0 and (da_plot > 0).any():
        river_threshold = np.percentile(da_plot[da_plot > 0], river_threshold_percentile)
    else:
        river_threshold = 0
    print(f"河流网络阈值 ({river_threshold_percentile}%): {river_threshold:.1f}")
    river_mask = da_plot > river_threshold
    # 绘制 DEM 背景
    ax.imshow(dem_np, cmap='Greys', alpha=0.5, origin='lower')
    # 绘制高于阈值的汇水面积 (使用 Blues colormap 和线性 norm)
    cmap_rivers = plt.cm.Blues
    cmap_rivers.set_under('none') # 低于阈值的部分透明
    norm_rivers = plt.Normalize(vmin=river_threshold, vmax=vmax_da) # 使用之前计算的 vmax_da
    im_rivers = ax.imshow(np.ma.masked_where(~river_mask, da_plot),
                          cmap=cmap_rivers, norm=norm_rivers,
                          interpolation='none', origin='lower') # 河流网络保持 none 插值
    ax.set_title(f'River Network (> {river_threshold_percentile}th Percentile)')
    ax.set_xlabel('X Index')
    ax.set_ylabel('Y Index')
    fig.colorbar(im_rivers, ax=ax, label='Drainage Area', shrink=0.8)
    ax.axis('equal')

    # --- 子图 4 (1, 1): 坡度 ---
    ax = axes[1, 1]
    vmax_slope = np.percentile(slope_np, 99.5) if slope_np.size > 0 else 1.0
    im_slope = ax.imshow(slope_np, cmap='viridis', origin='lower', vmin=0, vmax=vmax_slope, interpolation='bilinear') # 添加插值
    ax.set_title('Slope Magnitude')
    ax.set_xlabel('X Index')
    ax.set_ylabel('Y Index')
    fig.colorbar(im_slope, ax=ax, label='Slope Magnitude', shrink=0.8) # 更清晰的标签
    ax.axis('equal')

    # --- 子图 5 (0, 2): SPL 河流侵蚀率 ---
    ax = axes[0, 2]
    erosion_plot = np.maximum(erosion_np, 0) # 避免负值
    vmax_erosion = np.percentile(erosion_plot[erosion_plot > 0], 99.5) if (erosion_plot > 0).any() else 1.0
    im_erosion = ax.imshow(erosion_plot, cmap='plasma', origin='lower', vmin=0, vmax=vmax_erosion, interpolation='bilinear')
    ax.set_title('SPL Erosion Rate')
    ax.set_xlabel('X Index')
    ax.set_ylabel('Y Index')
    fig.colorbar(im_erosion, ax=ax, label='Erosion Rate (m/yr)', shrink=0.8)
    ax.axis('equal')

    # --- 子图 6 (1, 2): 侵蚀率与河流网络叠加 ---
    ax = axes[1, 2]
    # 绘制DEM背景
    ax.imshow(dem_np, cmap='Greys', alpha=0.5, origin='lower', interpolation='bilinear')
    # 绘制侵蚀率热图，使用透明度
    erosion_norm = plt.Normalize(vmin=0, vmax=vmax_erosion)
    erosion_rgba = plt.cm.plasma(erosion_norm(erosion_plot))
    # 设置透明度，侵蚀率越大越不透明
    erosion_rgba[..., -1] = np.clip(erosion_plot / vmax_erosion, 0.1, 0.8)
    ax.imshow(erosion_rgba, origin='lower', interpolation='bilinear')
    # 叠加河流网络
    ax.imshow(np.ma.masked_where(~river_mask, da_plot),
              cmap=cmap_rivers, norm=norm_rivers,
              interpolation='none', origin='lower', alpha=0.7)
    ax.set_title('Erosion Rate + River Network')
    ax.set_xlabel('X Index')
    ax.set_ylabel('Y Index')
    ax.axis('equal')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局

    # 保存图像
    # 使用输入 DEM 文件名生成简化的输出文件名
    dem_filename = os.path.basename(dem_path)
    dem_name = os.path.splitext(dem_filename)[0]
    # 进一步简化文件名，如果名称太长
    if len(dem_name) > 20:
        dem_name = dem_name[:20]
    output_filename = f"drainage_{dem_name}.png" # 简化文件名
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    try:
        plt.savefig(output_path, dpi=300)  # 确认DPI为300
        print(f"可视化结果已保存到: {output_path}")
    except Exception as e:
        print(f"保存图像时出错: {e}")

    # 显示图像 (已禁用，使用非GUI后端)
    # plt.show()  # 使用Agg后端时不能调用show()

    print("可视化完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化汇水面积计算。")
    parser.add_argument('--dem', type=str, required=True, help='输入 DEM 文件的路径 (.npy)。')
    # 可以添加其他参数，如 dx, dy, precip 等
    # parser.add_argument('--dx', type=float, default=10.0, help='网格间距 X')
    # parser.add_argument('--dy', type=float, default=10.0, help='网格间距 Y')
    # parser.add_argument('--precip', type=float, default=1.0, help='降水率')
    args = parser.parse_args()
    print(f"命令行参数已解析: DEM路径 = {args.dem}")

    visualize_drainage_area(args.dem)