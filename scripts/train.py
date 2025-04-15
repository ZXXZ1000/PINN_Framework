# PINN_Framework/scripts/train.py
"""
训练 PINN 模型的主脚本。
"""

import argparse
import logging
import os
import sys
import torch
from typing import Dict, Optional, Tuple, Any, Union, List

# 将项目根目录添加到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # scripts 目录的上级是项目根目录
if project_root not in sys.path:
    sys.path.append(project_root)

# 从新框架导入必要的模块
try:
    # 假设 utils, data_utils, models, trainer 都在 src 目录下
    from src.utils import load_config, setup_logging, set_seed
    from src.data_utils import create_dataloaders
    from src.models import AdaptiveFastscapePINN # 直接导入主线模型
    from src.trainer import PINNTrainer # 导入简化的训练器
except ImportError as e:
    print(f"错误：无法导入必要的模块: {e}。请确保你在 PINN_Framework 目录下运行，并且 src 目录包含所需文件。")
    sys.exit(1)

def main(args):
    """主函数，用于训练 PINN 模型。"""
    # --- 配置和日志设置 ---
    try:
        config = load_config(args.config) # 使用 utils 中的加载函数 (返回 OmegaConf 对象)
    except Exception as e:
        print(f"错误：无法加载配置文件 {args.config}: {e}")
        sys.exit(1)

    # 使用 OmegaConf 访问配置，提供默认值
    try:
        from omegaconf import OmegaConf
        if not isinstance(config, OmegaConf): config = OmegaConf.create(config)
    except ImportError:
        logging.warning("OmegaConf 未安装，使用标准字典访问。")

    train_config = config.get('training', {})
    output_dir = config.get('output_dir', 'results/')
    run_name = train_config.get('run_name', 'pinn_run') # Trainer 会处理默认值
    log_dir = os.path.join(output_dir, run_name, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'train.log')
    log_level = config.get('logging', {}).get('log_level', 'INFO')
    setup_logging(log_level=log_level, log_file=log_file_path, log_to_console=True)

    # --- 设置随机种子 ---
    seed = train_config.get('seed', 42)
    set_seed(seed)
    logging.info(f"随机种子设置为 {seed}")

    # --- 创建数据加载器 ---
    try:
        # create_dataloaders 返回包含 'train', 'val', 'test', 'norm_stats' 的字典
        dataloaders_dict = create_dataloaders(config) # 传递 OmegaConf 对象或字典
        train_loader = dataloaders_dict.get('train')
        val_loader = dataloaders_dict.get('val')
        # test_loader = dataloaders_dict.get('test') # 可以获取测试加载器以备后用
        # norm_stats = dataloaders_dict.get('norm_stats') # 可以获取归一化统计数据
        if train_loader is None:
             raise ValueError("训练数据加载器未能创建。")
    except Exception as e:
         logging.error(f"创建数据加载器失败: {e}", exc_info=True)
         sys.exit(1)

    # --- 初始化模型 (固定为 AdaptiveFastscapePINN) ---
    model_config = config.get('model', {}).copy() # 获取模型特定配置
    # 移除不再需要的 'name' 字段（如果存在）
    model_config.pop('name', None)
    model_dtype_str = model_config.pop('dtype', 'float32')
    model_dtype = torch.float32 if model_dtype_str == 'float32' else torch.float64

    # AdaptiveFastscapePINN 需要 domain_x 和 domain_y
    # 尝试从物理参数或数据配置中获取
    physics_params = config.get('physics', {})
    data_params = config.get('data', {})
    # OmegaConf 插值应该已经解析了这些值
    domain_x = physics_params.get('domain_x', data_params.get('domain_x'))
    domain_y = physics_params.get('domain_y', data_params.get('domain_y'))

    if domain_x is None or domain_y is None:
         logging.warning("无法从配置中找到 'domain_x' 或 'domain_y'。模型可能使用默认值 [0, 1]。")
         # 提供默认值以防万一
         model_config['domain_x'] = model_config.get('domain_x', [0.0, 1.0])
         model_config['domain_y'] = model_config.get('domain_y', [0.0, 1.0])
    else:
         # 确保是列表或元组
         model_config['domain_x'] = list(domain_x) if isinstance(domain_x, (list, tuple)) else [0.0, float(domain_x)]
         model_config['domain_y'] = list(domain_y) if isinstance(domain_y, (list, tuple)) else [0.0, float(domain_y)]


    try:
        # 使用 **model_config 将剩余参数传递给模型构造函数
        # 确保传递给模型的配置是标准字典
        model_config_dict = OmegaConf.to_container(model_config, resolve=True) if isinstance(model_config, OmegaConf) else model_config
        model = AdaptiveFastscapePINN(**model_config_dict).to(dtype=model_dtype)
        logging.info(f"已初始化模型: AdaptiveFastscapePINN")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"模型可训练参数数量: {num_params:,}")
    except Exception as e:
        logging.error(f"初始化 AdaptiveFastscapePINN 模型失败: {e}", exc_info=True)
        logging.error(f"使用的模型配置: {model_config_dict if 'model_config_dict' in locals() else model_config}")
        sys.exit(1)

    # --- 初始化训练器 ---
    try:
        # 将完整配置传递给训练器，它会提取所需部分
        trainer = PINNTrainer(model, config, train_loader, val_loader) # Pass OmegaConf object or dict
        logging.info("训练器初始化完成。")
    except Exception as e:
        logging.error(f"初始化 PINNTrainer 失败: {e}", exc_info=True)
        sys.exit(1)

    # --- 开始训练 ---
    logging.info("开始训练流程...")
    try:
        trainer.train()
        logging.info("训练流程完成。")
    except Exception as e:
        logging.error(f"训练过程中发生错误: {e}", exc_info=True)
        sys.exit(1)

    # --- 可选：训练后在测试集上评估 ---
    # run_test = train_config.get('run_test_evaluation', False)
    # test_loader = dataloaders_dict.get('test') # 获取测试加载器
    # if run_test and test_loader is not None:
    #     logging.info("在测试集上运行评估...")
    #     # TODO: 实现评估逻辑 (可能在 trainer 中添加 evaluate 方法)
    #     # test_loss, test_metrics = trainer.evaluate(test_loader)
    #     # logging.info(f"测试损失: {test_loss:.4f}, 测试指标: {test_metrics}")
    #     logging.warning("测试集评估逻辑尚未实现。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 AdaptiveFastscapePINN 模型。")
    parser.add_argument('--config', type=str, required=True, help='配置文件的路径。')
    args = parser.parse_args()
    main(args)