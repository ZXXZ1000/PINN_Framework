"""
Train the LandscapeNeuralOperator with the current PDE residual.
"""

import argparse
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from train import main as train_main


if __name__ == "__main__":
    default_config = os.path.join(project_root, "configs", "train_operator_config.yaml")
    parser = argparse.ArgumentParser(description="训练 LandscapeNeuralOperator step operator。")
    parser.add_argument("--config", type=str, default=default_config, help="配置文件路径。")
    args = parser.parse_args()
    train_main(args)
