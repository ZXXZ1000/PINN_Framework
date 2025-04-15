# PINN Framework

## 1. 概述

`PINN_Framework` 是一个基于物理信息神经网络 (Physics-Informed Neural Network, PINN) 的地貌演化模拟框架。它是从 `PINN_Fastscape_Framework` 精简而来，旨在提供一个更简洁、聚焦于单一主线实现的代码库。

该框架的核心目标是：

*   **模拟地貌演化**: 使用 PINN 近似求解地貌演化的控制偏微分方程 (PDE)。
*   **可微分计算**: 实现端到端可微分，支持基于梯度的参数反演（例如，推断抬升率场）。
*   **简洁高效**: 移除冗余组件，专注于一条经过验证的核心实现路径，提高代码可维护性和易用性。

**主线实现:**

*   **模型**: `AdaptiveFastscapePINN` (位于 `src/models.py`)，支持多分辨率处理和双输出（状态和导数）。
*   **物理计算**:
    *   汇水面积: 基于 IDA (Iterative Drainage Area) 和 D∞ (D-infinity) 的可微计算方法 (位于 `src/physics.py`)。
    *   其他: 包括坡度、拉普拉斯算子、河流侵蚀和坡面扩散的可微实现。
*   **损失函数**:
    *   PDE 残差: `compute_pde_residual_dual_output`，利用模型的双输出来计算物理损失 (位于 `src/losses.py`)。
    *   其他: 包括数据损失 (`compute_data_loss`) 和平滑度惩罚 (`compute_smoothness_penalty`)。
*   **训练器**: 简化的 `PINNTrainer` (位于 `src/trainer.py`)，专注于双输出模型的训练流程。
*   **优化器**: 基于 PyTorch 的参数优化器 (`ParameterOptimizer` 位于 `src/optimizer_utils.py`)，用于反演问题。

## 2. 目录结构

```
PINN_Framework/
├── configs/                # 配置文件 (YAML)
│   ├── data_gen_config.yaml   # 数据生成参数
│   ├── train_config.yaml      # 模型训练参数
│   └── optimize_config.yaml   # 参数优化/反演参数
├── data/                   # 数据存储目录 (建议结构)
│   ├── processed/          # 处理后的训练/验证数据 (.pt 文件)
│   │   └── resolution_HxW/ # 按分辨率组织的子目录 (可选)
│   └── observations/       # 观测数据 (例如，目标 DEM .npy 文件)
├── scripts/                # 可执行脚本
│   ├── generate_data.py    # 生成模拟数据 (使用 fastscape/xsimlab)
│   ├── train.py            # 训练 PINN 模型
│   └── optimize.py         # 运行参数优化/反演
├── src/                    # 框架源代码
│   ├── __init__.py
│   ├── data_utils.py       # 数据集和数据加载器
│   ├── losses.py           # 损失函数 (数据, 双输出PDE, 平滑度)
│   ├── models.py           # PINN 模型架构 (AdaptiveFastscapePINN)
│   ├── optimizer_utils.py  # 参数优化工具
│   ├── physics.py          # 物理计算 (导数, IDA/D∞汇水面积, PDE项)
│   ├── trainer.py          # 训练循环管理器
│   └── utils.py            # 通用辅助函数 (日志, 配置, 设备等)
├── tests/                  # 测试代码 (待补充)
│   └── __init__.py
├── environment.yml         # Conda 环境依赖文件
├── requirements.txt        # pip 开发/测试依赖文件
└── README.md               # 本文件
```

## 3. 安装

建议使用 Conda 来管理环境，以确保所有依赖项（包括 `fastscape` 及其 Fortran 依赖）正确安装。

1.  **安装 Conda**: 如果尚未安装，请从 [Anaconda](https://www.anaconda.com/products/distribution) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 官网下载并安装。
2.  **创建环境**: 打开终端（Anaconda Prompt, PowerShell, bash 等），导航到 `PINN_Framework` 根目录，然后运行：
    ```bash
    conda env create -f environment.yml
    ```
    这将创建一个名为 `pinn-framework-env` 的新环境并安装所有核心依赖。
3.  **激活环境**: 每次运行代码前，激活环境：
    ```bash
    conda activate pinn-framework-env
    ```
4.  **(可选) 安装开发/测试工具**: 如果需要运行测试或进行开发，在激活环境后，使用 pip 安装 `requirements.txt` 中的工具：
    ```bash
    pip install -r requirements.txt
    ```

## 4. 使用方法

框架的使用主要通过 `scripts/` 目录下的脚本进行，并通过 `configs/` 目录下的 YAML 文件进行配置。

1.  **配置**:
    *   修改 `configs/` 目录下的 `.yaml` 文件以设置您的实验参数。
    *   `data_gen_config.yaml`: 配置数据生成参数，如样本数量、分辨率、参数范围、输出目录等。
    *   `train_config.yaml`: 配置模型训练参数，如数据路径、模型超参数、优化器、学习率、损失权重、训练轮数、检查点路径等。
    *   `optimize_config.yaml`: 配置参数优化（反演）任务，如训练好的模型路径、观测数据路径、要优化的参数及其初始猜测/边界、优化器设置等。
    *   **注意**: 配置文件支持使用 `${...}` 语法进行变量插值（需要安装 `omegaconf`）。

2.  **生成数据**:
    *   配置好 `configs/data_gen_config.yaml`。
    *   运行脚本：
        ```bash
        conda activate pinn-framework-env
        python scripts/generate_data.py --config configs/data_gen_config.yaml
        ```
    *   生成的数据将保存在 `data_gen_config.yaml` 中指定的 `base_output_dir` 下的对应分辨率子目录中。

3.  **训练模型**:
    *   确保已生成训练数据。
    *   配置好 `configs/train_config.yaml`，特别是数据路径和模型/训练参数。
    *   运行脚本：
        ```bash
        conda activate pinn-framework-env
        python scripts/train.py --config configs/train_config.yaml
        ```
    *   训练日志、TensorBoard 文件和模型检查点将保存在 `train_config.yaml` 中指定的 `output_dir`/`run_name` 下。

4.  **参数优化 (反演)**:
    *   确保已训练好模型并准备好观测数据。
    *   配置好 `configs/optimize_config.yaml`，特别是模型检查点路径、观测数据路径、要优化的参数和固定参数。
    *   运行脚本：
        ```bash
        conda activate pinn-framework-env
        python scripts/optimize.py --config configs/optimize_config.yaml
        ```
    *   优化结果（例如，推断出的参数场）和日志将保存在 `optimize_config.yaml` 中指定的 `output_dir`/`run_name` 下。

## 5. 核心组件

*   **`src/models.py`**: 定义了 `AdaptiveFastscapePINN`，这是框架的核心预测模型。
*   **`src/physics.py`**: 实现了地貌演化 PDE 中的关键物理过程的可微版本，包括使用 IDA/D∞ 方法计算汇水面积。
*   **`src/losses.py`**: 实现了用于训练 PINN 的损失函数，包括数据损失、基于双输出的 PDE 残差损失和平滑度惩罚。
*   **`src/trainer.py`**: 包含 `PINNTrainer` 类，管理训练和验证循环。
*   **`src/optimizer_utils.py`**: 包含 `ParameterOptimizer` 类和 `optimize_parameters` 函数，用于执行参数反演。
*   **`src/data_utils.py`**: 定义了 `FastscapeDataset` 类和 `create_dataloaders` 函数，用于加载和处理模拟数据。
*   **`src/utils.py`**: 包含日志记录、配置加载、设备管理等通用辅助函数。

## 6. 后续工作

*   **测试**: 补充 `tests/` 目录下的单元测试和集成测试，以确保代码的正确性和稳定性。
*   **文档**: 为各个模块和函数添加更详细的文档字符串。
*   **示例**: 提供更具体的示例配置文件和运行说明。