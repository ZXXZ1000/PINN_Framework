# PINN_Framework 安装指南

这份安装说明只覆盖“把项目带到一台新宿主机后如何跑起来”。完整训练流程见 [RUNBOOK.md](RUNBOOK.md)。

## 推荐方式

在仓库根目录运行：

```bash
bash scripts/setup_environment.sh
conda activate pinn-framework-env
```

这个脚本会：

```text
1. 根据 environment.yml 创建或更新 pinn-framework-env
2. 运行 scripts/check_environment.py 检查核心依赖
3. 默认运行一次 operator smoke training
```

如果只想装环境，不想跑 smoke training：

```bash
PINN_SKIP_SMOKE=1 bash scripts/setup_environment.sh
conda activate pinn-framework-env
python scripts/check_environment.py
```

## 手动安装

```bash
conda env create -f environment.yml
conda activate pinn-framework-env
python scripts/check_environment.py --smoke
```

如果环境已存在：

```bash
conda env update -n pinn-framework-env -f environment.yml --prune
conda activate pinn-framework-env
python scripts/check_environment.py --smoke
```

## 验收

环境和代码基础验收：

```bash
python scripts/check_environment.py --smoke
pytest -q
```

通过时应看到：

```text
Environment check passed.
operator training verification passed.
pytest: all tests passed
```

## 训练入口

生成数据：

```bash
python scripts/generate_data.py --config configs/data_gen_config.yaml
```

训练默认 LandscapeNeuralOperator：

```bash
python scripts/train_operator.py --config configs/train_operator_config.yaml
```

训练输出默认在：

```text
results/<run_name>/
  checkpoints/
  logs/
  tensorboard/
```

不应该输出到：

```text
PINN_Framework/results/
```

## 验证入口

单步验证：

```bash
python scripts/evaluate_operator.py \
  --config configs/train_operator_config.yaml \
  --checkpoint results/<run_name>/checkpoints/best_model.pth \
  --split val \
  --output results/<run_name>/evaluation_val.json
```

多步 rollout 验证：

```bash
python scripts/evaluate_rollout.py \
  --config configs/train_operator_config.yaml \
  --checkpoint results/<run_name>/checkpoints/best_model.pth \
  --split val \
  --steps 10 \
  --output results/<run_name>/rollout_val.json
```

## 常见问题

### 默认 python 没有 torch

请先激活 conda 环境：

```bash
conda activate pinn-framework-env
python scripts/check_environment.py
```

### Fastscape/xsimlab 导入失败

通常是 `distributed`、`xsimlab`、`fastscape` 或 `netCDF4` 缺失。运行：

```bash
python scripts/check_environment.py
conda env update -n pinn-framework-env -f environment.yml --prune
```

### Mac 可以跑吗

Mac 可以跑 smoke test 和小规模调试。正式大规模训练建议用 4090 或服务器。
