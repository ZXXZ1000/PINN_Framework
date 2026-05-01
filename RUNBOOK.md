# PINN_Framework 接手运行手册

这份文档是给接手人用的最短路径。目标是从一台新宿主机进入项目后，能快速判断环境是否可用、如何生成数据、如何训练、如何验证模型是否真的学到了东西。

## 0. 当前项目状态

当前默认训练目标是：

```text
Fastscape/xsimlab 离线生成 teacher trajectory
        ->
LandscapeNeuralOperator 学习 h_t -> h_{t+dt}
        ->
data loss + increment loss + derivative loss + soft_mfd physics residual
        ->
one-step validation + multi-step rollout validation
```

默认模型不是旧 PINN，也不是纯 CNN，而是：

```text
LandscapeNeuralOperator = UNO 编解码骨架 + FNO 频域层 + soft flow-graph 水文分支
```

默认配置文件：

```text
configs/train_operator_config.yaml
```

默认输出目录：

```text
results/<run_name>/
  checkpoints/
  tensorboard/
  logs/
```

正常情况下不应该生成：

```text
PINN_Framework/results/
```

如果看到这个目录，通常是旧测试配置或错误的相对路径造成的。

## 1. 环境配置

推荐在宿主机上直接运行：

```bash
bash scripts/setup_environment.sh
conda activate pinn-framework-env
```

如果只想检查环境，不想跑 smoke training：

```bash
PINN_SKIP_SMOKE=1 bash scripts/setup_environment.sh
conda activate pinn-framework-env
python scripts/check_environment.py
```

完整自检：

```bash
python scripts/check_environment.py --smoke
pytest -q
```

通过标准：

```text
Environment check passed.
operator training verification passed.
pytest: all tests passed
```

## 2. 数据生成

训练需要 `.pt` 数据。默认数据生成配置是：

```text
configs/data_gen_config.yaml
```

运行：

```bash
python scripts/generate_data.py --config configs/data_gen_config.yaml
```

输出位置由配置里的 `data_generation.base_output_dir` 控制。建议统一成：

```yaml
data_generation:
  base_output_dir: data/processed
```

生成后应看到类似：

```text
data/processed/resolution_64x64/sample_00000.pt
data/processed/resolution_64x64/sample_00001.pt
```

每个样本最好包含：

```text
initial_topo
final_topo
trajectory_topo
time
dt
uplift_rate
k_f
k_d
m
n
run_time
```

`trajectory_topo` 是 multi-step rollout validation 的关键字段。

## 3. 开始训练

运行默认 neural operator 训练：

```bash
python scripts/train_operator.py --config configs/train_operator_config.yaml
```

训练时重点看：

```text
Train Loss
Val Loss
data_loss
increment_loss
derivative_data_loss
physics_loss
```

不能只看 train loss。train loss 下降但 val loss 不下降，说明可能过拟合或数据分布不对。

## 4. 训练后验证

### 4.1 单步验证

```bash
python scripts/evaluate_operator.py \
  --config configs/train_operator_config.yaml \
  --checkpoint results/<run_name>/checkpoints/best_model.pth \
  --split val \
  --output results/<run_name>/evaluation_val.json
```

重点看：

```text
learned.state_rmse
learned.delta_rmse
learned.derivative_rmse
learned.slope_rmse
learned.physics_loss
improvement.state_rmse_vs_persistence
improvement.state_rmse_vs_physics_euler
```

判断：

```text
learned 比 persistence 好：至少比“不变化”更有用
learned 比 physics_euler 好：operator 才有 surrogate 价值
```

### 4.2 多步 rollout 验证

```bash
python scripts/evaluate_rollout.py \
  --config configs/train_operator_config.yaml \
  --checkpoint results/<run_name>/checkpoints/best_model.pth \
  --split val \
  --steps 10 \
  --output results/<run_name>/rollout_val.json
```

重点看：

```text
per_step["1"].learned.state_rmse
per_step["5"].learned.state_rmse
per_step["10"].learned.state_rmse
per_step["10"].improvement.state_rmse_vs_persistence
per_step["10"].improvement.state_rmse_vs_physics_euler
```

判断：

```text
one-step 好，但 rollout 爆炸：只能短期拟合，不能当模拟器
rollout 多步稳定优于 baseline：模型才算真的学到可泛化动力学
```

## 5. 常见问题

### 默认 python 没有 torch

不要直接用系统默认 `python`。先：

```bash
conda activate pinn-framework-env
python scripts/check_environment.py
```

### Fastscape/xsimlab 导入失败

先查：

```bash
python scripts/check_environment.py
```

常见缺失：

```text
distributed
xsimlab
fastscape
netCDF4
```

解决方式：

```bash
conda env update -n pinn-framework-env -f environment.yml --prune
```

### 训练输出到了 PINN_Framework/results

这是错误路径。正确输出应该是：

```text
results/<run_name>/
```

检查配置：

```yaml
output_dir: results/
training:
  results_dir: results/
```

一般只需要设置顶层 `output_dir`，不要手动写 `PINN_Framework/results`。

### Mac 可以跑吗

Mac 可以跑 smoke test 和小规模训练，但不适合大实验。正式训练建议在 4090 或服务器上跑。

## 6. 最短验收命令

新宿主机接手后，按顺序跑：

```bash
bash scripts/setup_environment.sh
conda activate pinn-framework-env
python scripts/check_environment.py --smoke
pytest -q
python scripts/generate_data.py --config configs/data_gen_config.yaml
python scripts/train_operator.py --config configs/train_operator_config.yaml
```

训练完成后：

```bash
python scripts/evaluate_operator.py \
  --config configs/train_operator_config.yaml \
  --checkpoint results/<run_name>/checkpoints/best_model.pth \
  --split val \
  --output results/<run_name>/evaluation_val.json

python scripts/evaluate_rollout.py \
  --config configs/train_operator_config.yaml \
  --checkpoint results/<run_name>/checkpoints/best_model.pth \
  --split val \
  --steps 10 \
  --output results/<run_name>/rollout_val.json
```
