# PINN_Framework 安装指南

本文档提供了安装和设置 PINN_Framework 环境的详细步骤。

## 系统要求

- Python 3.10 或更高版本
- Conda 包管理器 (推荐使用 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 [Anaconda](https://www.anaconda.com/download/))
- 对于 GPU 加速 (可选): CUDA 兼容的 NVIDIA GPU

## 安装步骤

### 1. 克隆代码库

```bash
git clone <repository-url>
cd PINN_Framework
```

### 2. 创建并激活 Conda 环境

我们提供了 `environment.yml` 文件，其中包含了所有必要的依赖项。

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate pinn-framework-env
```

### 3. 安装额外的开发/测试依赖项 (可选)

如果你需要进行开发或运行测试，可以安装额外的依赖项：

```bash
pip install -r requirements.txt
```

### 4. 验证安装

运行以下命令验证环境是否正确设置：

```bash
# 检查 PyTorch 是否可用
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# 检查 Fastscape 是否可用
python -c "import fastscape; print(f'Fastscape version: {fastscape.__version__}')"

# 检查 xarray 和 xsimlab 是否可用
python -c "import xarray as xr, xsimlab as xs; print(f'xarray version: {xr.__version__}, xsimlab version: {xs.__version__}')"
```

## GPU 支持 (可选)

默认情况下，环境配置为使用 CPU 版本的 PyTorch。如果你想使用 GPU 加速，请按照以下步骤操作：

1. 编辑 `environment.yml` 文件：
   - 删除或注释掉 `- cpuonly` 行
   - 取消注释 `# - pytorch-cuda=11.8` 行，并根据你的 CUDA 版本进行调整

2. 创建新的环境或更新现有环境：
   ```bash
   # 创建新环境
   conda env create -f environment.yml
   
   # 或更新现有环境
   conda env update -f environment.yml
   ```

3. 验证 GPU 是否可用：
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"
   ```

## 常见问题

### Fastscape 安装问题

Fastscape 依赖于 Fortran 编译器。如果安装过程中遇到与 Fastscape 相关的错误，请确保系统上安装了适当的 Fortran 编译器：

- **Windows**: 安装 MinGW-w64 with gfortran
- **Linux**: 安装 gfortran (`sudo apt-get install gfortran` 或类似命令)
- **macOS**: 使用 Homebrew 安装 gfortran (`brew install gcc`)

### PyTorch 版本问题

如果你需要特定版本的 PyTorch，可以在创建环境后手动安装：

```bash
# 激活环境
conda activate pinn-framework-env

# 安装特定版本的 PyTorch (示例)
pip install torch==2.0.0 torchvision==0.15.0
```

## 其他资源

- [PyTorch 安装指南](https://pytorch.org/get-started/locally/)
- [Fastscape 文档](https://fastscape.readthedocs.io/)
- [xarray 文档](https://docs.xarray.dev/en/stable/)
- [xsimlab 文档](https://xsimlab.readthedocs.io/)
