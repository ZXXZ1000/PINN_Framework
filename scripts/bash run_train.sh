#!/bin/bash

# 1. 检查conda是否安装
if ! command -v conda &> /dev/null
then
    echo "Conda 未检测到，正在下载安装 Miniconda..."
    # 这里以Windows为例，Linux用户请替换为合适的Miniconda安装包
    curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
    ./Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%USERPROFILE%\Miniconda3
    export PATH="$HOME/Miniconda3/Scripts:$PATH"
fi

# 2. 创建并激活环境
conda env list | grep pinn-framework-env
if [ $? -ne 0 ]; then
    echo "创建conda环境 pinn-framework-env ..."
    conda env create -f environment.yml -n pinn-framework-env
fi
echo "激活环境..."
conda activate pinn-framework-env

# 3. 安装Python依赖（如有requirements.txt）
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

# 4. 生成数据
echo "开始生成训练数据..."
python scripts/generate_data.py --config configs\data_gen_config.yaml
if [ $? -ne 0 ]; then
    echo "数据生成失败，终止流程。"
    exit 1
fi

# 5. 启动训练
echo "开始训练..."
python scripts/train.py --config configs/train_config_4090.yaml