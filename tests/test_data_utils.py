# PINN_Framework/tests/test_data_utils.py
import pytest
import os
import glob
import json
import logging
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader
from unittest.mock import patch, MagicMock, call, ANY

# 导入被测试的模块
from src import data_utils
from src.data_utils import FastscapeDataset, collate_fn_filter_none, compute_normalization_stats, create_dataloaders

# --- Fixtures ---

@pytest.fixture
def temp_data_dir(tmp_path):
    """创建一个临时数据目录。"""
    data_dir = tmp_path / "processed_data"
    data_dir.mkdir()
    return data_dir

@pytest.fixture
def dummy_pt_files(temp_data_dir):
    """在临时目录中创建一些虚拟的 .pt 数据文件。"""
    files = []
    # 文件 1: 标准数据
    data1 = {
        'initial_topo': torch.rand(10, 10) * 100,
        'final_topo': torch.rand(10, 10) * 110,
        'uplift_rate': torch.tensor(0.001),
        'k_f': np.array(1e-5),
        'k_d': 5e-9,
        'm': 0.5,
        'n': 1.0,
        'run_time': 10000.0
    }
    file1 = temp_data_dir / "sample_01.pt"
    # 兼容新版本 PyTorch
    try:
        torch.save(data1, file1, weights_only=False)  # 显式使用 weights_only=False
    except TypeError:
        torch.save(data1, file1)
    files.append(str(file1))

    # 文件 2: 包含 NumPy 数组和不同值
    data2 = {
        'initial_topo': np.random.rand(10, 10) * 90,
        'final_topo': np.random.rand(10, 10) * 105,
        'uplift_rate': np.array(0.002),
        'k_f': torch.tensor(2e-5),
        'k_d': torch.tensor(6e-9),
        'm': np.float32(0.4),
        'n': 1.1,
        'run_time': 15000.0
    }
    file2 = temp_data_dir / "sample_02.pt"
    # 兼容新版本 PyTorch
    try:
        torch.save(data2, file2, weights_only=False)  # 显式使用 weights_only=False
    except TypeError:
        torch.save(data2, file2)
    files.append(str(file2))

    # 文件 3: 缺少 'k_d' (用于测试错误处理)
    data3 = {
        'initial_topo': torch.rand(10, 10) * 95,
        'final_topo': torch.rand(10, 10) * 100,
        'uplift_rate': 0.0015,
        'k_f': 1.5e-5,
        # 'k_d': missing,
        'm': 0.6,
        'n': 0.9,
        'run_time': 12000.0
    }
    file3 = temp_data_dir / "sample_03_missing.pt"
    # 兼容新版本 PyTorch
    try:
        torch.save(data3, file3, weights_only=False)  # 显式使用 weights_only=False
    except TypeError:
        torch.save(data3, file3)
    files.append(str(file3))

    # 文件 4: 另一个标准数据
    data4 = {
        'initial_topo': torch.rand(10, 10) * 100,
        'final_topo': torch.rand(10, 10) * 110,
        'uplift_rate': torch.tensor(0.001),
        'k_f': np.array(1e-5),
        'k_d': 5e-9,
        'm': 0.5,
        'n': 1.0,
        'run_time': 10000.0
    }
    file4 = temp_data_dir / "sample_04.pt"
    # 兼容新版本 PyTorch
    try:
        torch.save(data4, file4, weights_only=False)  # 显式使用 weights_only=False
    except TypeError:
        torch.save(data4, file4)
    files.append(str(file4))

    return files

@pytest.fixture
def dummy_norm_stats():
    """创建一个虚拟的归一化统计字典。"""
    return {
        'topo': {'min': 0.0, 'max': 110.0},
        'uplift_rate': {'min': 0.0005, 'max': 0.0025},
        'k_f': {'min': 0.5e-5, 'max': 2.5e-5},
        'k_d': {'min': 4e-9, 'max': 7e-9}
    }

@pytest.fixture
def dummy_norm_stats_file(tmp_path, dummy_norm_stats):
    """创建一个包含虚拟归一化统计信息的临时 JSON 文件。"""
    stats_file = tmp_path / "norm_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(dummy_norm_stats, f)
    return str(stats_file)

@pytest.fixture
def dummy_config(temp_data_dir, dummy_norm_stats_file):
    """创建一个用于 create_dataloaders 的虚拟配置字典。"""
    return {
        'data': {
            'processed_dir': str(temp_data_dir),
            'train_split': 0.5, # 2 files
            'val_split': 0.25,  # 1 file
            # test_split is implicitly 0.25 (1 file)
            'num_workers': 0,
            'normalization': {
                'enabled': True,
                'compute_stats': False, # Try loading first
                'stats_file': dummy_norm_stats_file
            }
        },
        'training': {
            'batch_size': 2
        }
    }

# --- 测试 FastscapeDataset ---

def test_fastscape_dataset_len(dummy_pt_files):
    """测试 FastscapeDataset 的 __len__ 方法。"""
    dataset = FastscapeDataset(file_list=dummy_pt_files)
    assert len(dataset) == len(dummy_pt_files)

def test_fastscape_dataset_getitem_basic(dummy_pt_files):
    """测试 FastscapeDataset 的 __getitem__ 基本加载和类型转换。"""
    dataset = FastscapeDataset(file_list=[dummy_pt_files[0]]) # 只加载第一个文件
    sample = dataset[0]

    assert isinstance(sample, dict)
    assert 'initial_topo' in sample and isinstance(sample['initial_topo'], torch.Tensor)
    assert 'final_topo' in sample and isinstance(sample['final_topo'], torch.Tensor)
    assert 'uplift_rate' in sample and isinstance(sample['uplift_rate'], torch.Tensor)
    assert 'k_f' in sample and isinstance(sample['k_f'], torch.Tensor)
    assert 'k_d' in sample and isinstance(sample['k_d'], torch.Tensor)
    assert 'm' in sample and isinstance(sample['m'], torch.Tensor)
    assert 'n' in sample and isinstance(sample['n'], torch.Tensor)
    assert 'run_time' in sample and isinstance(sample['run_time'], torch.Tensor)
    assert 'target_shape' in sample and sample['target_shape'] == (10, 10)

    # 检查数据类型是否为 float32
    for key in ['initial_topo', 'final_topo', 'uplift_rate', 'k_f', 'k_d', 'm', 'n', 'run_time']:
        assert sample[key].dtype == torch.float32

def test_fastscape_dataset_getitem_missing_key(dummy_pt_files, caplog):
    """测试 __getitem__ 在缺少键时返回 None 并记录错误。"""
    dataset = FastscapeDataset(file_list=[dummy_pt_files[2]]) # 加载缺少 k_d 的文件
    with caplog.at_level(logging.WARNING): # 改为 WARNING 级别，因为我们的代码现在使用 WARNING 而不是 ERROR
        sample = dataset[0]

    assert sample is None
    assert "Missing required data fields" in caplog.text
    assert "k_d" in caplog.text

@patch('torch.load', side_effect=Exception("Corrupted file"))
def test_fastscape_dataset_getitem_load_error(mock_load, dummy_pt_files, caplog):
    """测试 __getitem__ 在 torch.load 失败时返回 None 并记录错误。"""
    dataset = FastscapeDataset(file_list=[dummy_pt_files[0]])
    with caplog.at_level(logging.ERROR): # 保持 ERROR 级别，因为我们的代码仍然使用 ERROR 来记录加载错误
        sample = dataset[0]

    assert sample is None
    assert "Error loading/processing sample" in caplog.text
    assert "Corrupted file" in caplog.text
    # 兼容新版本 PyTorch
    # 检查是否调用了 torch.load，但不检查特定参数
    mock_load.assert_called_once()

def test_fastscape_dataset_getitem_normalization(dummy_pt_files, dummy_norm_stats):
    """测试 __getitem__ 应用归一化。"""
    dataset = FastscapeDataset(file_list=[dummy_pt_files[0]], normalize=True, norm_stats=dummy_norm_stats)
    sample = dataset[0]

    assert isinstance(sample, dict)
    # 检查被归一化的字段值是否在 [0, 1] 范围内（或接近，考虑 epsilon）
    epsilon = dataset.epsilon
    # 直接定义哪些字段应该被归一化及其对应的统计键
    fields_to_check = {
        'initial_topo': 'topo',
        'final_topo': 'topo',
        'uplift_rate': 'uplift_rate',
        'k_f': 'k_f',
        'k_d': 'k_d',
    }
    for field, stats_key in fields_to_check.items():
         if field in sample:
             tensor = sample[field]
             stats = dummy_norm_stats[stats_key]
             min_val, max_val = stats['min'], stats['max']
             # 检查是否大致在 [0, 1] 范围内
             assert torch.all(tensor >= (0.0 - epsilon)), f"{field} 的最小值 {tensor.min().item()} 小于 0"
             assert torch.all(tensor <= (1.0 + epsilon * 2)), f"{field} 的最大值 {tensor.max().item()} 大于 1"
             # 抽样检查一个点是否正确归一化
             # 直接使用 weights_only=False，因为我们的数据包含 numpy 数组
             original_value = torch.load(dummy_pt_files[0], weights_only=False)[field]
             original_tensor = dataset._to_float_tensor(original_value)
             expected_normalized = (original_tensor - min_val) / (max_val - min_val + epsilon)
             assert torch.allclose(tensor, expected_normalized, atol=1e-6), f"{field} 归一化值与预期不符"


def test_fastscape_dataset_getitem_no_normalization(dummy_pt_files, dummy_norm_stats):
    """测试 __getitem__ 在 normalize=False 时不应用归一化。"""
    dataset = FastscapeDataset(file_list=[dummy_pt_files[0]], normalize=False, norm_stats=dummy_norm_stats)
    sample = dataset[0]
    # 直接使用 weights_only=False，因为我们的数据包含 numpy 数组
    original_data = torch.load(dummy_pt_files[0], weights_only=False)

    # 检查值是否与原始值（转换为 float32 张量后）相同
    assert torch.allclose(sample['initial_topo'], dataset._to_float_tensor(original_data['initial_topo']))
    assert torch.allclose(sample['uplift_rate'], dataset._to_float_tensor(original_data['uplift_rate']))

def test_fastscape_dataset_denormalize(dummy_pt_files, dummy_norm_stats):
    """测试 denormalize_state 方法。"""
    dataset = FastscapeDataset(file_list=[dummy_pt_files[0]], normalize=True, norm_stats=dummy_norm_stats)
    sample = dataset[0]
    normalized_topo = sample['final_topo']

    denormalized_topo = dataset.denormalize_state(normalized_topo)

    # 直接使用 weights_only=False，因为我们的数据包含 numpy 数组
    original_topo = dataset._to_float_tensor(torch.load(dummy_pt_files[0], weights_only=False)['final_topo'])

    # 检查反归一化后的值是否接近原始值
    assert torch.allclose(denormalized_topo, original_topo, atol=1e-5)

def test_fastscape_dataset_denormalize_no_stats(dummy_pt_files, caplog):
    """测试 denormalize_state 在缺少统计信息时返回原张量并记录警告。"""
    # 提供部分统计信息，缺少 'topo'
    partial_stats = {'uplift_rate': {'min': 0, 'max': 1}}
    dataset = FastscapeDataset(file_list=[dummy_pt_files[0]], normalize=True, norm_stats=partial_stats)
    normalized_tensor = torch.rand(5, 5)

    with caplog.at_level(logging.WARNING):
        denormalized_tensor = dataset.denormalize_state(normalized_tensor)

    assert torch.equal(denormalized_tensor, normalized_tensor)
    assert "Normalization stats for 'topo' missing. Cannot denormalize state." in caplog.text

def test_fastscape_dataset_transform(dummy_pt_files):
    """测试 FastscapeDataset 应用 transform。"""
    mock_transform = MagicMock(return_value={'transformed': True})
    dataset = FastscapeDataset(file_list=[dummy_pt_files[0]], transform=mock_transform)
    sample = dataset[0]

    mock_transform.assert_called_once()
    # 检查 transform 是否被调用，并且返回的是 transform 的结果
    assert sample == {'transformed': True}

# --- 测试 collate_fn_filter_none ---

def test_collate_fn_filter_none_all_valid():
    """测试 collate_fn 处理所有有效样本的批次。"""
    batch = [{'a': torch.tensor([1, 2]), 'b': torch.tensor(1)},
             {'a': torch.tensor([3, 4]), 'b': torch.tensor(2)}]
    collated = collate_fn_filter_none(batch)
    assert isinstance(collated, dict)
    assert torch.equal(collated['a'], torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(collated['b'], torch.tensor([1, 2]))

def test_collate_fn_filter_none_with_nones():
    """测试 collate_fn 过滤掉 None 样本。"""
    batch = [{'a': torch.tensor([1, 2])}, None, {'a': torch.tensor([3, 4])}, None]
    collated = collate_fn_filter_none(batch)
    assert isinstance(collated, dict)
    assert torch.equal(collated['a'], torch.tensor([[1, 2], [3, 4]]))

def test_collate_fn_filter_none_all_nones():
    """测试 collate_fn 处理所有样本都为 None 的批次。"""
    batch = [None, None, None]
    collated = collate_fn_filter_none(batch)
    assert collated is None

@patch('torch.utils.data.dataloader.default_collate', side_effect=RuntimeError("Collate error"))
def test_collate_fn_filter_none_collate_error(mock_default_collate, caplog):
    """测试 collate_fn 在 default_collate 出错时返回 None 并记录错误。"""
    batch = [{'a': torch.tensor(1)}, {'a': torch.tensor(2)}] # 有效批次
    with caplog.at_level(logging.WARNING): # 改为 WARNING 级别
        collated = collate_fn_filter_none(batch)

    assert collated is None
    assert "RuntimeError during collation" in caplog.text
    # 我们的新代码不再输出第一个样本的键，所以删除这个断言
    mock_default_collate.assert_called_once_with(batch)


# --- 测试 compute_normalization_stats ---

def test_compute_normalization_stats_success(dummy_pt_files):
    """测试 compute_normalization_stats 成功计算统计信息。"""
    # 使用前两个有效文件
    valid_files = [f for f in dummy_pt_files if 'missing' not in f]
    fields = ['topo', 'uplift_rate', 'k_f', 'k_d']
    stats = compute_normalization_stats(valid_files, fields)

    assert isinstance(stats, dict)
    assert set(stats.keys()) == set(fields)

    # 检查 topo (合并 initial 和 final)
    # 直接使用 weights_only=False，因为我们的数据包含 numpy 数组
    data1 = torch.load(valid_files[0], weights_only=False)
    data2 = torch.load(valid_files[1], weights_only=False)
    data4 = torch.load(valid_files[2], weights_only=False) # index 3 in original list
    all_topo_values = torch.cat([
        data1['initial_topo'].flatten(), data1['final_topo'].flatten(),
        torch.from_numpy(data2['initial_topo']).float().flatten(), torch.from_numpy(data2['final_topo']).float().flatten(),
        data4['initial_topo'].flatten(), data4['final_topo'].flatten()
    ])
    assert stats['topo']['min'] == pytest.approx(all_topo_values.min().item())
    assert stats['topo']['max'] == pytest.approx(all_topo_values.max().item())

    # 检查 uplift_rate
    all_uplift = [data1['uplift_rate'].item(), data2['uplift_rate'].item(), data4['uplift_rate'].item()]
    assert stats['uplift_rate']['min'] == pytest.approx(min(all_uplift))
    assert stats['uplift_rate']['max'] == pytest.approx(max(all_uplift))

    # 检查 k_f
    all_kf = [data1['k_f'].item(), data2['k_f'].item(), data4['k_f'].item()]
    assert stats['k_f']['min'] == pytest.approx(min(all_kf))
    assert stats['k_f']['max'] == pytest.approx(max(all_kf))

    # 检查 k_d
    all_kd = [data1['k_d'], data2['k_d'].item(), data4['k_d']]
    assert stats['k_d']['min'] == pytest.approx(min(all_kd))
    assert stats['k_d']['max'] == pytest.approx(max(all_kd))


def test_compute_normalization_stats_empty_list(caplog):
    """测试 compute_normalization_stats 处理空文件列表。"""
    with caplog.at_level(logging.WARNING):
        stats = compute_normalization_stats([], ['topo'])
    assert stats is None
    assert "Cannot compute normalization stats: No training files provided." in caplog.text

# Removed patch decorator here, will apply inside the function
def test_compute_normalization_stats_skip_error_file(dummy_pt_files, caplog):
    """测试 compute_normalization_stats 跳过加载失败的文件。"""
    # 我们不使用 dummy_pt_files，但需要保留这个参数以使用 fixture
    # 测试文件不存在的情况
    with caplog.at_level(logging.WARNING):
        stats = compute_normalization_stats(["bad_file.pt"], ['topo'])

    # 应该返回 None，因为没有有效文件
    assert stats is None
    # 检查日志中是否包含文件不存在的警告
    assert "File not found: bad_file.pt" in caplog.text
    # 检查日志中是否包含失败消息
    assert "Failed to compute normalization stats: No files processed successfully" in caplog.text


def test_compute_normalization_stats_no_valid_data_for_field(dummy_pt_files, caplog):
    """测试 compute_normalization_stats 在某字段无有效数据时的处理。"""
    # 创建一个只包含 uplift_rate 的文件
    minimal_data = {'uplift_rate': 0.01}
    minimal_file = os.path.join(os.path.dirname(dummy_pt_files[0]), "minimal.pt")
    # 兼容旧版本 PyTorch
    try:
        torch.save(minimal_data, minimal_file, weights_only=False)
    except TypeError:
        torch.save(minimal_data, minimal_file)

    fields = ['topo', 'uplift_rate'] # 请求 topo 和 uplift_rate 的统计
    with caplog.at_level(logging.WARNING):
        stats = compute_normalization_stats([minimal_file], fields)

    assert stats is not None
    assert stats['uplift_rate']['min'] == 0.01
    assert stats['uplift_rate']['max'] == 0.01
    # 检查是否记录了关于无效数据的警告
    assert "Could not compute valid stats for field 'topo'" in caplog.text
    # 检查是否使用了默认值
    assert stats['topo']['min'] == 0.0
    assert stats['topo']['max'] == 1.0


# --- 测试 create_dataloaders ---

def test_create_dataloaders_success_load_stats(dummy_config, dummy_pt_files, dummy_norm_stats):
    """测试 create_dataloaders 成功创建加载器并加载现有统计信息。"""
    with patch('glob.glob', return_value=dummy_pt_files), \
         patch('random.shuffle'), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', MagicMock()), \
         patch('json.load', return_value=dummy_norm_stats) as mock_json_load:

        result = create_dataloaders(dummy_config)

        assert isinstance(result, dict)
        assert 'train' in result and isinstance(result['train'], DataLoader)
        assert 'val' in result and isinstance(result['val'], DataLoader)
        assert 'test' in result and isinstance(result['test'], DataLoader)
        assert 'norm_stats' in result and result['norm_stats'] == dummy_norm_stats

        # 验证文件分割（基于配置：总共4个，train=2, val=1, test=1）
        assert len(result['train'].dataset.file_list) == 2
        assert len(result['val'].dataset.file_list) == 1
        assert len(result['test'].dataset.file_list) == 1

        # 验证 Dataset 使用了加载的统计信息
        assert result['train'].dataset.normalize is True
        assert result['train'].dataset.norm_stats == dummy_norm_stats
        mock_json_load.assert_called_once() # 确认加载了文件

@patch('glob.glob')
@patch('random.shuffle')
@patch('os.path.exists')
@patch('builtins.open')
@patch('json.load')
@patch('json.dump')
@patch('src.data_utils.compute_normalization_stats')
def test_create_dataloaders_compute_stats(mock_compute_stats, mock_json_dump, mock_json_load, mock_open, mock_exists, _, mock_glob,
                                          dummy_config, dummy_pt_files, dummy_norm_stats):
    """测试 create_dataloaders 在需要时计算并保存统计信息。"""
    mock_glob.return_value = dummy_pt_files
    mock_exists.return_value = False # 模拟 stats 文件不存在
    mock_compute_stats.return_value = dummy_norm_stats # 模拟计算结果
    dummy_config['data']['normalization']['compute_stats'] = True # 允许计算
    stats_file_path = dummy_config['data']['normalization']['stats_file']

    result = create_dataloaders(dummy_config)

    assert result['norm_stats'] == dummy_norm_stats
    mock_json_load.assert_not_called() # 不应尝试加载
    mock_compute_stats.assert_called_once() # 应该调用计算
    # 验证是否尝试保存了文件
    mock_open.assert_called_with(stats_file_path, 'w')
    mock_json_dump.assert_called_once_with(dummy_norm_stats, ANY, indent=2) # ANY 代表文件句柄
    # 验证 Dataset 使用了计算出的统计信息
    assert result['train'].dataset.normalize is True
    assert result['train'].dataset.norm_stats == dummy_norm_stats

@patch('glob.glob')
@patch('random.shuffle')
@patch('os.path.exists')
@patch('src.data_utils.compute_normalization_stats')
def test_create_dataloaders_normalization_disabled(mock_compute_stats, mock_exists, _, mock_glob,
                                                  dummy_config, dummy_pt_files):
    """测试 create_dataloaders 在配置中禁用归一化。"""
    mock_glob.return_value = dummy_pt_files
    dummy_config['data']['normalization']['enabled'] = False

    result = create_dataloaders(dummy_config)

    assert result['norm_stats'] is None
    mock_exists.assert_not_called() # 不应检查文件
    mock_compute_stats.assert_not_called() # 不应计算
    # 验证 Dataset 未启用归一化
    assert result['train'].dataset.normalize is False
    assert result['train'].dataset.norm_stats is None

@patch('glob.glob')
@patch('random.shuffle')
@patch('os.path.exists', return_value=False) # Stats file doesn't exist
@patch('src.data_utils.compute_normalization_stats', return_value=None) # Compute fails
def test_create_dataloaders_stats_fail_disable_norm(mock_compute_stats, mock_exists, mock_shuffle, mock_glob,
                                                     dummy_config, dummy_pt_files, caplog):
    # 忽略未使用的模拟对象
    del mock_exists, mock_shuffle
    """测试 create_dataloaders 在无法加载或计算统计信息时禁用归一化。"""
    mock_glob.return_value = dummy_pt_files
    dummy_config['data']['normalization']['enabled'] = True
    dummy_config['data']['normalization']['compute_stats'] = True # Allow compute

    with caplog.at_level(logging.WARNING):
        result = create_dataloaders(dummy_config)

    assert result['norm_stats'] is None
    assert "Normalization enabled, but no stats were loaded or computed. Disabling normalization." in caplog.text
    # 验证 Dataset 最终禁用归一化
    assert result['train'].dataset.normalize is False
    assert result['train'].dataset.norm_stats is None
    mock_compute_stats.assert_called_once() # 尝试了计算

def test_create_dataloaders_invalid_split(dummy_config):
    """测试 create_dataloaders 在 split 无效时引发 ValueError。"""
    dummy_config['data']['train_split'] = 0.8
    dummy_config['data']['val_split'] = 0.3 # 总和 > 1.0
    with pytest.raises(ValueError, match=r"train_split \(0.8\) \+ val_split \(0.3\) cannot exceed 1.0"):
        create_dataloaders(dummy_config)

@patch('glob.glob', return_value=[]) # No files found
def test_create_dataloaders_no_files_found(mock_glob, dummy_config):
    """测试 create_dataloaders 在找不到文件时引发 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError, match="No .pt files found"):
        create_dataloaders(dummy_config)
    mock_glob.assert_called_once()

@patch('os.path.isdir', return_value=False) # Data dir is not a directory
def test_create_dataloaders_data_dir_not_found(_, dummy_config, caplog):
    """测试 create_dataloaders 在数据目录无效时记录错误并引发 FileNotFoundError。"""
    data_dir = dummy_config['data']['processed_dir']
    with pytest.raises(FileNotFoundError, match="No .pt files found"), caplog.at_level(logging.ERROR):
        create_dataloaders(dummy_config)
    assert f"Data directory not found or is not a directory: {data_dir}" in caplog.text

def test_create_dataloaders_split_edge_cases(dummy_config, dummy_pt_files):
    """测试 create_dataloaders 处理分割的边缘情况（例如 val=0 但有剩余）。"""
    dummy_config['data']['train_split'] = 0.9 # 3 files
    dummy_config['data']['val_split'] = 0.0   # 0 files initially
    # test split = 0.1 (1 file)

    with patch('glob.glob', return_value=dummy_pt_files), \
         patch('random.shuffle'):
        result = create_dataloaders(dummy_config)

    assert len(result['train'].dataset.file_list) == 3
    assert len(result['val'].dataset.file_list) == 0 # val_split is 0
    assert len(result['test'].dataset.file_list) == 1

    # 测试 val_split > 0 但计算为 0 的情况
    dummy_config['data']['train_split'] = 0.8 # 3 files
    dummy_config['data']['val_split'] = 0.1   # 计算为 0, 但应分配 1
    # test split = 0.1 (0 files)

    with patch('glob.glob', return_value=dummy_pt_files), \
         patch('random.shuffle'):
        result = create_dataloaders(dummy_config)

    assert len(result['train'].dataset.file_list) == 3 # Train 调整为 3
    assert len(result['val'].dataset.file_list) == 1   # Val 获得 1
    assert len(result['test'].dataset.file_list) == 0   # Test 变为 0