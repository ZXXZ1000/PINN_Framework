# PINN_Framework/tests/conftest.py
import sys
import os
import pytest

# 将项目根目录 (PINN_Framework) 添加到 sys.path
# conftest.py 位于 tests 目录下，其父目录是 PINN_Framework
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    print(f"Adding project root to sys.path: {project_root}")
    sys.path.insert(0, project_root)

# 可以在这里定义通用的 fixtures，例如临时目录等
# @pytest.fixture(scope="session")
# def common_resource():
#     print("Setting up common resource")
#     yield "resource_data"
#     print("Tearing down common resource")