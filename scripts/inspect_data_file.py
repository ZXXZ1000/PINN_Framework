import os
import sys
import torch
import argparse

def inspect_data_file(file_path):
    """
    Inspect the content of a .pt data file and print its structure.

    Args:
        file_path: Path to the .pt file to inspect
    """
    print(f"Inspecting file: {file_path}")

    try:
        # Load the data file
        data = torch.load(file_path, map_location=torch.device('cpu'))

        # Print the type of the data
        print(f"Data type: {type(data)}")

        # If it's a dictionary, print the keys and their types
        if isinstance(data, dict):
            print("\nKeys and their types:")
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: Tensor of shape {value.shape}, dtype={value.dtype}")
                else:
                    print(f"  {key}: {type(value)}")

            # Print a sample of each tensor (first few elements)
            print("\nSample values:")
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    if value.numel() > 0:
                        flat_value = value.flatten()
                        sample_size = min(5, flat_value.numel())
                        print(f"  {key} (first {sample_size} elements): {flat_value[:sample_size]}")
                    else:
                        print(f"  {key}: Empty tensor")
                elif isinstance(value, (int, float, str, bool)):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: Complex type, not showing value")
        else:
            # If it's not a dictionary, print what we can about it
            print("\nData is not a dictionary. Basic information:")
            if isinstance(data, torch.Tensor):
                print(f"Tensor of shape {data.shape}, dtype={data.dtype}")
                if data.numel() > 0:
                    flat_data = data.flatten()
                    sample_size = min(5, flat_data.numel())
                    print(f"First {sample_size} elements: {flat_data[:sample_size]}")
            else:
                print(f"Data: {data}")

    except Exception as e:
        print(f"Error inspecting file: {e}")

def main():
    # 直接在文件中置入文件路径
    file_path = r"D:\OneDrive\MR.Z  所有资料\code\PINN_TEST\PINN_Framework\data\processed\resolution_64x64\sample_00002.pt"

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"Error: File does not exist: {file_path}")
        return

    inspect_data_file(file_path)

if __name__ == "__main__":
    main()
