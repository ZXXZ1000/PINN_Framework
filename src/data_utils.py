# PINN_Framework/src/data_utils.py
"""
数据处理工具，包括数据集类和数据加载器创建函数。
"""

import os
import glob
import logging
import json
import math
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import List, Dict, Optional, Callable, Any

# --- Fastscape Dataset Class ---

class FastscapeDataset(Dataset):
    """PyTorch Dataset for loading Fastscape simulation data stored as .pt files."""
    def __init__(self,
                 file_list: List[str],
                 normalize: bool = False,
                 norm_stats: Optional[Dict[str, Dict[str, float]]] = None,
                 transform: Optional[Callable] = None):
        """
        Args:
            file_list: List of paths to the data files (.pt).
            normalize: Whether to apply Min-Max normalization.
            norm_stats: Dictionary containing min/max statistics for normalization.
                        Required if normalize is True. Keys should match fields to normalize.
            transform: Optional transform to be applied on a sample.
        """
        self.file_list = file_list
        self.normalize = normalize
        self.norm_stats = norm_stats
        self.transform = transform
        self.epsilon = 1e-8 # For safe normalization

        if not self.file_list:
            logging.warning("FastscapeDataset initialized with an empty file list.")

        if self.normalize and self.norm_stats is None:
            logging.warning("Normalization enabled but no norm_stats provided. Disabling normalization.")
            self.normalize = False

        if self.normalize:
            logging.info("FastscapeDataset: Min-Max normalization enabled.")
            logging.debug(f"Normalization stats available for keys: {list(self.norm_stats.keys())}")
        else:
            logging.info("FastscapeDataset: Normalization disabled.")

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.file_list)

    def _to_float_tensor(self, value: Any) -> torch.Tensor:
        """Safely converts scalar, numpy array, or tensor to a FloatTensor."""
        if isinstance(value, torch.Tensor):
            return value.float()
        elif isinstance(value, np.ndarray):
            return torch.from_numpy(value).float()
        elif isinstance(value, (int, float)):
            return torch.tensor(float(value), dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported type for parameter conversion: {type(value)}")

    def _normalize_field(self, tensor: torch.Tensor, field_key: str) -> torch.Tensor:
        """
        Applies Min-Max normalization using stored stats.

        Args:
            tensor: The tensor to normalize.
            field_key: The key to use for looking up normalization stats.

        Returns:
            The normalized tensor, or the original tensor if normalization fails.

        Note:
            If normalization stats are missing or invalid, a warning is logged and
            the original tensor is returned unchanged.
        """
        stats = self.norm_stats.get(field_key)
        if not stats:
            logging.debug(f"Normalization stats missing for key '{field_key}'. Skipping normalization.")
            return tensor

        if 'min' not in stats or 'max' not in stats:
            logging.warning(f"Incomplete normalization stats for key '{field_key}'. Skipping normalization.")
            return tensor

        try:
            min_val = torch.tensor(stats['min'], device=tensor.device, dtype=tensor.dtype)
            max_val = torch.tensor(stats['max'], device=tensor.device, dtype=tensor.dtype)

            # Check for invalid stats (min >= max)
            if min_val >= max_val:
                logging.warning(f"Invalid normalization range for '{field_key}': min={min_val.item()}, max={max_val.item()}")
                return tensor

            range_val = max_val - min_val
            # Normalize: (value - min) / (range + epsilon)
            normalized = (tensor - min_val) / (range_val + self.epsilon)

            # Check for NaN or Inf values (can happen with extreme values)
            if not torch.isfinite(normalized).all():
                logging.warning(f"Normalization produced non-finite values for '{field_key}'. Using original values.")
                return tensor

            return normalized
        except Exception as e:
            logging.warning(f"Error during normalization of '{field_key}': {e}")
            return tensor

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Loads, preprocesses, and returns a sample from the dataset.

        Args:
            idx: Index of the sample to load.

        Returns:
            A dictionary containing the processed sample data, or None if an error occurred.

        Raises:
            FileNotFoundError: If the file doesn't exist (not caught internally).
        """
        filepath = self.file_list[idx]

        # Check if file exists before attempting to load
        if not os.path.exists(filepath):
            logging.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        try:
            # 加载数据：直接使用 weights_only=False，因为我们的数据包含 numpy 数组
            sample_data = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)

            # --- Extract and Convert Required Fields ---
            required_keys = ['initial_topo', 'final_topo', 'uplift_rate', 'k_f', 'k_d', 'm', 'n', 'run_time']
            processed_sample = {}

            # Check for missing keys first before processing any data
            missing_keys = [key for key in required_keys if key not in sample_data]
            if missing_keys:
                logging.warning(f"Missing required data fields {missing_keys} in {filepath}")
                return None  # Return None for missing keys, will be filtered by collate_fn

            # Process all required fields
            for key in required_keys:
                try:
                    processed_sample[key] = self._to_float_tensor(sample_data[key])
                except (TypeError, ValueError) as e:
                    logging.warning(f"Error converting field '{key}' in {filepath}: {e}")
                    return None  # Return None for conversion errors, will be filtered by collate_fn

            # --- Normalization ---
            if self.normalize:
                # Define which fields to normalize and their corresponding keys in norm_stats
                fields_to_normalize = {
                    'initial_topo': 'topo', # Use combined 'topo' stats
                    'final_topo': 'topo',
                    'uplift_rate': 'uplift_rate',
                    'k_f': 'k_f',
                    'k_d': 'k_d',
                    # m, n, run_time are usually not normalized
                }
                for field, stats_key in fields_to_normalize.items():
                    if field in processed_sample:
                         processed_sample[field] = self._normalize_field(processed_sample[field], stats_key)

            # --- Add target shape (useful for some models/losses) ---
            processed_sample['target_shape'] = processed_sample['final_topo'].shape

            # --- Apply other transforms ---
            if self.transform:
                 processed_sample = self.transform(processed_sample)

            return processed_sample

        except torch.serialization.pickle.UnpicklingError as e:
            logging.error(f"Corrupted file or incompatible format in {filepath}: {e}")
            return None  # Return None on error, handled by collate_fn
        except Exception as e:
            logging.error(f"Error loading/processing sample {idx} from {filepath}: {e}", exc_info=True)
            return None  # Return None on error, handled by collate_fn

    def denormalize_state(self, normalized_state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalizes a state tensor (e.g., topography) using stored 'topo' stats.

        Args:
            normalized_state_tensor: The normalized tensor to denormalize.

        Returns:
            The denormalized tensor, or the original tensor if denormalization fails.

        Note:
            This method assumes the tensor was normalized using the 'topo' stats.
            If normalization is disabled or stats are missing, the original tensor is returned.
        """
        if not self.normalize or self.norm_stats is None:
            return normalized_state_tensor

        topo_stats = self.norm_stats.get('topo')
        if not topo_stats:
            logging.warning("Normalization stats for 'topo' missing. Cannot denormalize state.")
            return normalized_state_tensor

        if 'min' not in topo_stats or 'max' not in topo_stats:
            logging.warning("Incomplete normalization stats for 'topo'. Cannot denormalize state.")
            return normalized_state_tensor

        try:
            min_val = torch.tensor(topo_stats['min'], device=normalized_state_tensor.device, dtype=normalized_state_tensor.dtype)
            max_val = torch.tensor(topo_stats['max'], device=normalized_state_tensor.device, dtype=normalized_state_tensor.dtype)

            # Check for invalid stats (min >= max)
            if min_val >= max_val:
                logging.warning(f"Invalid normalization range for 'topo': min={min_val.item()}, max={max_val.item()}")
                return normalized_state_tensor

            range_val = max_val - min_val
            # Denormalize: normalized * (range + epsilon) + min
            denormalized = normalized_state_tensor * (range_val + self.epsilon) + min_val

            # Check for NaN or Inf values
            if not torch.isfinite(denormalized).all():
                logging.warning("Denormalization produced non-finite values. Using original values.")
                return normalized_state_tensor

            return denormalized
        except Exception as e:
            logging.warning(f"Error during denormalization: {e}")
            return normalized_state_tensor

# --- Utility Function to Create DataLoaders ---

def collate_fn_filter_none(batch: List[Optional[Dict]]) -> Optional[Dict]:
    """
    Custom collate_fn that filters out None results from __getitem__.

    Args:
        batch: A list of samples from the dataset, potentially containing None values.

    Returns:
        A collated batch dictionary, or None if all samples were None or collation failed.
    """
    # Filter out None samples
    valid_samples = list(filter(lambda x: x is not None, batch))

    # Log information about filtered samples
    if len(valid_samples) < len(batch):
        logging.info(f"Filtered out {len(batch) - len(valid_samples)} invalid samples from batch of {len(batch)}")

    # Return None if no valid samples remain
    if not valid_samples:
        logging.warning("All samples in batch were invalid (None). Returning None.")
        return None

    try:
        # Use default collate to combine samples into a batch
        return torch.utils.data.dataloader.default_collate(valid_samples)
    except RuntimeError as e:
        # Common error: tensor size mismatch
        logging.warning(f"RuntimeError during collation: {e}")
        if "size mismatch" in str(e):
            logging.warning("Size mismatch detected. Check if all samples have consistent tensor shapes.")
        return None
    except TypeError as e:
        # Common error: unexpected data type
        logging.warning(f"TypeError during collation: {e}")
        if valid_samples and isinstance(valid_samples[0], dict):
            logging.warning(f"First sample keys and types: {[(k, type(v)) for k, v in valid_samples[0].items()]}")
        return None
    except Exception as e:
        # Other unexpected errors
        logging.error(f"Unexpected error during collation: {e}")
        if valid_samples:
            logging.error(f"First item keys: {valid_samples[0].keys() if isinstance(valid_samples[0], dict) else 'Not a dict'}")
        return None

def compute_normalization_stats(train_files: List[str], fields_for_stats: List[str]) -> Optional[Dict]:
    """
    Computes Min-Max statistics from a list of training files.

    Args:
        train_files: List of paths to training data files.
        fields_for_stats: List of field names to compute statistics for.

    Returns:
        Dictionary containing min/max statistics for each field, or None if computation fails.
    """
    if not train_files:
        logging.warning("Cannot compute normalization stats: No training files provided.")
        return None

    if not fields_for_stats:
        logging.warning("Cannot compute normalization stats: No fields specified.")
        return None

    logging.info(f"Computing Min-Max normalization statistics from {len(train_files)} training files...")
    logging.info(f"Fields to compute stats for: {fields_for_stats}")

    # Initialize stats dictionary with infinity values
    norm_stats = {field: {'min': float('inf'), 'max': float('-inf')} for field in fields_for_stats}
    num_processed = 0
    field_counts = {field: 0 for field in fields_for_stats}  # Track how many files contributed to each field

    for f_path in train_files:
        try:
            # Check if file exists
            if not os.path.exists(f_path):
                logging.warning(f"File not found: {f_path}. Skipping.")
                continue

            # Load data with appropriate error handling
            try:
                # 直接使用 weights_only=False，因为我们的数据包含 numpy 数组
                data = torch.load(f_path, map_location='cpu', weights_only=False)
            except torch.serialization.pickle.UnpicklingError as e:
                logging.warning(f"Corrupted file or incompatible format: {f_path}. Error: {e}. Skipping.")
                continue
            except Exception as e:
                logging.warning(f"Error loading file {f_path}: {e}. Skipping.")
                continue

            num_processed += 1

            # Progress logging for large datasets
            if num_processed % 100 == 0 or num_processed == len(train_files):
                logging.info(f"Processed {num_processed}/{len(train_files)} files ({num_processed/len(train_files)*100:.1f}%)")

            def _update_stats(value, field_key):
                """Helper function to update min/max stats for a given field."""
                if field_key not in norm_stats:
                    return  # Only update requested fields

                current_min, current_max = float('inf'), float('-inf')

                # Handle different data types
                if isinstance(value, torch.Tensor):
                    if value.numel() > 0:  # Avoid errors on empty tensors
                        if not torch.isfinite(value).all():
                            logging.warning(f"Non-finite values found in {field_key} in file {f_path}. Using only finite values.")
                            finite_mask = torch.isfinite(value)
                            if not finite_mask.any():
                                return  # Skip if no finite values
                            value = value[finite_mask]
                        current_min = value.min().item()
                        current_max = value.max().item()
                    else:
                        return  # Skip empty tensors
                elif isinstance(value, np.ndarray):
                    if value.size > 0:
                        if not np.isfinite(value).all():
                            logging.warning(f"Non-finite values found in {field_key} in file {f_path}. Using only finite values.")
                            finite_mask = np.isfinite(value)
                            if not finite_mask.any():
                                return  # Skip if no finite values
                            value = value[finite_mask]
                        current_min = float(value.min())
                        current_max = float(value.max())
                    else:
                        return  # Skip empty arrays
                elif isinstance(value, (int, float)):
                    if not math.isfinite(value):
                        logging.warning(f"Non-finite value found in {field_key} in file {f_path}. Skipping.")
                        return
                    current_min = current_max = float(value)
                else:
                    logging.debug(f"Unsupported type for {field_key} in file {f_path}: {type(value)}. Skipping.")
                    return  # Skip unsupported types

                # Update global min/max
                norm_stats[field_key]['min'] = min(norm_stats[field_key]['min'], current_min)
                norm_stats[field_key]['max'] = max(norm_stats[field_key]['max'], current_max)
                field_counts[field_key] += 1

            # Update stats for relevant fields
            # Use 'topo' key for both initial and final topo stats
            if 'topo' in fields_for_stats:
                if 'initial_topo' in data: _update_stats(data['initial_topo'], 'topo')
                if 'final_topo' in data: _update_stats(data['final_topo'], 'topo')
            if 'uplift_rate' in fields_for_stats and 'uplift_rate' in data:
                _update_stats(data['uplift_rate'], 'uplift_rate')
            if 'k_f' in fields_for_stats and 'k_f' in data:
                _update_stats(data['k_f'], 'k_f')
            if 'k_d' in fields_for_stats and 'k_d' in data:
                _update_stats(data['k_d'], 'k_d')
            # Add other fields if needed

        except Exception as e:
            logging.warning(f"Skipping file {f_path} during stats computation due to error: {e}")

    if num_processed > 0:
        logging.info(f"Min-Max stats computed from {num_processed} training files.")

        # Check for infinite values which indicate no valid data found for a field
        valid_fields = 0
        for field, stats in norm_stats.items():
            if stats['min'] == float('inf') or stats['max'] == float('-inf'):
                logging.warning(f"Could not compute valid stats for field '{field}'. No valid data found in {field_counts[field]} files.")
                # Set default values for fields with no valid data
                norm_stats[field] = {'min': 0.0, 'max': 1.0}
                logging.warning(f"Using default values for '{field}': min=0.0, max=1.0")
            else:
                valid_fields += 1
                # Check for very small ranges which might cause numerical issues
                if abs(stats['max'] - stats['min']) < 1e-6:
                    logging.warning(f"Very small range for field '{field}': {stats['min']} to {stats['max']}. This may cause numerical issues.")
                logging.info(f"Stats for '{field}': min={stats['min']:.6g}, max={stats['max']:.6g} (from {field_counts[field]} files)")

        if valid_fields == 0:
            logging.error("No valid statistics could be computed for any field.")
            return None

        logging.debug(f"Computed norm_stats: {norm_stats}")
        return norm_stats
    else:
        logging.error("Failed to compute normalization stats: No files processed successfully.")
        return None

def create_dataloaders(config: Dict) -> Dict[str, DataLoader]:
    """
    Creates train, validation, and test dataloaders with Min-Max normalization handling.

    Args:
        config: Dictionary containing configuration, typically loaded from YAML.
                Expected structure:
                config:
                  data:
                    processed_dir: Path to processed data.
                    train_split: Fraction for training set (e.g., 0.8).
                    val_split: Fraction for validation set (e.g., 0.1).
                    num_workers: Number of workers for DataLoader.
                    normalization:
                      enabled: True/False.
                      compute_stats: True/False (compute from train set if stats_file not found).
                      stats_file: Path to save/load normalization stats (JSON).
                  training:
                    batch_size: Batch size for DataLoaders.

    Returns:
        Dictionary containing 'train', 'val', and 'test' DataLoaders, and 'norm_stats'.
    """
    data_config = config.get('data', {})
    norm_config = data_config.get('normalization', {})
    train_config = config.get('training', {})

    data_dir = data_config.get('processed_dir', 'data/processed')
    batch_size = train_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 0)
    train_split = data_config.get('train_split', 0.8)
    val_split = data_config.get('val_split', 0.1)
    if train_split + val_split > 1.0:
        raise ValueError(f"train_split ({train_split}) + val_split ({val_split}) cannot exceed 1.0")
    test_split = 1.0 - train_split - val_split

    logging.info(f"Creating dataloaders from: {data_dir}")
    logging.info(f"Batch size: {batch_size}, Num workers: {num_workers}")
    logging.info(f"Splits: Train={train_split:.2f}, Val={val_split:.2f}, Test={test_split:.2f}")

    # --- Find data files ---
    all_files = []
    if os.path.isdir(data_dir):
        # Search recursively for .pt files
        all_files = glob.glob(os.path.join(data_dir, '**', '*.pt'), recursive=True)
    else:
         logging.error(f"Data directory not found or is not a directory: {data_dir}")

    if not all_files:
        raise FileNotFoundError(f"No .pt files found in the specified data directory structure: {data_dir}")

    # --- Shuffle and Split Files ---
    random.shuffle(all_files)
    num_total = len(all_files)
    num_train = int(num_total * train_split)
    num_val = int(num_total * val_split)
    # Ensure val gets at least one sample if possible and requested
    if val_split > 0 and num_val == 0 and num_total > num_train: num_val = 1
    num_train = min(num_train, num_total - num_val) # Adjust train if val took samples

    # Split files according to calculated sizes
    train_files = all_files[:num_train]
    val_files = all_files[num_train : num_train + num_val]
    test_files = all_files[num_train + num_val:]

    # Log the actual split sizes
    logging.info(f"Total files found: {num_total}")
    logging.info(f"Requested split ratio: Train={train_split:.2f}, Val={val_split:.2f}, Test={1.0-train_split-val_split:.2f}")
    logging.info(f"Actual split sizes: Train={len(train_files)} ({len(train_files)/num_total:.2f}), "
                f"Val={len(val_files)} ({len(val_files)/num_total:.2f}), "
                f"Test={len(test_files)} ({len(test_files)/num_total:.2f})")

    # --- Normalization Handling ---
    normalize_data = norm_config.get('enabled', False)
    norm_stats = None
    stats_loaded_or_computed = False

    if normalize_data:
        stats_file = norm_config.get('stats_file')
        compute_stats_flag = norm_config.get('compute_stats', False)
        # Define fields for which stats are needed (match keys used in FastscapeDataset)
        fields_for_stats = ['topo', 'uplift_rate', 'k_f', 'k_d']

        # 1. Try loading stats
        if stats_file and os.path.exists(stats_file):
            logging.info(f"Attempting to load normalization stats from: {stats_file}")
            try:
                with open(stats_file, 'r') as f: norm_stats = json.load(f)
                logging.info("Normalization stats loaded successfully.")
                stats_loaded_or_computed = True
            except Exception as e:
                logging.error(f"Failed to load normalization stats from {stats_file}: {e}. Will attempt to compute if enabled.")

        # 2. Compute stats if not loaded and compute_stats is True
        if not stats_loaded_or_computed and compute_stats_flag:
            norm_stats = compute_normalization_stats(train_files, fields_for_stats)
            if norm_stats:
                stats_loaded_or_computed = True
                # Save computed stats if path provided
                if stats_file:
                    try:
                        # Ensure directory exists before saving
                        stats_dir = os.path.dirname(stats_file)
                        if stats_dir: # Check if dirname is not empty (i.e., not in root)
                             os.makedirs(stats_dir, exist_ok=True)
                        with open(stats_file, 'w') as f: json.dump(norm_stats, f, indent=2)
                        logging.info(f"Normalization statistics saved to: {stats_file}")
                    except Exception as e:
                        logging.error(f"Failed to save normalization stats to {stats_file}: {e}")

        # 3. Disable normalization if stats are still missing
        if not stats_loaded_or_computed:
            logging.warning("Normalization enabled, but no stats were loaded or computed. Disabling normalization.")
            normalize_data = False
            norm_stats = None
    else:
        logging.info("Normalization is disabled in the configuration.")

    # --- Create Datasets and DataLoaders ---
    train_dataset = FastscapeDataset(train_files, normalize=normalize_data, norm_stats=norm_stats)
    val_dataset = FastscapeDataset(val_files, normalize=normalize_data, norm_stats=norm_stats)
    test_dataset = FastscapeDataset(test_files, normalize=normalize_data, norm_stats=norm_stats)

    # Use persistent_workers only if num_workers > 0
    persist_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, collate_fn=collate_fn_filter_none, persistent_workers=persist_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, collate_fn=collate_fn_filter_none, persistent_workers=persist_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, collate_fn=collate_fn_filter_none, persistent_workers=persist_workers
    )

    # Return loaders and the norm_stats used (or None)
    return {'train': train_loader, 'val': val_loader, 'test': test_loader, 'norm_stats': norm_stats}