# PINN_Framework/src/trainer.py
"""
模型训练器模块，负责管理训练和验证循环。
"""

import os
import logging
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm # Keep tqdm for progress visualization
import time
from typing import Dict, Optional, Tuple, Any

# Import necessary components from the new framework
try:
    from .models import AdaptiveFastscapePINN, TimeDerivativePINN # Main model and base class
    from .losses import compute_total_loss, compute_pde_residual_dual_output # Main loss functions
    from .utils import set_seed, get_device # Assuming utils.py exists
except ImportError as e:
    logging.error(f"Failed to import necessary components: {e}. Ensure models.py, losses.py, utils.py exist.")
    # Define placeholders to allow module loading but fail at runtime
    class AdaptiveFastscapePINN: pass
    class TimeDerivativePINN: pass
    def compute_total_loss(*args, **kwargs): raise NotImplementedError("compute_total_loss not imported")
    def compute_pde_residual_dual_output(*args, **kwargs): raise NotImplementedError("compute_pde_residual_dual_output not imported")
    def set_seed(*args, **kwargs): pass
    def get_device(*args, **kwargs): return torch.device('cpu')


class DynamicWeightScheduler:
    """简单的动态（目前为静态）损失权重调度器。"""
    def __init__(self, config: Dict):
        self.weights_config = config.get('training', {}).get('loss_weights', {})
        logging.info(f"Initializing DynamicWeightScheduler with weights: {self.weights_config}")

    def get_weights(self, epoch: int) -> Dict[str, float]:
        # Placeholder: return static weights
        # TODO: Implement dynamic scheduling if needed
        return self.weights_config

class PINNTrainer:
    """Handles the training and validation loops for the AdaptiveFastscapePINN model."""
    def __init__(self, model: AdaptiveFastscapePINN, config: Dict, train_loader: DataLoader, val_loader: Optional[DataLoader]):
        """
        Initializes the PINN trainer.

        Args:
            model: The AdaptiveFastscapePINN model instance.
            config: Configuration dictionary.
            train_loader: Training data loader.
            val_loader: Validation data loader (optional).
        """
        if not isinstance(model, AdaptiveFastscapePINN):
             # Although we expect AdaptiveFastscapePINN, check base class for safety
             if not isinstance(model, TimeDerivativePINN):
                 raise TypeError(f"Model must be an instance of AdaptiveFastscapePINN or TimeDerivativePINN, got {type(model)}")
             else:
                  logging.warning(f"Received model of type {type(model)}, expected AdaptiveFastscapePINN. Proceeding, but ensure compatibility.")

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Extract relevant config sections
        self.train_config = config.get('training', {})
        self.physics_params = config.get('physics', {})
        self.data_config = config.get('data', {}) # Needed for domain info fallback

        # Setup device
        self.device = get_device(self.train_config.get('device', 'auto'))
        self.model = self.model.to(self.device)
        logging.info(f"Trainer using device: {self.device}")

        # Setup optimizer
        self.optimizer = self._setup_optimizer()

        # Other training params needed by scheduler setup
        self.max_epochs = self.train_config.get('max_epochs', 100)

        # Setup LR scheduler (needs max_epochs for CosineAnnealingLR)
        self.scheduler = self._setup_lr_scheduler()

        # Loss weights
        self.loss_weight_scheduler = DynamicWeightScheduler(config) # Pass main config

        # Ensure model is configured for dual output (required by this trainer)
        if isinstance(self.model, TimeDerivativePINN):
            self.model.set_output_mode(state=True, derivative=True)
            logging.info("Model configured for dual output (state and derivative).")
        else:
            # This case should ideally not happen due to the initial check
             logging.warning("Model is not TimeDerivativePINN, cannot configure output mode.")


        # Other training params
        # self.max_epochs = self.train_config.get('max_epochs', 100) # Moved earlier
        self.results_dir = self.train_config.get('results_dir', 'results')
        self.run_name = self.train_config.get('run_name', 'pinn_run')

        # Mixed Precision
        self.use_amp = self.train_config.get('use_mixed_precision', False) and self.device.type == 'cuda'
        # 使用新的 API 路径
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        if self.use_amp: logging.info("Mixed precision training enabled (CUDA).")
        else: logging.info("Mixed precision training disabled.")

        # Setup checkpoint directory
        self.checkpoint_dir = os.path.join(self.results_dir, self.run_name, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Setup TensorBoard
        tensorboard_dir = os.path.join(self.results_dir, self.run_name, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tensorboard_dir)

        self.start_epoch = 0
        self.best_val_loss = float('inf')

        # Load checkpoint if specified
        load_path = self.train_config.get('load_checkpoint')
        if load_path:
            self.load_checkpoint(load_path)

    def _setup_optimizer(self) -> optim.Optimizer:
        """Sets up the optimizer based on config."""
        optimizer_name = self.train_config.get('optimizer', 'adam').lower()
        lr = self.train_config.get('learning_rate', 1e-3)
        weight_decay = self.train_config.get('weight_decay', 0.0)
        params = self.model.parameters()

        if optimizer_name == 'adam':
            optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'lbfgs':
             # LBFGS requires special handling in the training loop (closure)
             optimizer = optim.LBFGS(params, lr=lr, line_search_fn="strong_wolfe")
             logging.warning("LBFGS optimizer selected. Ensure the training loop handles the closure correctly.")
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_name}")

        logging.info(f"Optimizer: {optimizer_name}, LR: {lr}, Weight Decay: {weight_decay}")
        return optimizer

    def _setup_lr_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Sets up the learning rate scheduler based on config."""
        scheduler_type = self.train_config.get('lr_scheduler', 'none').lower()
        scheduler_config = self.train_config.get('lr_scheduler_config', {})

        if scheduler_type == 'step':
            step_size = scheduler_config.get('step_size', 30)
            gamma = scheduler_config.get('gamma', 0.1)
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
            logging.info(f"Using StepLR scheduler: step_size={step_size}, gamma={gamma}")
        elif scheduler_type == 'plateau':
            patience = scheduler_config.get('patience', 10)
            factor = scheduler_config.get('factor', 0.1)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=factor, patience=patience, verbose=False)
            logging.info(f"Using ReduceLROnPlateau scheduler: patience={patience}, factor={factor}")
        elif scheduler_type == 'cosine':
            t_max = scheduler_config.get('t_max', self.max_epochs)
            eta_min = scheduler_config.get('eta_min', 0)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=eta_min)
            logging.info(f"Using CosineAnnealingLR scheduler: T_max={t_max}, eta_min={eta_min}")
        elif scheduler_type == 'none':
            scheduler = None
            logging.info("No learning rate scheduler used.")
        else:
            logging.warning(f"Unknown scheduler type: {scheduler_type}. No scheduler used.")
            scheduler = None
        return scheduler

    def _run_epoch(self, epoch: int, is_training: bool) -> Tuple[float, Dict[str, float]]:
        """Runs a single epoch of training or validation."""
        self.model.train(is_training)
        loader = self.train_loader if is_training else self.val_loader
        if loader is None and not is_training:
             logging.warning("Validation loader not provided, skipping validation.")
             return float('inf'), {} # Return inf loss if no validation loader

        epoch_loss = 0.0
        epoch_loss_components = {}
        batch_count = 0
        finite_batch_count = 0 # Count batches with finite loss

        progress_bar = tqdm(loader, desc=f"Epoch {epoch} {'Train' if is_training else 'Val'}", leave=False)

        for batch_data in progress_bar:
            if batch_data is None:
                logging.warning(f"Skipping batch due to loading error (collate_fn returned None).")
                continue
            batch_count += 1

            # --- Prepare Batch Data ---
            try:
                if not isinstance(batch_data, dict):
                    logging.error(f"Epoch {epoch}, Batch {batch_count}: Unexpected batch data type: {type(batch_data)}. Skipping.")
                    continue

                # Move tensors to device
                batch_data_device = {}
                final_topo = None
                for k, v in batch_data.items():
                    if isinstance(v, torch.Tensor):
                        if k == 'final_topo': final_topo = v.to(self.device)
                        else: batch_data_device[k] = v.to(self.device)
                    else: # Keep non-tensor data as is (e.g., target_shape)
                         batch_data_device[k] = v

                if final_topo is None:
                    logging.error(f"Epoch {epoch}, Batch {batch_count}: 'final_topo' not found in batch. Skipping.")
                    continue

                initial_state = batch_data_device.get('initial_topo')
                if initial_state is None:
                    logging.error(f"Epoch {epoch}, Batch {batch_count}: 'initial_topo' not found in batch. Skipping.")
                    continue

                # Prepare parameters dict for the model
                # Ensure parameters needed by the model are present
                params_dict = {
                    'K': batch_data_device.get('k_f'),
                    'D': batch_data_device.get('k_d'),
                    'U': batch_data_device.get('uplift_rate'),
                    'm': batch_data_device.get('m'), # Usually scalar
                    'n': batch_data_device.get('n')  # Usually scalar
                }
                # Check for None values in params_dict if they are critical for the model
                if params_dict['K'] is None or params_dict['D'] is None or params_dict['U'] is None:
                     logging.warning(f"Epoch {epoch}, Batch {batch_count}: One or more parameters (K, D, U) are None in batch data.")
                     # Decide how to handle: skip batch, use defaults, etc.
                     # For now, let the model handle potential None values if it can.

                t_target = batch_data_device.get('run_time')
                if t_target is None:
                    # Estimate target time if not provided (e.g., use a default from config)
                    default_time = self.physics_params.get('total_time', 1.0) # Example default
                    t_target = torch.tensor(default_time, device=self.device, dtype=initial_state.dtype)
                    logging.debug("Using default total_time as t_target.")

                model_input_state = {'initial_state': initial_state, 'params': params_dict, 't_target': t_target}

            except Exception as e:
                 logging.error(f"Error preparing data batch {batch_count}: {e}. Skipping batch.", exc_info=True)
                 continue

            # --- Define Closure for LBFGS ---
            # Note: LBFGS is less common with AMP. Standard optimizers are preferred.
            def closure() -> torch.Tensor:
                if is_training:
                    self.optimizer.zero_grad(set_to_none=True) # More efficient zeroing

                # --- Forward Pass & Loss Calculation ---
                with torch.set_grad_enabled(is_training):
                    # Use autocast for mixed precision
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        try:
                            # Model forward pass (predict_state mode for grid data)
                            # AdaptiveFastscapePINN should return {'state': ..., 'derivative': ...}
                            model_outputs = self.model(model_input_state, mode='predict_state')

                            if not isinstance(model_outputs, dict) or 'state' not in model_outputs or 'derivative' not in model_outputs:
                                 raise TypeError(f"Model did not return expected dual output dictionary. Got: {type(model_outputs)}")

                            data_pred = model_outputs['state'] # Prediction for data loss & smoothness

                            # Calculate Physics Loss (Dual Output)
                            # Pass necessary physics params for the loss function
                            physics_params_for_loss = self.physics_params.copy()
                            # Add da_params if they exist in main config's physics section
                            physics_params_for_loss['da_params'] = self.physics_params.get('drainage_area_kwargs', {})

                            physics_loss = compute_pde_residual_dual_output(
                                outputs=model_outputs, # Pass the full output dict
                                physics_params=physics_params_for_loss
                            )

                            # Compute Total Loss
                            current_loss_weights = self.loss_weight_scheduler.get_weights(epoch)
                            total_loss, loss_components = compute_total_loss(
                                data_pred=data_pred,
                                target_topo=final_topo,
                                physics_loss_value=physics_loss,
                                smoothness_pred=data_pred, # Use state prediction for smoothness
                                physics_params=self.physics_params, # Pass dx, dy etc.
                                loss_weights=current_loss_weights
                            )

                        except Exception as e:
                            logging.error(f"Error during forward/loss calculation in closure: {e}", exc_info=True)
                            # Return NaN to signal error to LBFGS or outer loop
                            return torch.tensor(float('nan'), device=self.device)

                # --- Backward Pass (if training) ---
                if is_training:
                    if not isinstance(total_loss, torch.Tensor) or not torch.isfinite(total_loss):
                        loss_val_repr = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
                        logging.warning(f"Epoch {epoch}, Batch {batch_count}: Invalid/non-finite loss ({loss_val_repr}) in closure. Skipping backward.")
                        return torch.tensor(float('nan'), device=self.device) # Signal error
                    else:
                        # Scale loss for mixed precision backward pass
                        self.scaler.scale(total_loss).backward()
                        # Return loss value for LBFGS or outer loop check
                        return total_loss
                else:
                    # Validation: return loss if valid, else NaN
                    return total_loss if isinstance(total_loss, torch.Tensor) and torch.isfinite(total_loss) else torch.tensor(float('nan'), device=self.device)
            # --- End of Closure ---

            # --- Optimization Step ---
            if is_training:
                if isinstance(self.optimizer, optim.LBFGS):
                    # LBFGS step requires the closure
                    loss = self.optimizer.step(closure)
                    # Note: loss_components might not reflect the final evaluation by LBFGS
                else:
                    # Standard optimizers: call closure once, then step
                    loss = closure() # Computes loss and gradients
                    if isinstance(loss, torch.Tensor) and torch.isfinite(loss):
                        # Optional gradient clipping
                        clip_norm = self.train_config.get('clip_grad_norm')
                        if clip_norm is not None:
                            self.scaler.unscale_(self.optimizer) # Unscale first
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
                        # Optimizer step
                        self.scaler.step(self.optimizer)
                        # Update scaler
                        self.scaler.update()
                    else:
                        logging.warning(f"Epoch {epoch}, Batch {batch_count}: Skipping optimizer step due to invalid loss ({loss}).")
            else:
                # Validation: just run closure to get loss
                loss = closure()

            # --- Accumulate and Log Batch Loss ---
            if isinstance(loss, torch.Tensor) and torch.isfinite(loss):
                finite_batch_count += 1
                loss_item = loss.item()
                epoch_loss += loss_item
                # Recompute components outside closure for accurate logging (less efficient but safer)
                # This avoids issues with LBFGS multiple evaluations affecting logged components
                with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_amp):
                     try:
                         model_outputs_log = self.model(model_input_state, mode='predict_state')
                         if isinstance(model_outputs_log, dict) and 'state' in model_outputs_log and 'derivative' in model_outputs_log:
                              physics_params_log = self.physics_params.copy()
                              physics_params_log['da_params'] = self.physics_params.get('drainage_area_kwargs', {})
                              physics_loss_log = compute_pde_residual_dual_output(model_outputs_log, physics_params_log)
                              _, loss_components_log = compute_total_loss(
                                   model_outputs_log['state'], final_topo, physics_loss_log, model_outputs_log['state'],
                                   self.physics_params, self.loss_weight_scheduler.get_weights(epoch)
                              )
                              for key, value in loss_components_log.items():
                                   if isinstance(value, (int, float)) and not isinstance(value, bool) and not np.isnan(value):
                                        epoch_loss_components[key] = epoch_loss_components.get(key, 0.0) + value
                         else:
                              logging.warning("Could not recompute loss components for logging: Model output invalid.")
                     except Exception as e_log:
                          logging.warning(f"Error recomputing loss components for logging: {e_log}")


                progress_bar.set_postfix({'loss': f"{loss_item:.4e}"})
            else:
                progress_bar.set_postfix({'loss': "NaN"})


        # --- End of Epoch ---
        progress_bar.close()
        if batch_count == 0:
             logging.warning(f"Epoch {epoch} {'Train' if is_training else 'Val'}: No batches processed.")
             return 0.0, {}
        if finite_batch_count == 0:
             logging.error(f"Epoch {epoch} {'Train' if is_training else 'Val'}: All batches resulted in non-finite loss.")
             return float('nan'), {}

        avg_loss = epoch_loss / finite_batch_count # Average over finite batches only
        avg_loss_components = {key: value / finite_batch_count for key, value in epoch_loss_components.items()}

        return avg_loss, avg_loss_components


    def train(self):
        """Main training loop."""
        epochs = self.max_epochs
        log_interval = self.train_config.get('log_interval', 1) # Log every epoch
        val_interval = self.train_config.get('val_interval', 1)
        save_best_only = self.train_config.get('save_best_only', True)
        save_interval = self.train_config.get('save_interval', 10)

        logging.info(f"Starting training from epoch {self.start_epoch} for {epochs} epochs.")

        for epoch in range(self.start_epoch, epochs):
            epoch_start_time = time.time()

            # --- Training Epoch ---
            train_loss, train_loss_components = self._run_epoch(epoch, is_training=True)
            logging.info(f"Epoch {epoch} Train Loss: {train_loss:.6f}")
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            for name, value in train_loss_components.items():
                self.writer.add_scalar(f'LossComponents/Train/{name}', value, epoch)

            # --- Validation Epoch ---
            val_loss = float('inf')
            if self.val_loader and (epoch + 1) % val_interval == 0:
                val_loss, val_loss_components = self._run_epoch(epoch, is_training=False)
                logging.info(f"Epoch {epoch} Val Loss: {val_loss:.6f}")
                self.writer.add_scalar('Loss/Val', val_loss, epoch)
                for name, value in val_loss_components.items():
                    self.writer.add_scalar(f'LossComponents/Val/{name}', value, epoch)

                # LR Scheduler Step (on validation loss)
                if self.scheduler and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar('LearningRate', current_lr, epoch)
                    logging.debug(f"Epoch {epoch} LR (on plateau): {current_lr:.6e}")

                # Checkpoint Saving (Best Model)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    logging.info(f"New best validation loss: {self.best_val_loss:.6f}")
                    self.save_checkpoint(epoch, 'best_model.pth', is_best=True)

            # --- Periodic Checkpoint Saving ---
            if not save_best_only and (epoch + 1) % save_interval == 0:
                 self.save_checkpoint(epoch, f'epoch_{epoch:04d}.pth')

            # --- LR Scheduler Step (Epoch-based) ---
            if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('LearningRate', current_lr, epoch)
                if current_lr != old_lr: logging.info(f"Epoch {epoch} LR changed: {old_lr:.6e} -> {current_lr:.6e}")
                elif (epoch + 1) % 10 == 0: logging.debug(f"Epoch {epoch} LR: {current_lr:.6e}")


            epoch_duration = time.time() - epoch_start_time
            logging.info(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds.")
            # Flush writer periodically
            if (epoch + 1) % 10 == 0:
                 self.writer.flush()


        self.writer.close()
        logging.info("Training finished.")
        logging.info(f"Best Validation Loss: {self.best_val_loss:.6f}")

    def save_checkpoint(self, epoch: int, filename: str, is_best: bool = False):
        """Saves model checkpoint."""
        # Resolve OmegaConf config if necessary
        try:
            from omegaconf import OmegaConf, DictConfig
            if isinstance(self.config, DictConfig):
                resolved_config = OmegaConf.to_container(self.config, resolve=True, throw_on_missing=False)
            else:
                resolved_config = self.config # Assume standard dict
        except ImportError:
             resolved_config = self.config # Save as is if OmegaConf not available

        state = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': resolved_config
        }
        if self.scheduler: state['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.use_amp: state['amp_scaler_state_dict'] = self.scaler.state_dict()

        filepath = os.path.join(self.checkpoint_dir, filename)
        try:
            # 直接使用 weights_only=False，因为我们的数据包含 numpy 数组
            torch.save(state, filepath, weights_only=False)
            logging.info(f"Checkpoint saved to {filepath}")
            if is_best:
                 best_filepath = os.path.join(self.checkpoint_dir, 'best_model.pth')
                 # 直接使用 weights_only=False，因为我们的数据包含 numpy 数组
                 torch.save(state, best_filepath, weights_only=False) # Overwrite best model
                 logging.info(f"Best model checkpoint updated: {best_filepath}")
        except Exception as e:
             logging.error(f"Failed to save checkpoint {filepath}: {e}", exc_info=True)


    def load_checkpoint(self, filepath: str):
        """Loads model checkpoint."""
        if not os.path.exists(filepath):
            logging.error(f"Checkpoint file not found: {filepath}")
            return

        try:
            # 加载检查点（使用 weights_only=True 增强安全性）
            # 直接使用 weights_only=False，因为我们的数据包含 numpy 数组
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint.get('epoch', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                try: self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e: logging.warning(f"Could not load scheduler state: {e}")
            if self.use_amp and 'amp_scaler_state_dict' in checkpoint:
                 try: self.scaler.load_state_dict(checkpoint['amp_scaler_state_dict'])
                 except Exception as e: logging.warning(f"Could not load AMP scaler state: {e}")

            logging.info(f"Checkpoint loaded from {filepath}. Resuming from epoch {self.start_epoch}.")
            # Optionally re-apply config from checkpoint? Be careful with compatibility.
            # loaded_config = checkpoint.get('config')
            # if loaded_config: self.config = loaded_config # Overwrite current config?

        except Exception as e:
            logging.error(f"Error loading checkpoint from {filepath}: {e}", exc_info=True)
            self.start_epoch = 0
            self.best_val_loss = float('inf')