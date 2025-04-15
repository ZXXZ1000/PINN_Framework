# PINN_Framework/src/physics.py
"""
核心物理计算模块，包括地形导数、汇水面积 (IDA/D∞)、侵蚀和扩散。
"""

import torch
import torch.nn.functional as F
import math
import time
import logging
import warnings
import time
from typing import Dict, Tuple, Union, Optional, List
import inspect # Import inspect for checking function signature

# --- Helper Functions for IDA/D∞ Drainage Area ---

def _map_to_1d(r: int, c: int, width: int) -> int:
    """Maps 2D grid coordinates (row, col) to 1D index."""
    return r * width + c

def _get_neighbor_coords(r: int, c: int, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the valid neighbor coordinates for cell (r,c) in a regular grid.
    Indices 0-7 correspond to [E, NE, N, NW, W, SW, S, SE].
    """
    offsets = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
    coords = []
    codes = []
    for i, (dr, dc) in enumerate(offsets):
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            coords.append((nr, nc))
            codes.append(i)
    # Return as torch tensor for consistency if needed
    return torch.tensor(coords, dtype=torch.long), torch.tensor(codes, dtype=torch.long) # Ensure long type for indexing

def _calculate_dinf_weights_sparse(
    h: torch.Tensor, dx: float, dy: float, eps: float = 1e-10
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates D-infinity flow directions and partitions flow.
    Returns sparse triplets (rows, cols, vals) for the W matrix construction.
    rows: 1D indices of receiving cells.
    cols: 1D indices of source cells.
    vals: Flow fraction values (w_ji) from cell j to cell i.
    """
    batch_size, _, height, width = h.shape
    N_grid = height * width
    device = h.device
    dtype = h.dtype

    h_padded = F.pad(h, (1, 1, 1, 1), mode='reflect')
    d_cardinal = dx
    d_diagonal = math.sqrt(dx * dx + dy * dy)
    neighbor_offsets = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
    neighbor_distances = torch.tensor(
        [d_cardinal, d_diagonal, dy, d_diagonal, d_cardinal, d_diagonal, dy, d_diagonal],
        device=device, dtype=dtype
    )
    neighbor_angles = torch.tensor([
        0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi,
        5*math.pi/4, 3*math.pi/2, 7*math.pi/4
    ], device=device, dtype=dtype)

    slopes = torch.zeros(batch_size, 8, height, width, device=device, dtype=dtype)
    h_center = h_padded[:, :, 1:-1, 1:-1]
    for k, (dr, dc) in enumerate(neighbor_offsets):
        h_neighbor = h_padded[:, :, 1 + dr: height + 1 + dr, 1 + dc: width + 1 + dc]
        slopes[:, k, :, :] = (h_center - h_neighbor) / (neighbor_distances[k] + eps)
    slopes = torch.clamp(slopes, min=0)
    total_positive_slope = torch.sum(slopes, dim=1, keepdim=True)

    s_e, s_ne, s_n, s_nw, s_w, s_sw, s_s, s_se = [slopes[:, i, :, :].unsqueeze(1) for i in range(8)]
    norm_factor_x = dx * (1 + math.sqrt(2)) + eps
    norm_factor_y = dy * (1 + math.sqrt(2)) + eps
    e0 = ((s_e + s_ne / math.sqrt(2) + s_se / math.sqrt(2)) / norm_factor_x -
          (s_w + s_nw / math.sqrt(2) + s_sw / math.sqrt(2)) / norm_factor_x)
    e1 = ((s_n + s_ne / math.sqrt(2) + s_nw / math.sqrt(2)) / norm_factor_y -
          (s_s + s_se / math.sqrt(2) + s_sw / math.sqrt(2)) / norm_factor_y)
    alpha = torch.atan2(e1, e0)

    row_indices, col_indices, weight_values = [], [], []
    neighbor_maps = {} # Cache neighbor lookups

    for b in range(batch_size):
        for r in range(height):
            for c in range(width):
                cell_idx_1d = _map_to_1d(r, c, width) + b * N_grid
                if total_positive_slope[b, 0, r, c] < eps: continue

                cell_slopes = slopes[b, :, r, c]
                cell_alpha = alpha[b, 0, r, c] % (2 * math.pi)
                sector_float = cell_alpha / (math.pi / 4)
                sector_idx = int(math.floor(sector_float)) % 8
                idx1, idx2 = sector_idx, (sector_idx + 1) % 8
                angle1, angle2 = neighbor_angles[idx1], neighbor_angles[idx2] if idx1 < 7 else 2 * math.pi
                total_angle = (angle2 - angle1) if angle2 > angle1 else (2 * math.pi - angle1 + angle2)
                relative_alpha = (cell_alpha - angle1) % (2 * math.pi)

                p2 = relative_alpha / total_angle if total_angle > eps else 0.5
                p1 = 1.0 - p2

                if cell_slopes[idx1] < eps and cell_slopes[idx2] >= eps: p1, p2 = 0.0, 1.0
                elif cell_slopes[idx2] < eps and cell_slopes[idx1] >= eps: p1, p2 = 1.0, 0.0

                if (r, c) not in neighbor_maps:
                    coords, codes = _get_neighbor_coords(r, c, height, width)
                    neighbor_maps[(r, c)] = {int(co.item()): (int(crds[0].item()), int(crds[1].item())) for co, crds in zip(codes, coords)}
                cell_neighbor_map = neighbor_maps[(r, c)]

                if p1 > eps and idx1 in cell_neighbor_map:
                    nr1, nc1 = cell_neighbor_map[idx1]
                    neighbor1_idx_1d = _map_to_1d(nr1, nc1, width) + b * N_grid
                    row_indices.append(neighbor1_idx_1d)
                    col_indices.append(cell_idx_1d)
                    weight_values.append(p1)
                if p2 > eps and idx2 in cell_neighbor_map:
                    nr2, nc2 = cell_neighbor_map[idx2]
                    neighbor2_idx_1d = _map_to_1d(nr2, nc2, width) + b * N_grid
                    row_indices.append(neighbor2_idx_1d)
                    col_indices.append(cell_idx_1d)
                    weight_values.append(p2)

    rows = torch.tensor(row_indices, dtype=torch.long, device=device)
    cols = torch.tensor(col_indices, dtype=torch.long, device=device)
    vals = torch.tensor(weight_values, dtype=dtype, device=device) if weight_values else torch.tensor([], dtype=dtype, device=device)

    return rows, cols, vals

def _build_W_matrix_torch_sparse(
    rows: torch.Tensor, cols: torch.Tensor, w_ji_vals: torch.Tensor,
    N_total: int, device, dtype
) -> torch.Tensor:
    """Builds the sparse IDA matrix W (COO format). W[i,i]=1, W[i,j]=-w_ji."""
    W_indices_offdiag = torch.stack([rows, cols])
    W_vals_offdiag = -w_ji_vals # Note the negative sign
    diag_indices = torch.arange(N_total, device=device, dtype=torch.long)
    W_indices_diag = torch.stack([diag_indices, diag_indices])
    W_vals_diag = torch.ones(N_total, device=device, dtype=dtype)
    all_indices = torch.cat([W_indices_diag, W_indices_offdiag], dim=1)
    all_vals = torch.cat([W_vals_diag, W_vals_offdiag])
    W_sparse = torch.sparse_coo_tensor(all_indices, all_vals, (N_total, N_total),
                                         device=device, dtype=dtype).coalesce()
    return W_sparse

def _richardson_solver(
    W: torch.Tensor,  # Sparse matrix (N, N)
    b: torch.Tensor,  # Dense vector (N,)
    a_init: Optional[torch.Tensor] = None,
    omega: float = 0.5,
    max_iters: int = 2000,
    tol: float = 1e-6,
    verbose: bool = False
) -> Tuple[torch.Tensor, bool]:
    """
    Solves Wa = b using Richardson iteration (Simpler version from Drainage_area_cal.py).
    """
    N = b.shape[0]
    device = b.device
    dtype = b.dtype
    if a_init is None:
        a = torch.zeros(N, device=device, dtype=dtype)
    else:
        a = a_init.clone()

    b_norm = torch.linalg.norm(b)
    if b_norm < 1e-15:
        return a, True

    converged = False
    # Check diagonal (optional, from Drainage_area_cal.py)
    # W_diag_indices = W._indices()[0] == W._indices()[1]
    # W_diag_values = W._values()[W_diag_indices]
    # if not torch.all(W_diag_values == 1.0):
    #     warnings.warn("W matrix diagonal elements are not all 1.0. This may cause instability.", RuntimeWarning)

    for k in range(max_iters):
        # Compute residual r = b - W * a
        Wa = torch.sparse.mm(W, a.unsqueeze(1)).squeeze(1)
        r = b - Wa

        r_norm = torch.linalg.norm(r)
        rel_res = r_norm / (b_norm + tol)

        # Check for numerical issues
        if torch.isnan(r_norm) or torch.isinf(r_norm):
            warnings.warn(f"Numerical instability detected at iteration {k+1}. Stopping early.", RuntimeWarning)
            return a, False # Return current 'a'

        if verbose and (k % 50 == 0 or k == max_iters - 1):
            logging.info(f"Richardson Iteration {k+1}/{max_iters} - Relative Residual: {rel_res:.3e}")

        if rel_res < tol:
            converged = True
            if verbose:
                logging.info(f"Richardson solver converged in {k+1} iterations.")
            break

        # Update: a = a + omega * r
        a += omega * r

    if not converged:
        warnings.warn(f"Richardson solver did not converge in {max_iters} iterations. Final relative residual: {rel_res:.3e}", RuntimeWarning)

    # Clamp at the end
    a = torch.clamp(a, min=0.0)
    return a, converged

class IDASolveRichardson(torch.autograd.Function):
    """Custom autograd function for IDA Richardson solver."""
    @staticmethod
    def forward(ctx, W_rows, W_cols, W_vals_offdiag, b_flat, N_total, omega, solver_max_iters, solver_tol, verbose): # Signature without stabilize/max_value
        device, dtype = b_flat.device, b_flat.dtype
        W_sparse = _build_W_matrix_torch_sparse(W_rows, W_cols, W_vals_offdiag, N_total, device, dtype)
        # Call the simpler solver
        a_flat, _ = _richardson_solver(
            W_sparse, b_flat, None, omega, solver_max_iters, solver_tol, verbose=verbose
        )
        ctx.W_sparse = W_sparse
        ctx.a = a_flat
        ctx.W_rows, ctx.W_cols = W_rows, W_cols # Save indices for grad_W calculation
        ctx.omega, ctx.solver_max_iters, ctx.solver_tol = omega, solver_max_iters, solver_tol
        ctx.verbose = verbose # Save correct context
        return a_flat

    @staticmethod
    def backward(ctx, grad_output):
        W_sparse, a, W_rows, W_cols = ctx.W_sparse, ctx.a, ctx.W_rows, ctx.W_cols
        omega, solver_max_iters, solver_tol = ctx.omega, ctx.solver_max_iters, ctx.solver_tol
        verbose = ctx.verbose # Load correct context

        grad_output_detached = grad_output.detach()
        # Call simpler solver for backward pass
        grad_a, converged_bwd = _richardson_solver(
            W_sparse.T.coalesce(), grad_output_detached, None, omega, solver_max_iters, solver_tol, verbose=verbose
        )
        if not converged_bwd:
            warnings.warn("Richardson solver for backward pass did not converge!", RuntimeWarning)
            grad_a = torch.zeros_like(grad_output_detached) # Return zero gradient on failure

        grad_b = grad_a # Gradient w.r.t. b
        # Gradient w.r.t. W_vals_offdiag (which are -w_ji)
        # dL/d(-w_ji) = dL/dW_ij = - grad_a[i] * a[j]
        grad_W_vals_offdiag = -grad_a[W_rows] * a[W_cols]

        # Return correct number of gradients for forward inputs
        return None, None, grad_W_vals_offdiag, grad_b, None, None, None, None, None

# --- Main IDA/D∞ Drainage Area Function ---

def calculate_drainage_area_ida_dinf_torch(
    h: torch.Tensor,
    dx: float,
    dy: float,
    precip: Union[float, torch.Tensor] = 1.0,
    omega: float = 0.5, # Default from simpler solver
    solver_max_iters: int = 2000, # Default from simpler solver
    solver_tol: float = 1e-6,   # Default from simpler solver
    eps: float = 1e-10,
    verbose: bool = False
    # Removed max_value and stabilize parameters from signature
) -> torch.Tensor:
    """
    Calculates differentiable drainage area using the IDA framework and D-infinity flow routing (PyTorch).
    """
    if not (isinstance(h, torch.Tensor) and h.ndim == 4 and h.shape[1] == 1):
        raise ValueError("Input h must be a 4D tensor (B, 1, H, W)")
    if dx <= 0 or dy <= 0: raise ValueError("dx and dy must be positive.")

    start_time_total = time.time()
    device, dtype = h.device, h.dtype
    batch_size, _, height, width = h.shape
    N_grid = height * width
    N_total = batch_size * N_grid

    if verbose: logging.info(f"Starting IDA-Dinf: grid={height}x{width}, batch={batch_size}, device={device}")

    # --- Construct b vector ---
    cell_area = dx * dy
    if isinstance(precip, (int, float)):
        b_grid = torch.full((batch_size, height, width), float(precip) * cell_area, dtype=dtype, device=device)
    elif isinstance(precip, torch.Tensor):
        precip = precip.to(device=device, dtype=dtype)
        if precip.shape == h.shape: b_grid = precip.squeeze(1) * cell_area
        elif precip.numel() == 1: b_grid = torch.full((batch_size, height, width), precip.item() * cell_area, dtype=dtype, device=device)
        else: raise ValueError(f"Precip shape {precip.shape} incompatible with h shape {h.shape}")
    else: raise TypeError(f"Unsupported precip type: {type(precip)}")
    b_flat = b_grid.view(N_total)

    # --- Calculate D-infinity weights ---
    if verbose: logging.info("Calculating D-infinity weights...")
    start_time_weights = time.time()
    rows, cols, w_ji_vals = _calculate_dinf_weights_sparse(h, dx, dy, eps)
    W_vals_offdiag = -w_ji_vals # W_ij = -w_ji for i != j
    if verbose: logging.info(f"Weight calculation took {time.time() - start_time_weights:.3f}s")

    # --- Solve Wa = b using custom autograd function ---
    if verbose: logging.info("Solving Wa = b using Richardson iteration...")
    start_time_solve = time.time()

    # Call apply with the simplified signature
    a_flat = IDASolveRichardson.apply(
        rows, cols, W_vals_offdiag, b_flat, N_total,
        omega, solver_max_iters, solver_tol, verbose
    )

    # Check if the result contains NaN or inf values
    if torch.isnan(a_flat).any() or torch.isinf(a_flat).any():
        if verbose: logging.warning("NaN or Inf values detected in IDA solution. Replacing with zeros.")
        a_flat = torch.where(torch.isnan(a_flat) | torch.isinf(a_flat), torch.zeros_like(a_flat), a_flat)

    if verbose: logging.info(f"Linear solver took {time.time() - start_time_solve:.3f}s")

    # --- Reshape solution ---
    drainage_area = a_flat.view(batch_size, 1, height, width)

    if verbose: logging.info(f"Total IDA-Dinf time: {time.time() - start_time_total:.3f}s")
    return drainage_area


# --- Terrain Derivatives (Copied from previous version, needed by components below) ---

def get_sobel_kernels(dx, dy):
    """Gets Sobel kernels for gradient calculation, scaled by grid spacing."""
    dx_float, dy_float = float(dx), float(dy)
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / (8.0 * dx_float)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32) / (8.0 * dy_float)
    return kernel_x.view(1, 1, 3, 3), kernel_y.view(1, 1, 3, 3)

def calculate_slope_magnitude(h, dx, dy, padding_mode='replicate'):
    """
    Calculates the magnitude of the terrain slope |∇h| using Sobel operators.
    """
    kernel_x, kernel_y = get_sobel_kernels(dx, dy)
    kernel_x = kernel_x.to(device=h.device, dtype=h.dtype)
    kernel_y = kernel_y.to(device=h.device, dtype=h.dtype)
    pad_size = 1
    h_padded = F.pad(h, (pad_size, pad_size, pad_size, pad_size), mode=padding_mode)
    dzdx = F.conv2d(h_padded, kernel_x, padding=0)
    dzdy = F.conv2d(h_padded, kernel_y, padding=0)
    slope_mag = torch.sqrt(dzdx**2 + dzdy**2) # No epsilon needed for stability
    return slope_mag

def get_laplacian_kernel(dx, dy):
    """Gets the 5-point finite difference kernel for the Laplacian."""
    kernel_dxx = torch.tensor([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=torch.float32) / (dx**2)
    kernel_dyy = torch.tensor([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=torch.float32) / (dy**2)
    return kernel_dxx.view(1, 1, 3, 3), kernel_dyy.view(1, 1, 3, 3)

def calculate_laplacian(h, dx, dy, padding_mode='replicate'):
    """
    Calculates the Laplacian ∇²h using a 5-point finite difference stencil. Handles dx != dy.
    """
    kernel_dxx, kernel_dyy = get_laplacian_kernel(dx, dy)
    kernel_dxx = kernel_dxx.to(device=h.device, dtype=h.dtype)
    kernel_dyy = kernel_dyy.to(device=h.device, dtype=h.dtype)
    pad_size = 1
    h_padded = F.pad(h, (pad_size, pad_size, pad_size, pad_size), mode=padding_mode)
    lap_x = F.conv2d(h_padded, kernel_dxx, padding=0)
    lap_y = F.conv2d(h_padded, kernel_dyy, padding=0)
    laplacian = lap_x + lap_y
    return laplacian

# --- Physics Components (Stream Power, Diffusion) ---

def stream_power_erosion(h, drainage_area, slope_magnitude, K_f, m, n):
    """Calculates erosion rate: E = K_f * A^m * S^n"""
    epsilon = 1e-10
    if not isinstance(K_f, torch.Tensor): K_f = torch.tensor(K_f, device=h.device, dtype=h.dtype)
    if K_f.ndim == 0: K_f = K_f.view(1, 1, 1, 1)
    elif K_f.ndim == 1 and K_f.shape[0] == h.shape[0]: K_f = K_f.view(-1, 1, 1, 1)
    # Ensure drainage_area and slope_magnitude are non-negative before exponentiation
    drainage_area_safe = torch.clamp(drainage_area, min=0.0)
    slope_magnitude_safe = torch.clamp(slope_magnitude, min=0.0)
    erosion_rate = K_f * (drainage_area_safe + epsilon)**m * (slope_magnitude_safe + epsilon)**n
    return erosion_rate

def hillslope_diffusion(h, K_d, dx, dy, padding_mode='replicate'):
    """Calculates diffusion rate: D = Kd * Laplacian(h)"""
    laplacian_h = calculate_laplacian(h, dx, dy, padding_mode=padding_mode)
    if not isinstance(K_d, torch.Tensor): K_d = torch.tensor(K_d, device=h.device, dtype=h.dtype)
    if K_d.ndim == 0: K_d = K_d.view(1, 1, 1, 1)
    elif K_d.ndim == 1 and K_d.shape[0] == h.shape[0]: K_d = K_d.view(-1, 1, 1, 1)
    diffusion_rate = K_d * laplacian_h
    return diffusion_rate

# --- Combined PDE Right Hand Side (Modified to use IDA/D∞) ---

def calculate_dhdt_physics(
    h: torch.Tensor,
    U: Union[float, torch.Tensor],
    K_f: Union[float, torch.Tensor],
    m: float,
    n: float,
    K_d: Union[float, torch.Tensor],
    dx: float,
    dy: float,
    precip: Union[float, torch.Tensor] = 1.0,
    padding_mode: str = 'replicate',
    da_params: Optional[Dict] = None # Changed name to reflect it's for DA
) -> torch.Tensor:
    """
    Calculates the physics-based time derivative of elevation (RHS of the PDE).
    dh/dt = U - E + D = U - K_f * A^m * S^n + K_d * Laplacian(h)
    Uses the IDA/D-infinity drainage area calculation.

    Args:
        h: Current topography (B, 1, H, W).
        U: Uplift rate (scalar, [B], [H,W], [B,H,W], or [B,1,H,W]).
        K_f: Stream power erodibility (scalar, [B], [H,W], [B,H,W], or [B,1,H,W]).
        m: Stream power area exponent.
        n: Stream power slope exponent.
        K_d: Hillslope diffusivity (scalar, [B], [H,W], [B,H,W], or [B,1,H,W]).
        dx: Grid spacing x.
        dy: Grid spacing y.
        precip: Precipitation for drainage area calculation (scalar or tensor).
        padding_mode: Padding mode for derivatives.
        da_params: Parameters dictionary for calculate_drainage_area_ida_dinf_torch
                   (e.g., {'omega': 0.5, 'solver_max_iters': 2000, 'solver_tol': 1e-6}).

    Returns:
        The calculated dh/dt based on physics (B, 1, H, W).
    """
    # Ensure U is a tensor with compatible shape
    if not isinstance(U, torch.Tensor): U = torch.tensor(U, device=h.device, dtype=h.dtype)
    if U.ndim == 0: U = U.view(1, 1, 1, 1)
    elif U.ndim == 1 and U.shape[0] == h.shape[0]: U = U.view(-1, 1, 1, 1)
    # Add checks/expansion for spatial U if needed, or rely on prepare_parameter if used upstream

    # Calculate slope magnitude
    slope_mag = calculate_slope_magnitude(h, dx, dy, padding_mode=padding_mode)

    # Calculate drainage area using the IDA/D-infinity method
    if da_params is None: da_params = {}
    # Correctly define ida_dinf_kwargs using defaults from the simplified function signature
    ida_dinf_kwargs = {
        'omega': da_params.get('omega', 0.5), # Default from simpler solver
        'solver_max_iters': da_params.get('solver_max_iters', 2000), # Default from simpler solver
        'solver_tol': da_params.get('solver_tol', 1e-6), # Default from simpler solver
        'eps': da_params.get('eps', 1e-10),
        'verbose': da_params.get('verbose', False)
    }
    # Filter out any unexpected keys just in case da_params contains extras
    sig = inspect.signature(calculate_drainage_area_ida_dinf_torch)
    valid_kwargs = {k: v for k, v in ida_dinf_kwargs.items() if k in sig.parameters}

    drainage_area = calculate_drainage_area_ida_dinf_torch(
        h, dx, dy, precip=precip, **valid_kwargs
    )

    # Calculate erosion rate
    erosion_rate = stream_power_erosion(h, drainage_area, slope_mag, K_f, m, n)

    # Calculate diffusion rate
    diffusion_rate = hillslope_diffusion(h, K_d, dx, dy, padding_mode=padding_mode)

    # Combine terms: dh/dt = U - E + D
    dhdt = U - erosion_rate + diffusion_rate

    return dhdt
