import numpy as np
import torch
import matplotlib.pyplot as pl

def interpolate_heights(T, h, z, fill_value=None):
    """
    Interpolates temperature T from an irregular, spatially-varying grid h 
    to a uniform height grid z.
    
    Args:
        T: Tensor of shape [B, H, X, Y] (Temperature)
        h: Tensor of shape [B, H, X, Y] (Heights corresponding to T in km)
        z: Tensor of shape [Z] (1D uniform height grid) 
           OR Tensor of shape [B, Z, X, Y]
           
    Returns:
        T_interp: Tensor of shape [B, Z, X, Y]
    """
    B, H_dim, X, Y = T.shape

    # 1. Permute to move the interpolation dimension (H) to the very end
    # Shapes become [B, X, Y, H]
    T_perm = T.permute(0, 2, 3, 1)
    h_perm = h.permute(0, 2, 3, 1)

    # Flatten spatial and batch dimensions to treat as a large batch of 1D arrays
    # Shapes become [N, H] where N = B * X * Y
    T_flat = T_perm.reshape(-1, H_dim)
    h_flat = h_perm.reshape(-1, H_dim)

    # 2. Format z to match the flattened shape [N, Z]
    if z.ndim == 1:
        Z_dim = z.shape[0]
        # Broadcast the 1D z-grid across all pixels
        z_flat = z.view(1, Z_dim).expand(B * X * Y, Z_dim)
    else:
        Z_dim = z.shape[1]
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, Z_dim)

    # 3. Check for monotonicity (searchsorted requires strictly increasing values)
    # Since optical depth goes from 1.0 down to -4.0, height (km) usually *increases*.
    # If your specific h array happens to decrease, we must flip it.
    if h_flat[0, 0] > h_flat[0, -1]:
        h_flat = torch.flip(h_flat, dims=[1])
        T_flat = torch.flip(T_flat, dims=[1])

    # 4. Find bounding indices for interpolation
    # searchsorted finds the index where each z would be inserted to maintain order
    idx = torch.searchsorted(h_flat, z_flat)

    # Clip indices to ensure we stay within valid bounds for gathering (1 to H-1)
    idx = torch.clamp(idx, 1, H_dim - 1)

    idx_left = idx - 1
    idx_right = idx

    # 5. Gather the surrounding heights and temperatures based on indices
    h_left = torch.gather(h_flat, dim=1, index=idx_left)
    h_right = torch.gather(h_flat, dim=1, index=idx_right)
    T_left = torch.gather(T_flat, dim=1, index=idx_left)
    T_right = torch.gather(T_flat, dim=1, index=idx_right)

    # 6. Compute interpolation weights
    # 1e-8 added to prevent division by zero in case of identical adjacent heights
    weights = (z_flat - h_left) / (h_right - h_left + 1e-8)

    # Optional: If you want constant extrapolation outside bounds (instead of linear),
    # uncomment the following line:
    weights = torch.clamp(weights, 0.0, 1.0)

    # 7. Apply linear interpolation
    T_interp_flat = T_left + weights * (T_right - T_left)

    # Remove interpolated values
    if fill_value is not None:
        # Since h_flat is now strictly increasing, index 0 is min and -1 is max
        h_min = h_flat[:, 0:1]
        h_max = h_flat[:, -1:]
        
        # Mask where the target z is outside the available h bounds
        out_of_bounds = (z_flat < h_min) | (z_flat > h_max)
        
        # Replace out-of-bounds indices with the fill_value
        fill_tensor = torch.tensor(fill_value, dtype=T_interp_flat.dtype, device=T_interp_flat.device)
        T_interp_flat = torch.where(out_of_bounds, fill_tensor, T_interp_flat)

    # 8. Reshape back and permute to [B, Z, X, Y]
    T_interp = T_interp_flat.reshape(B, X, Y, Z_dim).permute(0, 3, 1, 2)

    return T_interp


