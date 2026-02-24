## Update Note =================================================================
# 2025.7.21: Updated to v_0.0.4
#  - Review the code again
# =============================================================================

# =============================================================================
# 3D SinGAN PyTorch Implementation
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import argparse
import logging

try:
    import matplotlib.pyplot as plt
except ImportError:
    logging.info("matplotlib not found. Install if needed for visualization.")

# =============================================================================
# Utility Functions
# =============================================================================
def initialize_model(model, scale=1.):
    """
    Initializes weights for Conv and BatchNorm layers with specific distributions.
    Applied recursively to all modules in a model.
    """
    for m in model.modules():
        # Initialize Conv weights with a normal distribution
        if isinstance(m, nn.Conv3d):
            m.weight.data.normal_(0.0, 0.02)
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        # Initialize BatchNorm weights and biases
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.normal_(1.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        # Other layer types (like Linear, etc.) are not initialized by this function
        else:
            continue


def calculate_gradient_penalty(netD, real_data, fake_data, device, lambda_gp):
    #... (implementation as in Section 4.2)...
    batch_size = real_data.size(0) # Should be 1 for SinGAN
    alpha = torch.rand(batch_size, 1, 1, 1, 1, device=device) # Expand for 5D tensor

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                                  create_graph=True, retain_graph=True, only_inputs=True)

    gradients = gradients[0].view(gradients[0].size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty

# def generate_noise(size, device):
#     #... (implementation as in Section 4.2)...
#     # size = (channels, depth, height, width)
#     noise = torch.randn(1, 1, size, size, size, device=device)
#     return noise
def generate_noise(size_tuple_cdhw, device, batch_size=1): # Renamed 'size' to be more explicit
    """
    Generates a 5D noise tensor.
    The original comment was "size = (channels, depth, height, width)".
    This function interprets the 'size_tuple_cdhw' argument as that tuple.

    Args:
        size_tuple_cdhw (tuple): A tuple representing (channels, depth, height, width).
        device: The torch device to create the tensor on.
        batch_size (int): The batch size for the noise. Defaults to 1.
    Returns:
        torch.Tensor: A noise tensor of shape (batch_size, C, D, H, W).
    """
    if not (isinstance(size_tuple_cdhw, tuple) and len(size_tuple_cdhw) == 4):
        raise ValueError("size_tuple_cdhw must be a tuple of (channels, depth, height, width)")

    # Unpack the shape tuple for torch.randn
    # Shape will be (batch_size, channels, depth, height, width)
    noise = torch.randn(batch_size, *size_tuple_cdhw, device=device)
    return noise

def upsample_3d(x, scale_factor=2.0):
    #... (implementation as in Section 4.2)...
    return F.interpolate(x, scale_factor=scale_factor, mode='trilinear', align_corners=False)

def downsample_3d(x, scale_factor=0.5):
    #... (implementation as in Section 4.2)...
    return F.interpolate(x, scale_factor=scale_factor, mode='trilinear', align_corners=False, recompute_scale_factor=False)

def create_scale_pyramid(real_vol, scale_factor_base, min_size, device):
    #... (implementation as in Section 4.2, using size interpolation)...
    pyramid = []
    current_vol = real_vol.clone() # Clone to avoid modifying original
    target_scale_factor = scale_factor_base # This is the factor for *downsampling*

    # Add original volume (finest scale) first
    pyramid.append(current_vol)

    # Calculate dimensions iteratively
    current_dims = np.array(current_vol.shape[-3:])
    while True:
        # Calculate next scale dimensions using ceiling
        next_dims_float = current_dims * target_scale_factor
        # Use ceiling for dimensions, ensure minimum size of 1
        next_dims = np.maximum(1, np.ceil(next_dims_float)).astype(int)

        # Check if minimum dimension is reached
        if min(next_dims) <= min_size:
            break

        # Downsample using interpolate with target size
        current_vol = F.interpolate(current_vol, size=tuple(next_dims), mode='trilinear', align_corners=False)
        pyramid.append(current_vol)
        current_dims = next_dims

    return pyramid[::-1] # Return coarsest to finest

def load_real_volume(path, device):
    # Example loading function (adapt based on your file format)
    if path.endswith('.npy'):
        vol = np.load(path)
    else:
        raise ValueError("Unsupported file format. Please use.npy or.nii/.nii.gz or adapt the loading function.")

    # Ensure single channel and correct dimension order (C, D, H, W)
    if vol.ndim == 3: # Add channel dimension if missing
        vol = vol[np.newaxis,...]
    elif vol.ndim == 4: # Assume (D, H, W, C) or (C, D, H, W) - adjust if needed
        if vol.shape[-1] == 1: # Assume (D, H, W, C), permute
             vol = np.transpose(vol, (3, 0, 1, 2))
        elif vol.shape[0] != 1: # If first dim is not 1, and it's 4D, assume it's (D, H, W, C) where C>1
             raise ValueError("Input volume has 4 dimensions and the first dimension is not 1 (channel). Please provide single channel data (C,D,H,W or D,H,W,C=1).")
        # If shape[0] == 1, assume it's already (C, D, H, W)
    else:
        raise ValueError(f"Unsupported number of dimensions: {vol.ndim}. Expected 3 or 4.")

    # Convert to PyTorch tensor
    vol_tensor = torch.from_numpy(vol.astype(np.float32))

    # Normalize to [-1, 1] (adjust if your data has a different range)
    min_val = torch.min(vol_tensor)
    max_val = torch.max(vol_tensor)
    if max_val > min_val:
        vol_tensor = 2 * ((vol_tensor - min_val) / (max_val - min_val)) - 1
    else: # Handle constant volume case
        vol_tensor = torch.zeros_like(vol_tensor)

    # Add batch dimension (B, C, D, H, W)
    vol_tensor = vol_tensor.unsqueeze(0).to(device)
    return vol_tensor

def save_snapshot(vol_tensor, path):
    # Example saving function (adapt as needed)
    vol_np = vol_tensor.squeeze(0).squeeze(0).cpu().numpy() # Remove Batch and Channel
    plt.figure(figsize=(10,5))
    plt.imshow(vol_np.squeeze()[0])
    plt.savefig(path)
    plt.close()

def save_volume(vol_tensor, path):
    # Example saving function (adapt as needed)
    vol_np = vol_tensor.squeeze(0).squeeze(0).cpu().numpy() # Remove Batch and Channel
    # Denormalize if needed before saving (assuming original range was [0, X])
    # vol_np = (vol_np + 1) / 2 * MAX_ORIGINAL_VALUE
    if path.endswith('.npy'):
        np.save(path, vol_np)
    else:
        logging.warning(f"Unsupported save format for {path}. Saving as .npy")
        np.save(path.replace(os.path.splitext(path)[1], '.npy'), vol_np)
