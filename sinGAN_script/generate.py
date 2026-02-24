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
    from utilities import generate_noise, upsample_3d
    from models import Generator3D
except:
    import sys
    import os 
    sys.path.insert(0, os.path.join(os.getcwd(), 'script'))
    from utilities import generate_noise, upsample_3d
    from script.models import Generator3D
    
try:
    import matplotlib.pyplot as plt
except ImportError:
    logging.info("matplotlib not found. Install if needed for visualization.")

# =============================================================================
# Generation Function
# =============================================================================
def generate_3d_sample(trained_generators_state_dicts, 
                       # fixed_noise_maps, 
                       pyramid, opt, device, gen_start_scale=0, custom_noise_shape=None):
    num_scales = len(trained_generators_state_dicts)
    generators = []
    # Load generators from state dicts
    for i in range(num_scales):
        netG = Generator3D(opt).to(device)
        # Load state dict corresponding to scale i (0=finest, N=coarsest)
        # Note: trained_generators was returned finest-to-coarsest
        netG.load_state_dict(trained_generators_state_dicts[i])
        netG.eval() # Set to evaluation mode
        generators.append(netG)

    # Determine starting scale index (N = num_scales - 1)
    start_scale_idx_actual = num_scales - 1 - gen_start_scale

    # Generate initial noise at the starting scale
    if custom_noise_shape:
        # Use custom shape (C, D, H, W)
        noise_shape = (opt.nc_im,) + tuple(custom_noise_shape)
    else:
        # Use shape from the corresponding pyramid level
        noise_shape = pyramid[num_scales - 1 - start_scale_idx_actual].shape[1:] # Get C, D, H, W

    current_noise = generate_noise(noise_shape, device)
    current_vol = torch.zeros((1,) + noise_shape, device=device) # Initial previous output is zero

    scale_factor_r = opt.scale_factor

    # Generate through the pyramid from start_scale down to 0 (finest)
    with torch.no_grad():
        for scale_idx in range(start_scale_idx_actual, -1, -1): # Iterate N, N-1,..., start_scale,..., 0
            # Get the generator for this scale (index maps directly: 0=finest, N=coarsest)
            # Need to map scale_idx (N..0) to list index (0..N)
            generator_list_idx = num_scales - 1 - scale_idx
            netG = generators[generator_list_idx]

            # Upsample previous volume
            prev_vol_upsampled = upsample_3d(current_vol, scale_factor=scale_factor_r)

            # Determine target size for this scale
            if custom_noise_shape and scale_idx == start_scale_idx_actual:
                 target_size = noise_shape[-3:]
            elif scale_idx < num_scales -1 : # Not the coarsest scale being generated
                 # Infer target size by scaling up from the next coarser scale's pyramid shape
                 coarser_pyramid_idx = num_scales - 1 - (scale_idx + 1)
                 coarser_dims = np.array(pyramid[coarser_pyramid_idx].shape[-3:])
                 target_dims_float = coarser_dims * scale_factor_r
                 target_size = tuple(np.round(target_dims_float).astype(int))
                 # Ensure minimum size 1
                 target_size = tuple(max(1, d) for d in target_size)
            else: # Coarsest scale being generated (scale_idx == num_scales - 1)
                 target_size = noise_shape[-3:] # Use noise shape directly


            # Resize upsampled volume and noise to target size
            prev_vol_upsampled = F.interpolate(prev_vol_upsampled, size=target_size, mode='trilinear', align_corners=False)
            noise_this_scale = F.interpolate(current_noise, size=target_size, mode='trilinear', align_corners=False)

            # Generate volume for this scale
            current_vol = netG(noise_this_scale, prev_vol_upsampled)

            # Prepare noise for the next finer scale (if any)
            if scale_idx > 0:
                current_noise = generate_noise(target_size, device) # Generate new noise based on current size

    return current_vol