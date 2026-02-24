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
    from utilities import create_scale_pyramid, generate_noise, upsample_3d, initialize_model
    from utilities import save_snapshot, save_volume, calculate_gradient_penalty
    from models import Generator3D, Discriminator3D
except:
    import sys
    import os 
    sys.path.insert(0, os.path.join(os.getcwd(), 'script'))
    from utilities import create_scale_pyramid, generate_noise, upsample_3d, initialize_model
    from utilities import save_snapshot, save_volume, calculate_gradient_penalty
    from script.models import Generator3D, Discriminator3D
try:
    import matplotlib.pyplot as plt
except ImportError:
    logging.info("matplotlib not found. Install if needed for visualization.")

# =============================================================================
# Training Function
# =============================================================================

def train_3d_singan(real_volume, opt, device):
    #... (implementation as in Section 4.7)...
    # 1. Create image pyramid
    scale_factor_base = 1 / opt.scale_factor # For downsampling
    pyramid = create_scale_pyramid(real_volume, scale_factor_base, opt.min_size, device)
    num_scales = len(pyramid)
    logging.info(f"Created pyramid with {num_scales} scales.")
    for i, vol in enumerate(pyramid):
        logging.info(f"  Scale {num_scales - 1 - i} (Coarsest={i==0}): {vol.shape}")

    # 2. Initialize storage
    stored_reconstructions = {} # Key: scale index (N, N-1,... 0)
    trained_generators = [] # List of state dicts, finest to coarsest
    fixed_noise_maps = {} # Key: scale index (N, N-1,... 0)

    # Generate fixed noise z* for reconstruction at coarsest scale (index N = num_scales - 1)
    coarsest_scale_idx = num_scales - 1
    coarsest_shape = pyramid[0].shape # pyramid is coarsest
    fixed_noise_maps[coarsest_scale_idx] = generate_noise(coarsest_shape[1:], device)

    # 3. Loop through scales from Coarsest (index 0 in pyramid) to Finest (index num_scales-1 in pyramid)
    scale_factor_r = opt.scale_factor # For upsampling between scales

    # Ensure output directory exists
    os.makedirs(opt.outdir, exist_ok=True)

    for n in range(num_scales): # n iterates through pyramid list [0, num_scales-1]
        scale_idx_actual = num_scales - 1 - n # Actual scale index (N, N-1,..., 0)
        logging.info(f"\n--- Training Scale {scale_idx_actual} ---")
        real_vol_at_scale = pyramid[n].to(device) # Ensure it's on the correct device
        opt.nc_im = real_vol_at_scale.shape[1] # Set number of input channels for the model
        current_size = real_vol_at_scale.shape[-3:]
        logging.info(f"Volume size: {current_size}")

        # Adjust nfc potentially based on scale (optional, start simple)
        # opt.nfc = max(opt.min_nfc, opt.nfc // (2**(scale_idx_actual//4))) # Example decrease

        # Initialize Generator and Discriminator for this scale
        netG = Generator3D(opt).to(device)
        netD = Discriminator3D(opt).to(device)
        netG.apply(initialize_model)
        netD.apply(initialize_model)

        # Optimizers
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        # Schedulers (adjust milestones/gamma as needed)
        schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizerG, milestones=[opt.niter // 2, opt.niter * 3 // 4], gamma=0.1)
        schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizerD, milestones=[opt.niter // 2, opt.niter * 3 // 4], gamma=0.1)

        # Store zero reconstruction noise for finer scales
        if scale_idx_actual < coarsest_scale_idx:
             fixed_noise_maps[scale_idx_actual] = torch.zeros((1,) + real_vol_at_scale.shape[1:], device=device)

        # Inner training loop for the current scale
        for iter_num in range(opt.niter):
            # --- Discriminator Training ---
            netD.zero_grad()
            # Real data loss
            output_real = netD(real_vol_at_scale)
            errD_real = -output_real.mean() # WGAN loss for real

            # Fake data loss
            noise = generate_noise(real_vol_at_scale.shape[1:], device) # Use current scale size for noise
            # Get previous scale reconstruction (handle coarsest scale where prev is None)
            if scale_idx_actual == coarsest_scale_idx: # Coarsest scale
                # Input to G is just noise, prev_rec is effectively zero
                prev_rec_detached = torch.zeros_like(real_vol_at_scale, device=device)
            else:
                # Get reconstruction from the *previous actual scale* (coarser scale)
                prev_rec_detached = stored_reconstructions[scale_idx_actual + 1].detach()

            # Upsample previous reconstruction
            prev_rec_upsampled = upsample_3d(prev_rec_detached, scale_factor=scale_factor_r)
            # Ensure size matches current scale exactly
            prev_rec_upsampled = F.interpolate(prev_rec_upsampled, size=current_size, mode='trilinear', align_corners=False)

            fake_vol = netG(noise, prev_rec_upsampled).detach() # Detach fake_vol for D training
            output_fake = netD(fake_vol)
            errD_fake = output_fake.mean() # WGAN loss for fake

            # Gradient Penalty
            gradient_penalty = calculate_gradient_penalty(netD, real_vol_at_scale.data, fake_vol.data, device, opt.lambda_gp)

            # Total D loss
            errD = errD_real + errD_fake + gradient_penalty
            errD.backward()
            optimizerD.step()

            # Clip discriminator weights if not using GP (legacy WGAN) - Not recommended
            # for p in netD.parameters():
            #     p.data.clamp_(-opt.clip_value, opt.clip_value)

            # --- Generator Training (typically less frequent than D, e.g., every opt.Dsteps) ---
            if iter_num % opt.Gsteps == 0:
                netG.zero_grad()
                # We need to regenerate fake_vol for G update as graph was potentially freed
                # Reuse noise and prev_rec_upsampled from D step
                fake_vol_for_G = netG(noise, prev_rec_upsampled) # Don't detach here
                output_fake_G = netD(fake_vol_for_G)
                errG_adv = -output_fake_G.mean() # Generator wants to maximize D's output for fake

                # Reconstruction loss
                if opt.alpha > 0:
                    rec_noise_this_scale = fixed_noise_maps[scale_idx_actual] # Get appropriate noise
                    if scale_idx_actual == coarsest_scale_idx: # Coarsest scale
                        prev_rec_for_G = torch.zeros_like(real_vol_at_scale, device=device)
                    else:
                        prev_rec_for_G = stored_reconstructions[scale_idx_actual + 1].detach() # Use stored reconstruction

                    prev_rec_upsampled_for_G = upsample_3d(prev_rec_for_G, scale_factor=scale_factor_r)
                    prev_rec_upsampled_for_G = F.interpolate(prev_rec_upsampled_for_G, 
                                                             size=current_size, 
                                                             mode='trilinear', 
                                                             align_corners=False)

                    reconstruction = netG(rec_noise_this_scale, prev_rec_upsampled_for_G)
                    errG_rec = F.mse_loss(reconstruction, real_vol_at_scale) * opt.alpha
                else:
                    errG_rec = torch.tensor(0.0, device=device)

                # Total G loss
                errG = errG_adv + errG_rec
                errG.backward()
                optimizerG.step()

            # Log progress
            if iter_num % 1000 == 0:
                logging.info(f"Scale [{scale_idx_actual}] Iter [{iter_num}/{opt.niter}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} (Adv: {errG_adv.item():.4f} Rec: {errG_rec.item():.4f})")

            # Step schedulers
            schedulerG.step()
            schedulerD.step()

            # Optional: Save intermediate generated sample visualization
            if ((iter_num+1) % opt.save_every_epoch == 0) or (iter_num == 0):
                 with torch.no_grad():
                     vis_noise = generate_noise(real_vol_at_scale.shape[1:], device)
                     vis_fake = netG(vis_noise, prev_rec_upsampled).detach()
                     save_snapshot(vis_fake, os.path.join(opt.outdir, f"sample_scale{scale_idx_actual}_iter{iter_num}.png"))
                     save_volume(vis_fake, os.path.join(opt.outdir, f"sample_scale{scale_idx_actual}_iter{iter_num}.npy"))


        # --- End of scale training ---
        # Save trained generator state dict (append to list)
        trained_generators.append(netG.state_dict())
        # Save final reconstruction for this scale (needed for next scale's input)
        if opt.alpha > 0:
             with torch.no_grad():
                 rec_noise_this_scale = fixed_noise_maps[scale_idx_actual]
                 if scale_idx_actual == coarsest_scale_idx:
                     prev_rec_final = torch.zeros_like(real_vol_at_scale, device=device)
                 else:
                     prev_rec_final = stored_reconstructions[scale_idx_actual + 1].detach()
                 prev_rec_upsampled_final = upsample_3d(prev_rec_final, scale_factor=scale_factor_r)
                 prev_rec_upsampled_final = F.interpolate(prev_rec_upsampled_final, size=current_size, mode='trilinear', align_corners=False)
                 final_reconstruction = netG(rec_noise_this_scale, prev_rec_upsampled_final).detach()
                 stored_reconstructions[scale_idx_actual] = final_reconstruction
                 # Save final reconstruction image for inspection
                 save_volume(final_reconstruction, os.path.join(opt.outdir, f"reconstruction_scale{scale_idx_actual}.npy"))
                 save_snapshot(final_reconstruction, os.path.join(opt.outdir, f"reconstruction_scale{scale_idx_actual}.png"))

        else:
             # If no reconstruction loss, store a zero tensor or handle differently
             stored_reconstructions[scale_idx_actual] = torch.zeros_like(real_vol_at_scale, device=device)

        # Save model checkpoint for this scale
        torch.save({
            'scale': scale_idx_actual,
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
        }, os.path.join(opt.outdir, f"model_scale{scale_idx_actual}.pth"))


    logging.info("\n--- Training Complete ---")
    # Return trained generators (state dicts, ordered finest to coarsest)
    # and fixed noise maps (dict by scale index)
    return trained_generators[::-1], fixed_noise_maps, pyramid # Return Gs finest to coarsest
