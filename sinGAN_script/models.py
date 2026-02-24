## Update Note =================================================================
# 2025.7.21: Updated to v_0.0.4
#  - Review the code again
# =============================================================================
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Network Definitions
# =============================================================================

class ConvBlock3D(nn.Module):
    #... (implementation as in Section 4.3)...
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, batch_norm=True, activation=True):
        super(ConvBlock3D, self).__init__()
        layers = []
        layers.append(nn.Conv3d(in_channels,
                                out_channels,
                                kernel_size,
                                stride,
                                padding,
                                bias=bias))
        if batch_norm:
            # Use track_running_stats=True, affine=True by default
            layers.append(nn.BatchNorm3d(out_channels))
        if activation:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)

class Generator3D(nn.Module):
    #... (implementation as in Section 4.4, using opt attributes)...
    def __init__(self, opt):
        super(Generator3D, self).__init__()
        self.opt = opt
        N = opt.nfc # Base number of filters

        # Head: ConvBlock3D
        self.head = ConvBlock3D(int(opt.nc_im), int(N), opt.ker_size, 1, opt.padd_size)

        # Body: Sequence of ConvBlock3Ds
        body_blocks = []
        for _ in range(opt.num_layer):
            body_blocks.append(ConvBlock3D(N, N, opt.ker_size, 1, opt.padd_size))
        self.body = nn.Sequential(*body_blocks)

        # Tail: Final Conv3d layer
        self.tail = nn.Sequential(
            nn.Conv3d(N, opt.nc_im, opt.ker_size, 1, opt.padd_size),
            nn.Tanh() # Output normalized to [-1, 1]
        )

    def forward(self, noise, prev_out_upsampled):
        # Combine noise and previous output (simple addition here)
        # Ensure noise has same spatial dims as prev_out_upsampled
        if noise.shape[-3:]!= prev_out_upsampled.shape[-3:]:
             noise_resized = F.interpolate(noise, size=prev_out_upsampled.shape[-3:], mode='trilinear', align_corners=False)
        else:
             noise_resized = noise

        x = noise_resized + prev_out_upsampled

        x = self.head(x)
        x_body = self.body(x)
        residual = self.tail(x_body)

        # Apply residual connection - ensure sizes match for addition
        # We need to crop prev_out_upsampled to match residual's size if padding caused differences
        target_size = residual.shape[-3:]
        input_size = prev_out_upsampled.shape[-3:]

        # Calculate cropping indices (center crop)
        diff_d = input_size[0] - target_size[0]
        diff_h = input_size[1] - target_size[1]
        diff_w = input_size[2] - target_size[2]

        start_d = diff_d // 2
        start_h = diff_h // 2
        start_w = diff_w // 2

        # Handle cases where target is larger (shouldn't happen with padding=1, kernel=3, stride=1)
        if diff_d < 0 or diff_h < 0 or diff_w < 0:
            # Pad residual instead if it's smaller (less likely)
            # print("Padding residual instead of prev_out_upsampled")
            pad_d = max(0, -diff_d)
            pad_h = max(0, -diff_h)
            pad_w = max(0, -diff_w)
            residual = F.pad(residual, (pad_w//2, pad_w - pad_w//2,
                                        pad_h//2, pad_h - pad_h//2,
                                        pad_d//2, pad_d - pad_d//2))
            prev_cropped = prev_out_upsampled
        elif diff_d > 0 or diff_h > 0 or diff_w > 0:
            # Crop prev_out_upsampled
            print("Cropping prev_out_upsampled instead of residual")
            prev_cropped = prev_out_upsampled[:, :, start_d:start_d + target_size[0],
                                                    start_h:start_h + target_size[1],
                                                    start_w:start_w + target_size[2]]
        else:
            prev_cropped = prev_out_upsampled[:, :, start_d:start_d + target_size[0],
                                                    start_h:start_h + target_size[1],
                                                    start_w:start_w + target_size[2]]

        output = prev_cropped + residual
        return output


class Discriminator3D(nn.Module):

    #... (implementation as in Section 4.5, using opt attributes)...
    def __init__(self, opt):
        super(Discriminator3D, self).__init__()
        self.opt = opt
        N = opt.nfc # Base number of filters

        layers = []
        # Initial layer
        # No BN on first layer, bias might be okay
        layers.append(ConvBlock3D(opt.nc_im, N, opt.ker_size, 1, opt.padd_size, batch_norm=False, activation=True))

        # Downsampling layers
        current_filters = N
        # num_layer determines depth, adjust strides for downsampling
        # Example: 3 layers with stride 2 reduces size by 8x
        num_downsample = opt.num_layer # Or define separately
        for i in range(num_downsample):
            # Use stride=2 for downsampling
            # Increase filters, cap at 512 or opt.max_nfc
            next_filters = min(current_filters * 2, getattr(opt, 'max_nfc', 512))
            # Padding=1, kernel=3, stride=2 usually works well
            layers.append(ConvBlock3D(current_filters, next_filters, opt.ker_size, stride=1, padding=opt.padd_size, batch_norm=True, activation=True))
            current_filters = next_filters

        # Final layer for WGAN-GP (1 output channel, no activation/BN)
        # Kernel size might need adjustment based on final feature map size
        # Using kernel=3, padding=1 keeps size if stride=1
        layers.append(nn.Conv3d(current_filters, 1, kernel_size=opt.ker_size, stride=1, padding=opt.padd_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)