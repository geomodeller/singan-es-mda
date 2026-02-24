import torch
import torch.nn.functional as F
import sys
sys.path.insert(0,'./sinGAN_script')
from utilities import generate_noise
from models import Generator3D
import numpy as np
import pickle
from scipy.ndimage import rotate, shift

with open(r'C:\Users\DELL\Desktop\sinGAN-esmda\params_sand_ratio.pkl', 'rb') as f:
    param_sand_ratio = pickle.load(f)
coarset_pyramid_padding = np.load(r'C:\Users\DELL\Desktop\sinGAN-esmda\sinGAN_trained_model\coarsest_pyramid_padding.npy')

def generate_3d_sample(trained_generators_state_dicts, 
                       pyramid, 
                       options, device, 
                       starting_scale=0,
                       last_noise_scale = 4, 
                       noise_shape_override=None,
                       evaluation_mode=False,
                       return_intermediate_processes:bool = False,
                       verbose = False):
    """
    Generates a 3D volume sample using a trained SinGAN model.

    This function utilizes a list of trained generator state dictionaries to
    create a 3D volume by progressively generating through a pyramid of scales.
    The generation can start from a custom scale and optionally use a custom
    noise shape.

    Args:
        trained_generators_state_dicts (list): List of state dictionaries for
            the trained generator models, ordered from finest to coarsest scale.
        pyramid (list): A list of tensors representing the pyramid levels,
            ordered from coarsest to finest scale.
        options: Options object containing generation parameters such as 'nc_im'
            and 'scale_factor'.
        device: The torch device to perform the generation on.
        starting_scale (int, optional): The starting scale index for
            generation, with 0 being the finest scale. Defaults to 0.
        noise_shape_override (tuple, optional): A tuple specifying a custom
            noise shape (D, H, W) to use at the starting scale. If None, uses
            the shape from the pyramid.
        evaluation_mode (bool, optional): If True, sets the generators to evaluation mode.

    Returns:
        torch.Tensor: A 5D tensor representing the generated 3D volume
        sample with shape (batch_size, C, D, H, W).
    """

    num_scales = len(trained_generators_state_dicts)
    generators = []

    for i in range(num_scales):
        generator = Generator3D(options).to(device)
        generator.load_state_dict(trained_generators_state_dicts[i])
        if evaluation_mode:
            generator.eval()
        else:
            generator.train()
        generators.append(generator)
    
    generators.reverse()
    actual_starting_scale = num_scales - 1 - starting_scale

    if noise_shape_override:
        noise_shape = (options.nc_im,) + tuple(noise_shape_override)
    else:
        noise_shape = pyramid[starting_scale].shape[1:]
    current_noise = generate_noise(noise_shape, device)
    current_volume = pyramid[starting_scale]
    current_volume_ = []
    current_volume_.append(current_volume)
    with torch.no_grad():
        for scale_index in range(actual_starting_scale, -1, -1):
            generator_index = num_scales - 1 - scale_index
            generator = generators[generator_index]
            current_volume = generator(current_noise, current_volume)
            if verbose:
                print(f"at Generator_index: {generator_index}")
                print(f"Current volume shape: {current_volume.shape}")
                print(f"Current noise shape: {current_noise.shape}")
                        
            if scale_index > 0:
                target_dimensions = pyramid[generator_index+1].shape[2:]
                current_volume = F.interpolate(
                    current_volume, size=target_dimensions, 
                    mode='trilinear', align_corners=False
                )

                if scale_index < last_noise_scale:
                    current_noise = generate_noise((1, *target_dimensions), device)*.1
                else:
                    current_noise = generate_noise((1, *target_dimensions), device)
            current_volume_.append(current_volume)
    if return_intermediate_processes:
        return current_volume_
    else:
        return current_volume


def generate_3d_sample_given_noise(
                       noise,
                       trained_generators_state_dicts, 
                       pyramid, 
                       options, 
                       device, 
                       rotation:float = 0.0,
                       translation_x:float = 0.0,
                       translation_y:float = 0.0,
                       translation_z:float = 0.0,
                       starting_scale:int = 0,
                       last_noise_scale:int = 4, 
                       evaluation_mode:bool = True,
                       verbose:bool = False):

    num_scales = len(trained_generators_state_dicts)
    generators = []

    for i in range(num_scales):
        generator = Generator3D(options).to(device)
        generator.load_state_dict(trained_generators_state_dicts[i])
        if evaluation_mode:
            generator.eval()
        else:
            generator.train()
        generators.append(generator)
    
    generators.reverse()
    actual_starting_scale = num_scales - 1 - starting_scale
    current_noise = torch.tensor(noise[0], device = device)

    if all([i==0 for i in [rotation, translation_x, translation_y,translation_z]]):
        current_volume = pyramid[starting_scale]                                                ## <- consider to update. Updated in 08/18/2025
    else:
        _,_,nz,ny,nx = pyramid[starting_scale].shape
        current_volume = coarset_pyramid_padding.copy()
        if rotation != 0:
            assert isinstance(rotation, float), "Rotation must be a float"
            current_volume = rotate(current_volume, angle=rotation, axes=(1, 2), reshape=False, order=1)
        if translation_x != 0 or translation_y != 0 or translation_z != 0:
            current_volume = shift(current_volume, shift=[translation_z, translation_y, translation_x], order=1, mode='reflect')
        current_volume = current_volume[nz:-nz,ny:-ny,nx:-nx].reshape(1,1,nz,ny,nx)
        current_volume = torch.tensor(current_volume, device=device)

    
    with torch.no_grad():
        for scale_index in range(actual_starting_scale, -1, -1):
            generator_index = num_scales - 1 - scale_index
            generator = generators[generator_index]
            current_volume = generator(current_noise, current_volume)
            
            if verbose:
                print(f"at Generator_index: {generator_index}")
                print(f"Current volume shape: {current_volume.shape}")
                print(f"Current noise shape: {current_noise.shape}")

            if scale_index > 0:
                target_dimensions = pyramid[generator_index+1].shape[2:]
                current_volume = F.interpolate(
                    current_volume, size=target_dimensions, 
                    mode='trilinear', align_corners=False
                )

                if scale_index < last_noise_scale:
                    current_noise = torch.tensor(noise[generator_index+1], device = device)*.1
                else:
                    current_noise = torch.tensor(noise[generator_index+1], device = device)
    return current_volume




def cdf_mapping(source, target):
    """
    Apply CDF mapping to make the distribution of 'source' match that of 'target'.
    
    Parameters:
        source (np.ndarray): The input array to be transformed.
        target (np.ndarray): The reference array whose distribution will be matched.

    Returns:
        np.ndarray: Transformed array with the same shape as 'source', but with
                    distribution matched to 'target'.
    """
    # Flatten both arrays
    source_flat = source.ravel()
    target_flat = target.ravel()

    # Sort and compute CDF values for source
    source_sorted = np.sort(source_flat)
    source_cdf = np.linspace(0, 1, len(source_sorted), endpoint=False)

    # Sort and compute quantiles for target
    target_sorted = np.sort(target_flat)

    # Create interpolator for mapping source values to target quantiles
    source_to_cdf = np.interp(source_flat, source_sorted, source_cdf)
    mapped_values = np.interp(source_to_cdf, np.linspace(0, 1, len(target_sorted), endpoint=False), target_sorted)

    # Reshape back to original shape
    return mapped_values.reshape(source.shape)


def cdf_mapping_by_layer(source, target):
    """
    Apply CDF mapping to make the distribution of 'source' match that of 'target'.
    
    Parameters:
        source (np.ndarray): The input array to be transformed.
        target (np.ndarray): The reference array whose distribution will be matched.

    Returns:
        np.ndarray: Transformed array with the same shape as 'source', but with
                    distribution matched to 'target'.
    """
    mapped_values = np.zeros(source.shape)
    for iz in range(source.shape[0]):
        mapped_values[iz] = cdf_mapping(source[iz], target[iz])
    
    return mapped_values


def cdf_mapping_by_layer_given_portion(source, target_portion:list):
    """
    Apply CDF mapping to make the distribution of 'source' match that of 'target'.
    
    Parameters:
        source (np.ndarray): The input array to be transformed.
        target (np.ndarray): The reference array whose distribution will be matched.

    Returns:
        np.ndarray: Transformed array with the same shape as 'source', but with
                    distribution matched to 'target'.
    """
    mapped_values = np.zeros(source.shape)
    target_portion = np.array(target_portion).reshape(-1,)
    for iz in range(source.shape[0]):
        portion =  target_portion[iz].clip(param_sand_ratio['lower_bound'][iz], 
                                           param_sand_ratio['upper_bound'][iz])
        thres = np.percentile(source[iz].flatten(),(1-portion)*100)
        mapped_values[iz] = (source[iz]>thres).astype('float')
    
    return mapped_values

def visual_3d(digital_rock_data: np.ndarray | np.ma.MaskedArray,
              spacing: tuple[float] = (1,1,1), # micro-meter in x-,y-,z-directions
              origin: tuple[float] = (0,0,0),
              slice_x_ratio: float = 0.5, 
              slice_y_ratio: float = 0.5,
              slice_z_ratio: float = 0.5,
              cmap:str = 'viridis',
              show_edges:bool = False,
              save_path:str = None):
    
    import pyvista as pv
    # Generate random data for demonstration; replace with actual voxel data
    # Data could represent intensities or any scalar field you have.
    data = digital_rock_data
    # Define the dimensions
    nz, ny, nx = digital_rock_data.shape  # Dimensions of the voxel grid

    # Define the spacing and origin (optional)
    spacing = spacing  # Define spacing between voxels
    origin = origin   # Define origin if needed

    # Create the uniform grid
    grid = pv.ImageData()

    # Set the grid dimensions
    grid.dimensions = np.array([nx, ny, nz])

    # Set the grid spacing and origin
    grid.spacing = spacing
    grid.origin = origin

    # Add the data to the grid as 'values'
    grid.point_data["values"] = data.flatten(order="C")  # Flatten to 1D in Fortran order

    # Start the plotter
    plotter = pv.Plotter()
    plotter.add_mesh(grid.slice_orthogonal(x=int(nx*slice_x_ratio), y=int(ny*slice_y_ratio), z=int(nz*slice_z_ratio)), 
                    show_edges = show_edges, 
                    cmap=cmap)# , opacity="sigmoid")

    if save_path is not None:
        plotter.show(screenshot=save_path)
        # Optionally close to free resources
        plotter.close()
    else:
        plotter.show()
    

def visual_3d_full_volume(digital_rock_data: np.ndarray | np.ma.MaskedArray,
                          spacing: tuple[float] = (1, 1, 1),  # micrometers in x-, y-, z-directions
                          origin: tuple[float] = (0, 0, 0),
                          cmap: str = 'viridis',
                          show_edges: bool = False,
                          save_path: str = None):
    """
    Visualize the full 3D digital rock volume using pyvista.

    Parameters:
    - digital_rock_data: 3D NumPy array or masked array containing scalar values.
    - spacing: voxel spacing in micrometers (x, y, z).
    - origin: origin of the grid.
    - cmap: colormap for rendering.
    - show_edges: whether to show mesh edges.
    - save_path: path to save a screenshot (optional).
    """
    
    import pyvista as pv
    data = digital_rock_data
    nz, ny, nx = data.shape  # PyVista expects dimensions in (nx, ny, nz) order

    # Create the grid
    grid = pv.ImageData()
    grid.dimensions = np.array([nx, ny, nz])
    grid.spacing = spacing
    grid.origin = origin
    grid.point_data["values"] = data.flatten(order="F")  # Fortran order to match VTK

    # Create a plotter and add the full volume mesh
    plotter = pv.Plotter()
    plotter.add_volume(grid, cmap=cmap, opacity="sigmoid")

    if save_path:
        plotter.show(screenshot=save_path)
        plotter.close()
    else:
        plotter.show()