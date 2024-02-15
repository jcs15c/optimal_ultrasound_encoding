import numpy as np
import torch

from scipy.ndimage import binary_dilation

import src.settings as s
import src.ultrasound_utilities as uu

class BeamformAD(torch.autograd.Function):
    """ Beamform operator with custom adjoint operation """

    @staticmethod
    def forward(ctx, iq, r0, dr, delays, num_elements, x_px, z_px):
        """
        Focus the multistatic data set to the chosen (x,z) grid points and sum
        across transmit/receive dimensions
    
        iq: Multistatic IQ data (time sample x receive channel x transmit element)
        r0: Initial sample depth
        dr: difference between sample depths
        delays: delays for beamforming at pixels (x_px * z_px) x (num_elements)
        num_elements: The number of transmit/recieve elements
        x_px, z_px: number of pixels in lateral and axial directiosn
                
        From pytorch docs:
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. 
        """
        
        gpu_device = torch.device( "cuda:0" )
        
        # Save input constants for backwards
        ctx.iq_shape = iq.shape
        ctx.delays = delays
        ctx.r0 = r0
        ctx.dr = dr
        
        iq = iq.permute(1, 2, 0)
        
        if torch.cuda.is_available():
            gpu_iq = iq.to( gpu_device )
            iq_focused = torch.zeros([2, x_px * z_px], dtype=s.PTFLOAT, device=gpu_device )
            the_input = torch.zeros( [1, 2, 1, iq.shape[2]], device=gpu_device )
            grid = torch.zeros( [1, 1, x_px * z_px, 1], device=gpu_device )
            
            # Perform a 1D interpolation for each transmit-recieve element pair
            for i in range( num_elements * num_elements ):
                i_rx = i // num_elements
                i_tx = i %  num_elements
                
                the_input = torch.view_as_real( gpu_iq[i_rx, i_tx, :] ).T.view(1, 2, 1, -1)
                
                grid = (delays[i_rx, :] + delays[i_tx, :]).view(1, 1, -1, 1)
                grid = ((grid - r0) / dr * 2 + 1 ) / iq.shape[2] - 1
                grid = torch.cat((grid, 0 * grid), axis=-1)
                
                iq_focused += torch.nn.functional.grid_sample( the_input, grid, align_corners=False ).view(2, -1)
                
            iq_focused = (iq_focused[0,:] + 1j*iq_focused[1,:]).reshape([z_px, x_px])
            return iq_focused.to( torch.device( "cpu" ) )
        else:   
            iq_focused = torch.zeros([2, x_px * z_px], dtype=s.PTFLOAT )
            the_input = torch.zeros( [1, 2, 1, iq.shape[2]] )
            grid = torch.zeros( [1, 1, x_px * z_px, 1] )
            
            # Perform a 1D interpolation for each transmit-recieve element pair
            for i in range( num_elements * num_elements ):
                i_rx = i // num_elements
                i_tx = i %  num_elements
        
                the_input = torch.view_as_real( iq[i_rx, i_tx, :] ).T.view(1, 2, 1, -1)
                
                grid = (delays[i_rx, :] + delays[i_tx, :]).view(1, 1, -1, 1)
                grid = ((grid - r0) / dr * 2 + 1 ) / iq.shape[2] - 1
                grid = torch.cat((grid, 0 * grid), axis=-1)
                
                iq_focused += torch.nn.functional.grid_sample( the_input, grid, align_corners=False ).view(2, -1)
                
            iq_focused = (iq_focused[0,:] + 1j*iq_focused[1,:]).reshape([z_px, x_px])
            return iq_focused
        
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        From pytorch docs:
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        samples, num_elements = ctx.iq_shape[:2]
        
        gpu_device = torch.device( "cuda:0" )
        if torch.cuda.is_available():
            grad_output = grad_output.flatten().to( gpu_device )
            grad_input = torch.zeros( [samples + 4, num_elements, num_elements], dtype=s.PTCOMPLEX, device=gpu_device )
            curs = torch.zeros( grad_output.shape, device=gpu_device )
            whole = torch.zeros( grad_output.shape, dtype=torch.long, device=gpu_device )
            
            # Perform 1D interpolation adjoint for each transmit-recieve element pair
            for i in range( num_elements * num_elements ):
                i_rx = i // num_elements
                i_tx = i %  num_elements
                
                curs = torch.clamp( (ctx.delays[i_rx, :] + ctx.delays[i_tx, :] - ctx.r0) / ctx.dr + 2, 0, samples + 2 )
                whole = curs.long()
                
                grad_input[:, i_rx, i_tx].index_add_( 0, whole  , (1.0 - (curs - whole)) * grad_output )
                grad_input[:, i_rx, i_tx].index_add_( 0, whole+1,        (curs - whole)  * grad_output )    
            
            return grad_input[2:-2, :, :].to( torch.device("cpu") ), None, None, None, None, None, None
        else:
            grad_output = grad_output.flatten()
            grad_input = torch.zeros( [samples + 4, num_elements, num_elements], dtype=s.PTCOMPLEX )
            
            # Perform 1D interpolation adjoint for each transmit-recieve element pair
            for i in range( num_elements * num_elements ):
                i_rx = i // num_elements
                i_tx = i %  num_elements
                
                curs = torch.clamp( (ctx.delays[i_rx, :] + ctx.delays[i_tx, :] - ctx.r0) / ctx.dr + 2, 0, samples + 2 )
                whole = curs.long()
                
                grad_input[:, i_rx, i_tx].index_add_( 0, whole  , (1.0 - (curs - whole)) * grad_output )
                grad_input[:, i_rx, i_tx].index_add_( 0, whole+1,        (curs - whole)  * grad_output )    
           
            return grad_input[2:-2, :, :], None, None, None, None, None, None
        

def meshless_torch_beamform( iq, r0, dr, rx_pos, x, z):
    """
    Focus the multistatic iq data set to the chosen (x,z) points and sum
    across transmit/receive dimensions

    Parameters:
        iq: Multistatic IQ data (time sample x receive channel x transmit element)
        r0: Initial sample depth
        dr: difference between sample depths
        num_elements: The number of transmit/recieve elements
        x, z: Pixels to beamform the image at

    Returns:
        Focused iq data
    """
    gpu_device = torch.device( "cuda:0" )
    
    # Get beamforming delays for this set of points
    delays = torch.zeros([len(x), rx_pos.shape[0]], dtype=s.PTFLOAT)
    for i, pos in enumerate(rx_pos):
        delays[:, i] = torch.sqrt(np.square(x - pos[0])   +
                                                pos[1]**2 +
                                  np.square(z - pos[2]))
    
    num_elements = rx_pos.shape[0]
    iq = iq.permute(1, 2, 0)
        
    if torch.cuda.is_available():
        delays = delays.T.to( gpu_device )
        gpu_iq = iq.to( gpu_device )
        iq_focused = torch.zeros([2, len(x)], dtype=s.PTFLOAT, device=gpu_device )
        the_input = torch.zeros( [1, 2, 1, iq.shape[2]], device=gpu_device )
        grid = torch.zeros( [1, 1, len(x), 1], device=gpu_device )
        
        # Perform a 1D interpolation for each transmit-recieve element pair
        for i in range( rx_pos.shape[0] * rx_pos.shape[0] ):
            i_rx = i // rx_pos.shape[0]
            i_tx = i %  rx_pos.shape[0]
            
            the_input = torch.view_as_real( gpu_iq[i_rx, i_tx, :] ).T.view(1, 2, 1, -1)
            
            grid = (delays[i_rx, :] + delays[i_tx, :]).view(1, 1, -1, 1)
            grid = ((grid - r0) / dr * 2 + 1 ) / iq.shape[2] - 1
            grid = torch.cat((grid, 0 * grid), axis=-1)
            
            iq_focused += torch.nn.functional.grid_sample( the_input, grid, align_corners=False ).view(2, -1)
            
        iq_focused = (iq_focused[0,:] + 1j*iq_focused[1,:]).reshape(x.shape)
        return iq_focused.to( torch.device( "cpu" ) )
    else:   
        iq_focused = torch.zeros([2, len(x)], dtype=s.PTFLOAT )
        the_input = torch.zeros( [1, 2, 1, iq.shape[2]] )
        grid = torch.zeros( [1, 1, len(x), 1] )
        
        # Perform a 1D interpolation for each transmit-recieve element pair
        for i in range( num_elements * num_elements ):
            i_rx = i // num_elements
            i_tx = i %  num_elements
    
            the_input = torch.view_as_real( iq[i_rx, i_tx, :] ).T.view(1, 2, 1, -1)
            
            grid = (delays[i_rx, :] + delays[i_tx, :]).view(1, 1, -1, 1)
            grid = ((grid - r0) / dr * 2 + 1 ) / iq.shape[2] - 1
            grid = torch.cat((grid, 0 * grid), axis=-1)
            
            iq_focused += torch.nn.functional.grid_sample( the_input, grid, align_corners=False ).view(2, -1)
            
        iq_focused = (iq_focused[0,:] + 1j*iq_focused[1,:]).reshape(x.shape)
        return iq_focused


def create_target_mask( locs, bf_params, roi_pads=1.0 ):
    """
    Create a mask for point targets or anechoic targets in a lesion image
    
    Parameters:
        locs - Locations for imaging targets, stored in [x, y, z, radius] values.
                If radius is nonpositive, target is a point target
        image_grid - [xpts, zpts] Coordinates of each pixel in image (mm)
        roi_pads - Padding for pixels in lesion targets
        
    Returns:
        Imaging mask
    """

    # get array of angles
    angles = torch.atan2( -locs[:,:,2], locs[:,:,0] )
    xpts = torch.linspace( bf_params['image_range'][0], bf_params['image_range'][1], bf_params['image_dims'][0] ) / 1000
    zpts = torch.linspace( bf_params['image_range'][2], bf_params['image_range'][3], bf_params['image_dims'][1] ) / 1000

    Z, X = torch.meshgrid( zpts, xpts, indexing='ij' )
    mask = torch.zeros( [locs.shape[0], Z.shape[0], Z.shape[1]], dtype=torch.bool )
        
    # Ellipse parameters
    a = 0.0006
    b = 0.0012
    
    for k in range(locs.shape[0]):
        if locs[k,0,3] <= 0: # Indicates that this image has point targets
            for i in range(locs.shape[1]):
                x0 = locs[k,i,0]
                z0 = locs[k,i,2]
                t0 = angles[k, i]
                mask[k] = mask[k] + ( ((X - x0)*np.cos(t0) - (Z - z0)*np.sin(t0))**2/a**2 \
                                  +   ((X - x0)*np.sin(t0) + (Z - z0)*np.cos(t0))**2/b**2 <= 1 )
        else: #Indicates that this image has lesions
            for i in range(locs.shape[1]):
                x0 = locs[k,i,0]
                z0 = locs[k,i,2]
                r0 = locs[k,i,3]
                mask[k] = mask[k] + ( (X - x0)**2  + (Z - z0)**2 <= (roi_pads*r0)**2 )
            mask[k] = ~mask[k]
    return ~mask

def partial_histogram_matching( envs, locs, bf_params, mu_Y, sigma_Y, trunc=[0.4, 0.6] ):
    """
    Scale the image so that the mean and variance of it's spackle pattern matches
    that of a refernce (mu_Y, sigma_Y). Assumes that envs is an array of 
    images (num_images x xpx x zpx)
    
    Parameters:
        envs - Complex envelope for the image
        locs - Locations for imaging targets, stored in [x, y, z, radius] values.
                If radius is nonpositive, target is a point target
        image_grid - [xpts, zpts] Coordinates of each pixel in image (mm)
        mu_Y, sigma_y - Mean and standard deviation for background spackle pattern
        
    Returns:
        Scaled complex envelope
    """
    rescaled_envs = torch.zeros( envs.shape )
   
    if locs.shape[2] == 4:
        target_mask = ~create_target_mask( locs, bf_params )

        # Only want to consider middle fifth of image to avoid FOV issues
        trunc_mask = torch.zeros( envs.shape, dtype=torch.bool )
        trunc_mask[:, int(trunc_mask.shape[1]*0.4):int(trunc_mask.shape[1]*0.6), 
                      int(trunc_mask.shape[2]*trunc[0]):int(trunc_mask.shape[2]*trunc[1])] = 1

        dim = trunc_mask * target_mask
    else:
        resized_cmaps = resize_images( locs, envs.shape[2], envs.shape[1] ) != 0
        selem = np.ones( [9, 9] )
        dim =  [~binary_dilation( ~cmap, structure=selem ) for cmap in resized_cmaps]
    

    for i in range( envs.shape[0] ):
        if locs[i,0,3] > 0 or len(locs.shape) == 3: # Indicates that this image has lesions
            sigma_X = torch.std( envs[i, dim[i]] )
            mu_X = torch.mean( envs[i, dim[i]] )
        
            a = sigma_Y / sigma_X
            b = mu_Y - a*mu_X
    
            rescaled_envs[i] = a*envs[i] + b        
        else:  # Do nothing if point targets
            rescaled_envs[i] = envs[i]
    return rescaled_envs

def gCNR( env, loc, bf_params, filter_nan=True ):
    """
    Compute the gCNR for each lesion in an image.
    Need to get the mask, figure out which pixels are in/outside of the lesions,
    then compute histogram overlap.
    gCNR is 1 - overlap, and less overlap is better. Want to maximize this value
    
    Parameters:
        env - Complex envelope/B-mode image
        loc - Locations for imaging targets, stored in [x, y, z, radius] values.
                If radius is nonpositive, target is a point target
        loc_idx - Index of the lesion to compute gCNR for
        bf_params - parameters for beamforming (resolution, range, roi radii)
        
    Returns:
        the generalized Contrast to Noise Ratio
    """
    gCNRs = torch.zeros( loc.shape[0] )
    
    xpts = torch.linspace( bf_params['image_range'][0], bf_params['image_range'][1], bf_params['image_dims'][0] ) / 1000
    zpts = torch.linspace( bf_params['image_range'][2], bf_params['image_range'][3], bf_params['image_dims'][1] ) / 1000
    Z, X = torch.meshgrid( zpts, xpts, indexing='ij' )
    
    [m, M] = [env.min().item(), env.max().item()]


    for i in range(loc.shape[0]):
        x0 = loc[i,0]
        z0 = loc[i,2]
        r0 = loc[i,3]

        the_lesion = torch.zeros_like( env, dtype=torch.bool ) + ( (X - x0)**2  + (Z - z0)**2 <= (bf_params['roi_pads'][0]*r0)**2 )
        speckle_ring = torch.zeros_like( env, dtype=torch.bool ) + ( (X - x0)**2  + (Z - z0)**2 > (bf_params['roi_pads'][1]*r0)**2 )
        speckle_ring = speckle_ring * ( (X - x0)**2  + (Z - z0)**2 < (2 * bf_params['roi_pads'][1] * r0)**2 )
        all_speckle = ~torch.zeros_like(env, dtype=torch.bool)
        for j in range(loc.shape[0]):
            x1 = loc[j,0]
            z1 = loc[j,2]
            r1 = loc[j,3]
            all_speckle = all_speckle * ( (X - x1)**2  + (Z - z1)**2 > (bf_params['roi_pads'][1]*r1)**2 )

        # A gets all the pixels that are in the lesion
        A = env[the_lesion]

        # B gets all the pixels that are in the speckle ring and all_speckle
        B = env[speckle_ring * all_speckle]

        if A.numel() == 0 or B.numel() == 0:
            gCNRs[i] = torch.nan
        else:
            gCNRs[i] = 1 - torch.min( torch.histc( A, bins=bf_params['hist_params']['bins'], min=m, max=M ) / A.numel(), \
                                    torch.histc( B, bins=bf_params['hist_params']['bins'], min=m, max=M ) / B.numel() ).sum()
        
    if filter_nan:
        gCNRs = gCNRs[~torch.isnan(gCNRs)]
    
    return gCNRs

def average_gCNR( env, locs, bf_params, hist_params={'bins': 100, 'sigma':3}, smooth=True, truncated=False, roi_pads=[1.0, 1.0] ):
    """
    Compute the gCNR for an array of images need to get the mask, figure out which 
    pixels are in/outside of the lesions, then compute histogram overlap.
    gCNR is 1 - overlap, and less overlap is better. Want to maximize this value
    
    Parameters:
        env - Array of complex envelopes for image predictions
        locs - Locations for imaging targets, stored in [x, y, z, radius] values.
                If radius is nonpositive, target is a point target
        image_grid - [xpts, zpts] Coordinates of each pixel in image (mm)
        hist_param - [bins, smoothing parameter] for smooth histogram calculations
        smooth - If true, use smooth histogram for calculations
        truncated - If true, only compute gCNR of middle 5th of image
        roi_pads - [inner radius, outer radius] Padding for pixels in gCNR calculation
        
    Returns:
        the generalized Contrast to Noise Ratio
    """
    gCNRs = torch.zeros( env.shape[0] )
    
    outside = ~create_target_mask( locs, bf_params, roi_pads[1] )
    inside  =  create_target_mask( locs, bf_params, roi_pads[0] )

    if truncated == True:
        trunc_mask = torch.zeros( env.shape, dtype=torch.bool ) # Set middle fifth of each mask to 1
        trunc_mask[:, :, int(trunc_mask.shape[2]*0.3):int(trunc_mask.shape[2]*0.7)] = 1
    else: 
        trunc_mask = torch.ones( env.shape, dtype=torch.bool )
        
    if smooth == True: # Need this to be true if we want to optimize on this value
        for i in range( env.shape[0] ):
            
            A = env[i,  inside[i] * trunc_mask[i]]
            B = env[i, outside[i] * trunc_mask[i]]
            
            m = env[i].min()
            M = env[i].max()
            
            hist_A = uu.SoftHistogram( bins=hist_params['bins'], min=m, max=M, sigma=hist_params['sigma'] )( A ) / A.numel()         
            hist_B = uu.SoftHistogram( bins=hist_params['bins'], min=m, max=M, sigma=hist_params['sigma'] )( B ) / B.numel()
            
            gCNRs[i] = 1 - torch.min( hist_A, hist_B ).sum()
    else:
          for i in range( env.shape[0] ):
            A = env[i,  inside[i] * trunc_mask[i]]
            B = env[i, outside[i] * trunc_mask[i]]
            
            m = env[i].min()
            M = env[i].max()
            
            hist_A = torch.histc( A, bins=hist_params['bins'], min=m.item(), max=M.item() ) / A.numel()           
            hist_B = torch.histc( B, bins=hist_params['bins'], min=m.item(), max=M.item() ) / B.numel()
            
            gCNRs[i] = 1 - torch.min( hist_A, hist_B ).sum()   
            
    return torch.mean( gCNRs )

def binned_average_gCNR( env, locs, bf_params, gCNR_bin_count, hist_bin_count=100, rpads=[1.0, 1.0] ):
    """
    Compute the gCNR in a 3x3 grid along the imaging domain.
    Counts the running total of gCNRs and the number of lesions in each bin
      so that the value can be added across multiple images
    """
    gCNR_sums = torch.zeros( (gCNR_bin_count, gCNR_bin_count) )
    lesion_counts = torch.zeros( (gCNR_bin_count, gCNR_bin_count) )

    xpts = torch.linspace( bf_params['image_range'][0], bf_params['image_range'][1], bf_params['image_dims'][0] ) / 1000
    zpts = torch.linspace( bf_params['image_range'][2], bf_params['image_range'][3], bf_params['image_dims'][1] ) / 1000

    Z, X = torch.meshgrid( zpts, xpts, indexing='ij' )
    
    [m, M] = [env.min().item(), env.max().item()]
    
    for i in range( locs.shape[0] ):
        x0 = locs[i,0]
        z0 = locs[i,2]
        r0 = locs[i,3]

        outer_radius = 2 * r0

        the_lesion = torch.zeros_like( env, dtype=torch.bool ) + ( (X - x0)**2  + (Z - z0)**2 <= (rpads[0]*r0)**2 )
        speckle_ring = torch.zeros_like( env, dtype=torch.bool ) + ( (X - x0)**2  + (Z - z0)**2 > (rpads[1]*r0)**2 )
        speckle_ring = speckle_ring * ( (X - x0)**2  + (Z - z0)**2 < (outer_radius)**2 )
        all_speckle = ~torch.zeros_like(env, dtype=torch.bool)
        for j in range(locs.shape[0]):
            x1 = locs[j,0]
            z1 = locs[j,2]
            r1 = locs[j,3]
            all_speckle = all_speckle * ( (X - x1)**2  + (Z - z1)**2 > (rpads[1]*r1)**2 )

        # A gets all the pixels that are in the lesion
        A = env[the_lesion]

        # B gets all the pixels that are in the speckle ring and all_speckle
        B = env[speckle_ring * all_speckle]

        if A.numel() == 0 or B.numel() == 0:
            continue
        
        gCNRs = 1 - torch.min( torch.histc( A, bins=hist_bin_count, min=m, max=M ) / A.numel(), \
                               torch.histc( B, bins=hist_bin_count, min=m, max=M ) / B.numel() ).sum()
        
        x_index = int((x0 * 1000 - bf_params['image_range'][0] ) / (bf_params['image_range'][1] - bf_params['image_range'][0]) * gCNR_bin_count)
        z_index = int((z0 * 1000 - bf_params['image_range'][2] ) / (bf_params['image_range'][3] - bf_params['image_range'][2]) * gCNR_bin_count)

        x_index = max(0, min(x_index, gCNR_bin_count - 1))
        z_index = max(0, min(z_index, gCNR_bin_count - 1))

        gCNR_sums[x_index, z_index] += gCNRs
        lesion_counts[x_index, z_index] += 1

    return gCNR_sums, lesion_counts

def binary_image_gCNR( env, cmap, bf_params, hist_param=[100, 3], smooth=True ):
    """
    Compute the gCNR for an array of images. Need to get the mask, figure out which 
    pixels are in/outside of the lesions, then compute histogram overlap.
    gCNR is 1 - overlap, and less overlap is better. Want to maximize this value
    
    Parameters:
        env - Array of complex envelopes for image predictions
        locs - Locations for imaging targets, stored in [x, y, z, radius] values.
                If radius is nonpositive, target is a point target
        image_grid - [xpts, zpts] Coordinates of each pixel in image (mm)
        hist_param - [bins, smoothing parameter] for smooth histogram calculations
        smooth - If true, use smooth histogram for calculations
        truncated - If true, only compute gCNR of middle 5th of image
        rpad - [inner radius, outer radius] Padding for pixels in gCNR calculation
        
    Returns:
        the average generalized Contrast to Noise Ratio over all lesions in an image
    """
    bins, sigma = hist_param[0], hist_param[1]
    
    resized_cmap = resize_images( cmap.unsqueeze(0), bf_params['image_dims'][0], bf_params['image_dims'][1] )[0] != 0

    selem = np.ones( [9, 9] )

    inside = ~binary_dilation( ~resized_cmap, structure=selem )
    outside = ~binary_dilation( resized_cmap )

    if smooth == True: # Need this to be true if we want to optimize on this value
        A = env[inside]
        B = env[outside]
        
        m = env.min()
        M = env.max()
        
        hist_A = uu.SoftHistogram( bins=bins, min=m, max=M, sigma=sigma )( A ) / A.numel()         
        hist_B = uu.SoftHistogram( bins=bins, min=m, max=M, sigma=sigma )( B ) / B.numel()
        
        return 1 - torch.min( hist_A, hist_B ).sum()
    else:
        A = env[inside]
        B = env[outside]
        
        m = env.min()
        M = env.max()
        
        hist_A = torch.histc( A, bins=bins, min=m.item(), max=M.item() ) / A.numel()           
        hist_B = torch.histc( B, bins=bins, min=m.item(), max=M.item() ) / B.numel()
        
        return 1 - torch.min( hist_A, hist_B ).sum()  
    
def resize_images( cmaps, xpx, zpx ):
    return torch.nn.functional.interpolate( cmaps.view(cmaps.shape[0], 1, cmaps.shape[1], cmaps.shape[2]), (zpx, xpx)).view(cmaps.shape[0], zpx, xpx)

def gaussian_blur(input_tensor, kernel_size=3, sigma=1.0):
    kernel_size = int( kernel_size // 2 )

    # Create 2D Gaussian kernel
    kernel = torch.tensor([np.exp(-(x - kernel_size)**2/float(2*sigma**2)) for x in range(2 * kernel_size)], dtype=s.PTFLOAT)
    kernel = kernel / kernel.sum()

    # Reshape the kernel to have dimensions [1, 1, kH, kW]
    kernel = kernel.view(1, 1, -1, 1)

    # Pad the input tensor to handle borders
    padding = (kernel_size, kernel_size, kernel_size, kernel_size)

    input_tensor = torch.nn.functional.pad(input_tensor, padding, mode='reflect')
    
    # Apply convolution with the Gaussian kernel
    blurred_tensor = torch.nn.functional.conv2d(input_tensor, kernel, stride=1, groups=input_tensor.shape[1])
    cropped_tensor = blurred_tensor[:, :, kernel_size:-kernel_size, kernel_size:-kernel_size]
    
    return cropped_tensor

