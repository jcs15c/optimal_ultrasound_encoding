import numpy as np
import torch

import src.settings as s
import src.ultrasound_encoding as uu

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


def create_target_mask( locs, image_grid, rpad=1.0 ):
    """
    Create a mask for point targets or anechoic targets in a lesion image
    
    Parameters:
        locs - Locations for imaging targets, stored in [x, y, z, radius] values.
                If radius is nonpositive, target is a point target
        image_grid - [xpts, zpts] Coordinates of each pixel in image (mm)
        rpad - Padding for pixels in lesion targets
        
    Returns:
        Imaging mask
    """

    # get array of angles
    angles = torch.atan2( -locs[:,:,2], locs[:,:,0] )
    Z, X = torch.meshgrid( image_grid[1], image_grid[0], indexing='ij' )
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
                mask[k] = mask[k] + ( (X - x0)**2  + (Z - z0)**2 <= (rpad*r0)**2 )
            mask[k] = ~mask[k]
    return ~mask

def partial_histogram_matching( env, locs, image_grid, mu_Y, sigma_Y ):
    """
    Scale the image so that the mean and variance of it's spackle pattern matches
    that of a refernce (mu_Y, sigma_Y). Assumes that env is an array of 
    images (num_images x xpx x zpx)
    
    Parameters:
        env - Complex envelope for the image
        locs - Locations for imaging targets, stored in [x, y, z, radius] values.
                If radius is nonpositive, target is a point target
        image_grid - [xpts, zpts] Coordinates of each pixel in image (mm)
        mu_Y, sigma_y - Mean and standard deviation for background spackle pattern
        
    Returns:
        Scaled complex envelope
    """
    rescaled_env = torch.zeros( env.shape )
   
    target_mask = ~create_target_mask( locs, image_grid )
    
    # Only want to consider middle fifth of image to avoid FOV issues
    trunc_mask = torch.zeros( env.shape, dtype=torch.bool )
    trunc_mask[:, int(trunc_mask.shape[1]*0.4):int(trunc_mask.shape[1]*0.6), 
                  int(trunc_mask.shape[2]*0.4):int(trunc_mask.shape[2]*0.6)] = 1

    dim = trunc_mask * target_mask
     
    for i in range( env.shape[0] ):
        if locs[i,0,3] > 0: # Indicates that this image has lesions
            sigma_X = torch.std( env[i, dim[i]] )
            mu_X = torch.mean( env[i, dim[i]] )
        
            a = sigma_Y / sigma_X
            b = mu_Y - a*mu_X
    
            rescaled_env[i] = a*env[i] + b        
        else:  # Do nothing if point targets
            rescaled_env[i] = env[i]
    return rescaled_env


def mean_gCNR( env, locs, image_grid, hist_param=[100, 3], smooth=True, truncated=False, rpad=[1.0, 1.0] ):
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
        rpad - [inner radius, outer radius] Padding for pixels in gCNR calculation
        
    Returns:
        the generalized Contrast to Noise Ratio
    """
    bins, sigma = hist_param[0], hist_param[1]
    gCNRs = torch.zeros( env.shape[0] )
    
    outside = ~create_target_mask( locs, image_grid, rpad[1] )
    inside  =  create_target_mask( locs, image_grid, rpad[0] )

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
            
            hist_A = uu.SoftHistogram( bins=bins, min=m, max=M, sigma=sigma )( A ) / A.numel()         
            hist_B = uu.SoftHistogram( bins=bins, min=m, max=M, sigma=sigma )( B ) / B.numel()
            
            gCNRs[i] = 1 - torch.min( hist_A, hist_B ).sum()
    else:
          for i in range( env.shape[0] ):
            A = env[i,  inside[i] * trunc_mask[i]]
            B = env[i, outside[i] * trunc_mask[i]]
            
            m = env[i].min()
            M = env[i].max()
            
            hist_A = torch.histc( A, bins=bins, min=m.item(), max=M.item() ) / A.numel()           
            hist_B = torch.histc( B, bins=bins, min=m.item(), max=M.item() ) / B.numel()
            
            gCNRs[i] = 1 - torch.min( hist_A, hist_B ).sum()   
            
    return torch.mean( gCNRs )