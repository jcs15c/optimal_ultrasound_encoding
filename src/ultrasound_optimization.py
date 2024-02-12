import numpy as np
import torch

import src.ultrasound_imaging as ui

def gCNR_loss( predictions, targets, locs, bf_params ):
    """
    Loss function given by average gCNR in an image
    """
    return -ui.mean_gCNR( predictions, locs, bf_params, hist_params=bf_params['hist_params'], roi_pads=bf_params['roi_pads'] )

def anechoic_loss( predictions, targets, locs, bf_params ):
    """
    Loss function given by average gCNR in an image
    """
    lesion_mask = ui.create_target_mask( locs, bf_params, rpad=bf_params['rpads'][0] )
    return torch.mean( torch.sum( torch.square( lesion_mask * (predictions - bf_params['dB_min'] ) ), axis=(1,2)) / torch.sum( lesion_mask, axis=(1, 2) ) )

def L2_cyst_normalized_loss( predictions, targets, locs, bf_params ):
    """
    Average L^2 Loss over anechoic and speckle regions
    """
    lesion_mask = ui.create_target_mask( locs, bf_params, rpad=bf_params['rpads'][0] )
    cyst_loss = torch.sum( torch.square( lesion_mask * (predictions - targets ) ), axis=(1,2))
    speckle_loss = torch.sum( torch.square( ~lesion_mask * (predictions - targets ) ), axis=(1,2))
    return torch.mean( ( torch.sum(  lesion_mask, axis=(1, 2) ) * cyst_loss + \
                        torch.sum( ~lesion_mask, axis=(1, 2) ) * speckle_loss ) / lesion_mask.numel() )

def L2_loss( predictions, targets, locs=None, bf_params=None ):
    """
    Standard L^2 Loss
    """
    return torch.mean( torch.sum( torch.square( predictions - targets ), axis=(1,2)) ) / predictions[0].numel()

def L2_point_normalization_loss( predictions, targets, locs, bf_params=None ):
    """
    Standard L^2 Loss
    """
    return torch.mean( torch.sum( torch.square( predictions - targets ), axis=(1,2)) ) / predictions[0].numel() / locs.shape[1]