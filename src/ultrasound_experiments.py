import numpy as np
import torch

import scipy
import scipy.io

import pandas as pd

import src.ultrasound_encoding as ue
import src.ultrasound_imaging as ui
import src.settings as s

class ExperimentalDataset( torch.utils.data.Dataset ):
    """
    Return experimentally acquired RF data and locations within a given folder
    """
    def __init__(self, data_directory, subset_idx=None):
        """
        Load filenames for each piece of data
        """
        self.data_dir = data_directory
        self.data_names = pd.read_csv( data_directory + '/data_filenames.csv' )
        self.acq_names = pd.read_csv( data_directory + '/acq_filenames.csv' )
        if subset_idx is not None:
            self.subset_idx = subset_idx
        else:
            self.subset_idx = [i for i in range(len(self.data_names))]

    def __len__(self):
        return len(self.subset_idx)
    
    def __getitem__(self, sidx):
        """
        Return RF data and locationd data
        """
        idx = self.subset_idx[sidx]
        macq = scipy.io.loadmat( self.data_dir + '/' + self.acq_names.iloc[idx, 0])['acq_params'][0,0]
        acq_params = {'c' : np.array( macq['c'][0][0] ), 
                      'fs' : np.array( macq['fs'][0][0] ),
                      'samples' : np.array( macq['samples'][0][0] ),
                      'rx_pos' : np.array( macq['rx_pos'] ),
                      'tx_pos' : np.array( macq['tx_pos'] ),
                      'locs' : np.array( macq['locs'] ),
                      'f0' : np.array( macq['f0'][0][0] ),
                      't0' : np.array( macq['t0'][0][0] ),
                      'name' : str(macq['name'][0]),
                      'delays' : np.array( macq['tx_delays'] ),
                      'weights' : np.array( macq['apod'] )}
        
        acq_params['r0'] = (acq_params['t0'] / acq_params['fs'] * acq_params['c'])
        acq_params['dr'] = (acq_params['c'] / acq_params['fs'])

        return scipy.io.loadmat( self.data_dir + '/' + self.data_names.iloc[idx, 0])['rf'].astype( s.NPFLOAT ), \
               acq_params
    

def cystic_resolution_threshold_experimental( data, acq_params, threshold, tik_param=0.1 ):
    loc = torch.tensor( acq_params['locs'], dtype=s.PTFLOAT )
    x0 = loc[0,0]
    z0 = loc[0,2]

    delays = torch.tensor( acq_params['delays'], dtype=s.PTFLOAT )
    weights = torch.tensor( acq_params['weights'], dtype=s.PTFLOAT )

    H = ue.calc_H( acq_params['samples'], delays, weights )
    Hinv = ue.calc_Hinv_tikhonov(H, param=tik_param)

    npts = 400
    x_grid = torch.linspace( x0 - 0.02, x0 + 0.02, npts )
    z_grid = torch.linspace( z0 - 0.02, z0 + 0.02, npts )
    z_grid = z_grid[ z_grid >= 0.005 ]

    image_grid = [x_grid, z_grid]
    Z, X = torch.meshgrid( image_grid[1], image_grid[0], indexing='ij' )

    rf_dec = ue.encode( data, Hinv )
    
    num_elements = acq_params['rx_pos'].shape[0]
    bf_delays = ue.calc_delays_beamforming( acq_params['rx_pos'], x_grid, z_grid ).to( torch.device( "cuda:0" ) )
    iq_focused = ui.BeamformAD.apply(ue.hilbert( rf_dec ), acq_params['r0'], acq_params['dr'], 
                                        bf_delays, num_elements, x_grid.shape[0], z_grid.shape[0])
    unclipped_env = torch.abs( iq_focused )
    
    # Pixels expected to be in the lesion
    def get_cr( radius ):
        lesion_mask = torch.zeros_like( unclipped_env, dtype=torch.bool ) + ( (X - x0)**2  + (Z - z0)**2 <= (radius / 1000)**2 )
        lesion_px = unclipped_env[ lesion_mask ]

        return 20 * np.log10( torch.sqrt( 1 - lesion_px.square().sum() / unclipped_env.square().sum() ).item() )
    
    # Bisection method to find the point at which get_cr( radius ) = threshold
    # First, find the upper and lower bounds
    lower_radius = 0
    upper_radius = 10
    while get_cr( upper_radius ) > threshold:
        upper_radius *= 2

    # Now, do the bisection
    while upper_radius - lower_radius > 0.001:
        radius = (upper_radius + lower_radius) / 2
        if get_cr( radius ) > threshold:
            lower_radius = radius
        else:
            upper_radius = radius

    return (upper_radius + lower_radius) / 2