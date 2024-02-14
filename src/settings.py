import numpy as np
import torch

# precision for floating point operations
PTFLOAT = torch.float32
PTCOMPLEX = torch.complex64

NPFLOAT = np.float32
NPCOMPLEX = np.complex64

# General parameters
c = 1540
f0 = 3e6
fs = 12e6

# Lateral dimension transducer parameters
num_elements_x=64                   # Number of elements
pitch_x=300e-6                      # Element pitch [m]
kerf_x=.01/1000                     # Kerf [m]
element_width=pitch_x-kerf_x        # Element width [m]
sub_x=2                             # Number of subelements in x and y
lat_focus=40e-3                     # Axial-lateral focus [m]

# Elevation dimension transducer parameters
num_elements_y=1
kerf_y=0
pitch_y=10/1000;                  
element_height=pitch_y-kerf_y       # Element height [m]
sub_y=sub_x*round(element_height/element_width)
elev_focus=40/1000                  # Axial-elevation focus [m]

nelem = 1
num_lines = num_elements_x-nelem+1

# Generate transducer positions
aperture_rx = np.linspace(-(num_elements_x-1)/2*pitch_x, (num_elements_x-1)/2*pitch_x, num_elements_x)
aperture_tx = np.linspace(-(num_lines-1)/2*pitch_x, (num_lines-1)/2*pitch_x, num_lines)

rx_pos = torch.tensor( np.hstack( (aperture_rx.reshape(-1, 1), np.zeros( [aperture_rx.shape[0], 2] )) ) )
tx_pos = torch.tensor( np.hstack( (aperture_tx.reshape(-1, 1), np.zeros( [aperture_tx.shape[0], 2] )) ) )

# Expected speckle pattern brightness
hist_data = [-11.811993598937988, 5.649099826812744]

default_opt_params = {
    'trainable_params': 'both',
    'training_resolution': 1.0,
    'training_shuffle': True,
    'loss_func': None,
    'DelayProj': None,
    'WeightProj': None,
    'target_type': 'unencoded',
    'num_epochs': 15,
    'desc_alg': "SGD",
    'lrate': 0.01,
    'sched': None,
    'momentum': 0.0
}

default_enc_params = {
    'noise_params': None,
    'tik_param': 0.1
}

default_bf_params = {
    'image_dims': [300, 500],
    'image_range': None,
    'dB_min': -60,
    'roi_pads': [1.0, 1.0],
    'hist_params': {'bins': 100, 'sigma': 3},
    'hist_match': True
}