import project_root
import torch
import src.ultrasound_encoding as ue
import src.ultrasound_imaging as ui
import src.ultrasound_utilities as uu
import src.ultrasound_experiments as ux
import src.sequence_constraints as sc
import src.settings as s
from src.predictor_model import PredictorModel

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import os
import scipy
import scipy.io

import numpy as np

dataset = ux.ExperimentalDataset( f"./data/phantom_data" )

## Set up model parameters
enc_params = s.default_enc_params
bf_params = s.default_bf_params
bf_params['roi_pads'] = [0.65, 1.35]
bf_params['image_dims'] = [400, 800]
bf_params['image_range'] = [-30, 8, 12, 65]
flipped_image_range = [-30, 8, 65, 12]

def find_image( name ):
    # Find the data from the dataset and beamform it
    i = 0
    for i in range(len(dataset)):
        if name in dataset[i][1]['name']:
            break
    else:
        raise ValueError( f"Could not find {name} in the dataset" )
    
    data, acq_params = dataset[i]
    data = torch.tensor( data, dtype=s.PTFLOAT )
    loc = torch.tensor( acq_params['locs'], dtype=s.PTFLOAT )
    delays = torch.tensor( acq_params['delays'], dtype=s.PTFLOAT )
    weights = torch.tensor( acq_params['weights'], dtype=s.PTFLOAT )

    H = ue.calc_H( acq_params['samples'], delays, weights )
    Hinv = ue.calc_Hinv_tikhonov(H, param=enc_params['tik_param'] )

    # Decode the data
    rf_dec = ue.encode( data, Hinv )

    # Focus the data
    xpts = torch.linspace( bf_params['image_range'][0], bf_params['image_range'][1], bf_params['image_dims'][0] ) / 1000
    zpts = torch.linspace( bf_params['image_range'][2], bf_params['image_range'][3], bf_params['image_dims'][1] ) / 1000

    bf_delays = ue.calc_delays_beamforming( acq_params['rx_pos'], xpts, zpts ).to( torch.device( "cuda:0" ) )
    iq_focused = ui.BeamformAD.apply(ue.hilbert( rf_dec ), acq_params['r0'], acq_params['dr'], 
                                     bf_delays, 64, bf_params['image_dims'][0], bf_params['image_dims'][1])

    env = torch.abs( iq_focused )
    image = torch.clamp( 20*torch.log10( env / torch.amax(env) ), bf_params['dB_min'], 0 )

    return image, acq_params

opt_gCNRs_avg = 0.0
narrow_gCNRs_avg = 0.0
middle_gCNRs_avg = 0.0
wide_gCNRs_avg = 0.0 

fig = plt.figure(figsize=(15, 7))
width_ratios = [1, 0.01, 1, 0.01, 1, 0.01, 1, 0.01, 0.05]
height_ratios = [1]
gs = GridSpec(len(height_ratios), len(width_ratios), figure=fig, width_ratios=width_ratios, height_ratios=height_ratios )#, wspace=0, hspace=0 )

positions = [1]
for pos_idx in positions:
    # Optimized
    opt_image, opt_acq_params = find_image( f"position_{pos_idx}_noiseless_Trained" ) 
    opt_loc = torch.tensor( opt_acq_params['locs'], dtype=s.PTFLOAT )
    opt_image = ui.partial_histogram_matching(opt_image.unsqueeze(0), opt_loc.unsqueeze(0), bf_params, s.hist_data[0], s.hist_data[1], trunc=[0.1, 1.0])[0]

    # Narrow 
    narrow_image, narrow_acq_params = find_image( f"position_{pos_idx}_SP_15_" ) 
    narrow_loc = torch.tensor( narrow_acq_params['locs'], dtype=s.PTFLOAT )
    narrow_image = ui.partial_histogram_matching(narrow_image.unsqueeze(0), narrow_loc.unsqueeze(0), bf_params, s.hist_data[0], s.hist_data[1], trunc=[0.5, 0.8])[0]

    # Middle
    middle_image, middle_acq_params = find_image( f"position_{pos_idx}_SP_60_" ) 
    middle_loc = torch.tensor( middle_acq_params['locs'], dtype=s.PTFLOAT )
    middle_image = ui.partial_histogram_matching(middle_image.unsqueeze(0), middle_loc.unsqueeze(0), bf_params, s.hist_data[0], s.hist_data[1], trunc=[0.5, 1.0])[0]

    # Wide
    wide_image, wide_acq_params = find_image( f"position_{pos_idx}_SP_150_" ) 
    wide_loc = torch.tensor( wide_acq_params['locs'], dtype=s.PTFLOAT )
    wide_image = ui.partial_histogram_matching(wide_image.unsqueeze(0), wide_loc.unsqueeze(0), bf_params, s.hist_data[0], s.hist_data[1], trunc=[0.0, 1.0])[0]

    # Add the gCNRs to the image
    opt_gCNRs = ui.gCNR( opt_image, opt_loc, bf_params, filter_nan=False )
    narrow_gCNRs = ui.gCNR( narrow_image, narrow_loc, bf_params, filter_nan=False )
    middle_gCNRs = ui.gCNR( middle_image, middle_loc, bf_params, filter_nan=False )
    wide_gCNRs = ui.gCNR( wide_image, wide_loc, bf_params, filter_nan=False )

    opt_gCNRs_avg += opt_gCNRs
    narrow_gCNRs_avg += narrow_gCNRs
    middle_gCNRs_avg += middle_gCNRs
    wide_gCNRs_avg += wide_gCNRs 

opt_gCNRs_avg /= len(positions)
narrow_gCNRs_avg /= len(positions)
middle_gCNRs_avg /= len(positions)
wide_gCNRs_avg /= len(positions)

pos_idx = 2
fig = plt.figure(figsize=(15, 5.5))
height_ratios = [1]
width_ratios = [1, 0.1, 1, 0.1, 1, 0.1, 1, 0.1, 0.05, 0.3, 0.05]
gs = GridSpec(len(height_ratios), len(width_ratios), figure=fig, width_ratios=width_ratios, height_ratios=height_ratios, wspace=0, hspace=0 )

# Narrow 
narrow_image, narrow_acq_params = find_image( f"position_{pos_idx}_SP_15_" ) 
narrow_loc = torch.tensor( narrow_acq_params['locs'], dtype=s.PTFLOAT )
narrow_image = ui.partial_histogram_matching(narrow_image.unsqueeze(0), narrow_loc.unsqueeze(0), bf_params, s.hist_data[0], s.hist_data[1], trunc=[0.5, 0.8])[0]

ax1 = fig.add_subplot(gs[0])
im = ax1.imshow( narrow_image, cmap='gray', extent=flipped_image_range, vmin=bf_params['dB_min'], vmax=0 )
ax1.set_ylabel( "Axial (mm)")
ax1.set_xlabel( "Lateral (mm)")
ax1.set_xticks([-30, -20, -10, 0, 8])
ax1.set_xticklabels([-30, -20, -10, 0, 8])

# Middle
middle_image, middle_acq_params = find_image( f"position_{pos_idx}_SP_60_" ) 
middle_loc = torch.tensor( middle_acq_params['locs'], dtype=s.PTFLOAT )
middle_image = ui.partial_histogram_matching(middle_image.unsqueeze(0), middle_loc.unsqueeze(0), bf_params, s.hist_data[0], s.hist_data[1], trunc=[0.5, 1.0])[0]

ax2 = fig.add_subplot(gs[2])
ax2.imshow( middle_image, cmap='gray', extent=flipped_image_range, vmin=bf_params['dB_min'], vmax=0 )
ax2.set_yticklabels([])
ax2.set_xlabel( "Lateral (mm)")
ax2.set_xticks([-30, -20, -10, 0, 8])
ax2.set_xticklabels([-30, -20, -10, 0, 8])

# Wide
wide_image, wide_acq_params = find_image( f"position_{pos_idx}_SP_150_" ) 
wide_loc = torch.tensor( wide_acq_params['locs'], dtype=s.PTFLOAT )
wide_image = ui.partial_histogram_matching(wide_image.unsqueeze(0), wide_loc.unsqueeze(0), bf_params, s.hist_data[0], s.hist_data[1], trunc=[0.0, 1.0])[0]

ax3 = fig.add_subplot(gs[4])
ax3.imshow( wide_image, cmap='gray', extent=flipped_image_range, vmin=bf_params['dB_min'], vmax=0 )
ax3.set_yticklabels([])
ax3.set_xlabel( "Lateral (mm)")
ax3.set_xticks([-30, -20, -10, 0, 8])
ax3.set_xticklabels([-30, -20, -10, 0, 8])

# Optimized
opt_image, opt_acq_params = find_image( f"position_{pos_idx}_noiseless_Trained" ) 
opt_loc = torch.tensor( opt_acq_params['locs'], dtype=s.PTFLOAT )
opt_image = ui.partial_histogram_matching(opt_image.unsqueeze(0), opt_loc.unsqueeze(0), bf_params, s.hist_data[0], s.hist_data[1], trunc=[0.1, 1.0])[0]

ax4 = fig.add_subplot(gs[6])
im = ax4.imshow( opt_image, cmap='gray', extent=flipped_image_range, vmin=bf_params['dB_min'], vmax=0 )
ax4.set_yticklabels([])
ax4.set_xlabel( "Lateral (mm)")
ax4.set_xticks([-30, -20, -10, 0, 8])
ax4.set_xticklabels([-30, -20, -10, 0, 8])


plt.rcParams['text.usetex'] = True
ax1.text(0.5, 1.05, r"\underline{$15^\circ$ Span Planewaves}", transform=ax1.transAxes, ha='center', va='center', fontsize=14)
ax2.text(0.5, 1.05, r"\underline{$60^\circ$ Span Planewaves}", transform=ax2.transAxes, ha='center', va='center', fontsize=14)
ax3.text(0.5, 1.05, r"\underline{$150^\circ$ Span Planewaves}", transform=ax3.transAxes, ha='center', va='center', fontsize=14)
ax4.text(0.5, 1.05, r"\underline{Optimized}", transform=ax4.transAxes, ha='center', va='center', fontsize=14)
plt.rcParams['text.usetex'] = False

snap_vals_x = [-24, -9, 2]
snap_vals_z = [23, 33, 43, 53, 63]
def snap_value( val, snap_vals ):
    return snap_vals[ np.argmin( np.abs( np.array(snap_vals) - np.array(val) ) ) ]

# calc max and min among all gCNRs
max_gCNR = 1.0
min_gCNR = 0.8

# Define the plasma colormap
plasma = matplotlib.cm.get_cmap('plasma')
plasma_norm = matplotlib.colors.Normalize(vmin=min_gCNR, vmax=max_gCNR)
plasma_sm = matplotlib.cm.ScalarMappable(cmap=plasma, norm=plasma_norm)
plasma_sm.set_array([])

for i in range(opt_loc.shape[0]):
    if (bf_params['image_range'][0] < opt_loc[i][0] * 1e3 < bf_params['image_range'][1]) and \
    (bf_params['image_range'][2] < opt_loc[i][2] * 1e3 < bf_params['image_range'][3]):
        gCNRs = [narrow_gCNRs_avg[i], middle_gCNRs_avg[i], wide_gCNRs_avg[i], opt_gCNRs_avg[i]]
        pos_x = snap_value( opt_loc[i][0] * 1e3, snap_vals_x )
        pos_z = snap_value( opt_loc[i][2] * 1e3, snap_vals_z )

        for j, ax in enumerate([ax1, ax2, ax3, ax4]):
            the_gCNR = gCNRs[j].item()
            the_color = plasma( (the_gCNR - min_gCNR) / (max_gCNR - min_gCNR) )
            ax.annotate(f"{gCNRs[j]:.3f}", (pos_x, pos_z), bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=the_color, linewidth=1.5), 
                            color=the_color, backgroundcolor='white', fontsize=9, ha='center', va='center', weight='bold')

ax_cb = fig.add_subplot( gs[-3] )
cbar = fig.colorbar( im, cax=ax_cb)
cbar.set_label( "dB" )

ax_gCNR = fig.add_subplot( gs[-1] )
cbar = fig.colorbar( plasma_sm, cax=ax_gCNR, extend='min' )
cbar.set_label( "gCNR" )

for ext in ['png', 'pdf']:
    plt.savefig( f"./figures/figure8a.{ext}", bbox_inches='tight' )



