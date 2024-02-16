import project_root
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import src.ultrasound_utilities as uu
import src.ultrasound_encoding as ue
import src.settings as s
from src.predictor_model import PredictorModel

import scipy
import scipy.io

import torch

######################### Generate data for the plot #########################
## Load the data to work on
dataset = uu.UltrasoundDataset( f"./data/single_point_data" )
acq_params = scipy.io.loadmat( f"./data/single_point_data/acq_params.mat" )
[data, loc] = [torch.tensor( x ) for x in dataset[0]]

## Set up model parameters
enc_params = s.default_enc_params
bf_params = s.default_bf_params
bf_params['image_range'] = [-25, 25, 15, 55]
flipped_range = [-25, 25, 55, 15]

## Pull the sequences from files, or generate them
##  and define the PyTorch model with each sequence

# Optimized
opt_delays = torch.tensor( np.loadtxt( f"optimized_sequences/full_parameterization/delays.csv", delimiter="," ), dtype=s.PTFLOAT  )
opt_weights = torch.tensor( np.loadtxt( f"optimized_sequences/full_parameterization/weights.csv", delimiter="," ), dtype=s.PTFLOAT  )
opt_model = PredictorModel(opt_delays, opt_weights, acq_params, enc_params, bf_params )

# Narrow FOV
narrow_delays = ue.calc_delays_planewaves( opt_delays.shape[0], spacing=1 )
narrow_weights = ue.calc_uniform_weights( narrow_delays.shape[0] )
narrow_model = PredictorModel(narrow_delays, narrow_weights, acq_params,  enc_params, bf_params )

# Wide FOV
wide_delays = ue.calc_delays_planewaves( opt_delays.shape[0], span=75 )
wide_weights = ue.calc_uniform_weights( wide_delays.shape[0] )
wide_model = PredictorModel(wide_delays, wide_weights, acq_params, enc_params, bf_params )

# Hadamard encoding
hadamard_delays = ue.calc_delays_zeros( opt_delays.shape[0] )
hadamard_weights = ue.calc_hadamard_weights( wide_delays.shape[0] )
hadamard_model = PredictorModel(hadamard_delays, hadamard_weights, acq_params, enc_params, bf_params )

for model in [narrow_model, wide_model, opt_model, hadamard_model]:
    model.delays.requires_grad = False
    model.weights.requires_grad = False

## Store lesion radii for cystic resolution
rs = np.linspace(0, 10, 10)
npts = len(rs)

opt_cystic_contrasts = np.zeros( npts )
narrow_cystic_contrasts = np.zeros( npts )
wide_cystic_contrasts = np.zeros( npts )
hadamard_cystic_contrasts = np.zeros( npts )

for i in range(npts):
    opt_cystic_contrasts[i] = opt_model.cystic_contrast( data, loc, rs[i] )
    narrow_cystic_contrasts[i] = narrow_model.cystic_contrast( data, loc, rs[i] )
    wide_cystic_contrasts[i] = wide_model.cystic_contrast( data, loc, rs[i] )
    hadamard_cystic_contrasts[i] = hadamard_model.cystic_contrast( data, loc, rs[i] )

# Store minimum radius for detectability at -20 dB
contrast = -20
opt_detect = opt_model.cystic_resolution( data, loc, contrast )
narrow_detect = narrow_model.cystic_resolution( data, loc, contrast )
wide_detect = wide_model.cystic_resolution( data, loc, contrast )
hadamard_detect = hadamard_model.cystic_resolution( data, loc, contrast )

## Store the individual sample images
opt_image = opt_model.get_image_prediction( data.unsqueeze(0), loc.unsqueeze(0) )[0]
narrow_image = narrow_model.get_image_prediction( data.unsqueeze(0), loc.unsqueeze(0) )[0]
wide_image = wide_model.get_image_prediction( data.unsqueeze(0), loc.unsqueeze(0) )[0]
hadamard_image = hadamard_model.get_image_prediction( data.unsqueeze(0), loc.unsqueeze(0) )[0]

######################### Create the plot #########################
fig = plt.figure(figsize=(13, 6.5))
fig.subplots_adjust(wspace=0, hspace=0)

width_ratios = [0.45, 0.08, 0.2, 0.025, 0.2, 0.025, 0.015]
height_ratios = [0.1, 1, 0.1, 1]
gs = GridSpec(len(height_ratios), len(width_ratios), figure=fig, width_ratios=width_ratios, height_ratios=height_ratios )

data_axes = np.array( [[fig.add_subplot(gs[i, j]) for j in [2, 4]] for i in [1, 3]] )
label_axes = np.array( [[fig.add_subplot(gs[i, j]) for j in [2, 4]] for i in [0, 2]] )

plt.rcParams["figure.autolayout"] = True
plt.rcParams['text.usetex'] = True

ax_plot = fig.add_subplot(gs[1:, 0])
ax_plot.plot( rs, narrow_cystic_contrasts, 'b', label="Narrow FOV" )
ax_plot.plot( rs, wide_cystic_contrasts, 'y', label="Wide FOV" )
ax_plot.plot( rs, hadamard_cystic_contrasts, 'r', label="Hadamard" )
ax_plot.plot( rs, opt_cystic_contrasts, 'g', linewidth=3, label="Optimized" )
ax_plot.set_xlabel( "Cyst Radius (mm)" )
ax_plot.set_ylabel( "Cyst Contrast (dB)" )

min_y = ax_plot.get_ylim()[0]
max_y = ax_plot.get_ylim()[1]
ax_plot.plot( [narrow_detect, narrow_detect],     [min_y, contrast], 'b--' )
ax_plot.plot( [wide_detect, wide_detect],         [min_y, contrast], 'y--' )
ax_plot.plot( [hadamard_detect, hadamard_detect], [min_y, contrast], 'r--' )
ax_plot.plot( [opt_detect, opt_detect],           [min_y, contrast], 'g--', linewidth=3 )
ax_plot.plot( rs, np.ones_like( rs ) * contrast, 'k', label="Threshold" )
ax_plot.set_yticks( ax_plot.get_yticks() )
ax_plot.set_yticklabels( [int(x) for x in ax_plot.get_yticks()])
ax_plot.set_xticks( ax_plot.get_xticks() )
ax_plot.set_xticklabels( [int(x) for x in ax_plot.get_xticks()] )
ax_plot.set_ylim([min_y, max_y])
ax_plot.set_xlim([min(rs), max(rs)])
ax_plot.legend()

im = data_axes[0,0].imshow( opt_image, extent=flipped_range, cmap='gray', vmin=-60, vmax=0 )
data_axes[0,0].set_xlabel( "Lateral (mm)" )
data_axes[0,0].set_ylabel( "Axial (mm)" )
data_axes[0,0].set_xlim( flipped_range[0], flipped_range[1] )
data_axes[0,0].set_ylim( flipped_range[2], flipped_range[3] )
label_axes[0,0].text( 0.5, -0.75, r"\underline{Optimized Encoding}", transform=label_axes[0,0].transAxes, ha='center', va='center', fontsize=14 )
label_axes[0,0].set_axis_off()

data_axes[0,1].imshow( narrow_image, extent=flipped_range, cmap='gray', vmin=-60, vmax=0 )
data_axes[0,1].set_yticklabels([])
data_axes[0,1].set_xlabel( "Lateral (mm)" )
data_axes[0,1].set_xlim( flipped_range[0], flipped_range[1] )
data_axes[0,1].set_ylim( flipped_range[2], flipped_range[3] )
label_axes[0,1].text( 0.5, -0.75, r"\underline{Narrow PW Steering}", transform=label_axes[0, 1].transAxes, ha='center', va='center', fontsize=14 )
label_axes[0,1].set_axis_off()

data_axes[1,0].imshow( wide_image, extent=flipped_range, cmap='gray', vmin=-60, vmax=0 )
data_axes[1,0].set_xlabel( "Lateral (mm)" )
data_axes[1,0].set_ylabel( "Axial (mm)" )
data_axes[1,0].set_xlim( flipped_range[0], flipped_range[1] )
data_axes[1,0].set_ylim( flipped_range[2], flipped_range[3] )
label_axes[1,0].text( 0.5, -0.75, r"\underline{Wide PW Steering}", transform=label_axes[1, 0].transAxes, ha='center', va='center', fontsize=14 )
label_axes[1,0].set_axis_off()

data_axes[1,1].imshow( hadamard_image, extent=flipped_range, cmap='gray', vmin=-60, vmax=0 )
data_axes[1,1].set_xlabel( "Lateral (mm)" )
data_axes[1,1].set_yticklabels([])
data_axes[1,1].set_xlim( flipped_range[0], flipped_range[1] )
data_axes[1,1].set_ylim( flipped_range[2], flipped_range[3] )
label_axes[1,1].text( 0.5, -0.75, r"\underline{Hadamard Encoding}", transform=label_axes[1, 1].transAxes, ha='center', va='center', fontsize=14 )
label_axes[1,1].set_axis_off()

ay_cb = fig.add_subplot( gs[1:, -1] )
fig.colorbar( im, cax=ay_cb )
ay_cb.set_ylabel( "dB" )
ay_cb.set_yticks( ay_cb.get_yticks() )
ay_cb.set_yticklabels( [int(x) for x in ay_cb.get_yticks()] )

the_x = 0.49
line = plt.Line2D([the_x, the_x], [0, 0.9], transform=fig.transFigure, color='k', linestyle='--')
fig.add_artist(line)

fig.text( 0.1, 0.87, "(a)", transform=fig.transFigure, ha='center', va='center', fontsize=20 )
fig.text( 0.51, 0.87, "(b)", transform=fig.transFigure, ha='center', va='center', fontsize=20 )

for ext in ['png', 'pdf']:
    fig.savefig( f"figures/figure5.{ext}", bbox_inches='tight' )