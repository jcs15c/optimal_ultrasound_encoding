import project_root
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Bbox

import src.ultrasound_imaging as ui
import src.ultrasound_utilities as uu
import src.ultrasound_encoding as ue
import src.settings as s
from src.predictor_model import PredictorModel

import scipy
import scipy.io
import os

import torch

## Load the data to work on
dataset = uu.UltrasoundDataset( f"./data/single_point_grid_data" )
acq_params = scipy.io.loadmat( f"./data/single_point_grid_data/acq_params.mat" )

## Set up model parameters
enc_params = s.default_enc_params
bf_params = s.default_bf_params
bf_params['image_range'] = [-25, 25, 15, 55]
bf_params['roi_pads'] = [0.8, 1.2]
flipped_range = [-25, 25, 55, 15]

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

models = [opt_model, narrow_model, wide_model, hadamard_model]
for model in models:
    model.delays.requires_grad = False
    model.weights.requires_grad = False

contrast = -20

opt_cystic_resolutions = np.zeros( (len(dataset), 3 ) )
narrow_cystic_resolutions = np.zeros( (len(dataset), 3 ) )
wide_cystic_resolutions = np.zeros( (len(dataset), 3 ) )
hadamard_cystic_resolutions = np.zeros( (len(dataset), 3 ) )

for i in range( len(dataset) ):
    [data, loc] = [torch.tensor( x ) for x in dataset[i]]
    print( i, '/', len(dataset) )

    opt_cystic_resolutions[i][0] = loc[0,0]
    opt_cystic_resolutions[i][1] = loc[0,2]
    opt_cystic_resolutions[i][2] = opt_model.cystic_resolution( data, loc, contrast ) 

    narrow_cystic_resolutions[i][0] = loc[0,0]
    narrow_cystic_resolutions[i][1] = loc[0,2]
    narrow_cystic_resolutions[i][2] = narrow_model.cystic_resolution( data, loc, contrast )

    wide_cystic_resolutions[i][0] = loc[0,0]
    wide_cystic_resolutions[i][1] = loc[0,2]
    wide_cystic_resolutions[i][2] = wide_model.cystic_resolution( data, loc, contrast )

    hadamard_cystic_resolutions[i][0] = loc[0,0]
    hadamard_cystic_resolutions[i][1] = loc[0,2]
    hadamard_cystic_resolutions[i][2] = hadamard_model.cystic_resolution( data, loc, contrast )


# Get the grid that the targets are on
unique_x = np.unique( opt_cystic_resolutions[:, 0] )
unique_z = np.unique( opt_cystic_resolutions[:, 1] )
x_targets, y_targets = np.meshgrid( unique_x, unique_z )
narrow_z = np.zeros_like( x_targets )
opt_z = np.zeros_like( x_targets )
wide_z = np.zeros_like( x_targets )
hadamard_z = np.zeros_like( x_targets )

for i in range( len(opt_cystic_resolutions) ):
    x_i = np.where( unique_x == opt_cystic_resolutions[i, 0] )[0][0]
    z_i = np.where( unique_z == opt_cystic_resolutions[i, 1] )[0][0]
    narrow_z[z_i, x_i] = narrow_cystic_resolutions[i, 2]
    opt_z[z_i, x_i] = opt_cystic_resolutions[i, 2]
    wide_z[z_i, x_i] = wide_cystic_resolutions[i, 2]
    hadamard_z[z_i, x_i] = hadamard_cystic_resolutions[i, 2]

fig = plt.figure(figsize=(6.5, 7))
fig.subplots_adjust(wspace=0, hspace=0)

plt.rcParams["figure.autolayout"] = True

height_ratios = [0.1, 1, 0.1, 1]
width_ratios = [1, 0.1, 1, 0.1, 0.05]

gs = GridSpec(len(height_ratios), len(width_ratios), figure=fig, width_ratios=width_ratios, height_ratios=height_ratios )
data_axes = np.array( [[fig.add_subplot(gs[i, j]) for j in [0, 2]] for i in [1, 3]] )
label_axes = np.array( [[fig.add_subplot(gs[i, j]) for j in [0, 2]] for i in [0, 2]] )

vmin = 0
vmax = 10

# Plot opt
im = data_axes[0,0].imshow( opt_z, cmap='viridis', extent=flipped_range, vmin=vmin, vmax=vmax )
data_axes[0,0].set_yticks( data_axes[0,0].get_yticks() )
data_axes[0,0].set_yticklabels([int(x) for x in data_axes[0,0].get_yticks()])
data_axes[0,0].set_xticks( data_axes[0,0].get_xticks() )
data_axes[0,0].set_xticklabels( [int(x) for x in data_axes[0,0].get_xticks()] )
data_axes[0,0].set_xlabel("Lateral (mm)")
data_axes[0,0].set_ylabel( "Axial (mm)" )
data_axes[0,0].set_xlim( bf_params['image_range'][0], bf_params['image_range'][1] )
data_axes[0,0].set_ylim( bf_params['image_range'][3], bf_params['image_range'][2] )
label_axes[0,0].set_axis_off()

# Plot narrow
im = data_axes[0,1].imshow( narrow_z, cmap='viridis', extent=flipped_range, vmin=vmin, vmax=vmax )
data_axes[0,1].set_yticks( data_axes[0,1].get_yticks() )
data_axes[0,1].set_yticklabels([])
data_axes[0,1].set_xticks( data_axes[0,1].get_xticks() )
data_axes[0,1].set_xticklabels( [int(x) for x in data_axes[0,1].get_xticks()] )
data_axes[0,1].set_xlabel("Lateral (mm)")
data_axes[0,1].set_xlim( bf_params['image_range'][0], bf_params['image_range'][1] )
data_axes[0,1].set_ylim( bf_params['image_range'][3], bf_params['image_range'][2] )
label_axes[0,1].set_axis_off()

# Plot wide
im = data_axes[1,0].imshow( wide_z, cmap='viridis', extent=flipped_range, vmin=vmin, vmax=vmax )
data_axes[1,0].set_yticks( data_axes[1,0].get_yticks() )
data_axes[1,0].set_yticklabels([int(x) for x in data_axes[1,0].get_yticks()])
data_axes[1,0].set_xticks( data_axes[1,0].get_xticks() )
data_axes[1,0].set_xticklabels( [int(x) for x in data_axes[1,0].get_xticks()] )
data_axes[1,0].set_xlabel("Lateral (mm)")
data_axes[1,0].set_ylabel( "Axial (mm)" )
data_axes[1,0].set_xlim( bf_params['image_range'][0], bf_params['image_range'][1] )
data_axes[1,0].set_ylim( bf_params['image_range'][3], bf_params['image_range'][2] )
label_axes[1,0].set_axis_off()

# Plot hadamard
im = data_axes[1,1].imshow( hadamard_z, cmap='viridis', extent=flipped_range, vmin=vmin, vmax=vmax )
data_axes[1,1].set_yticks( data_axes[1,1].get_yticks() )
data_axes[1,1].set_yticklabels([])
data_axes[1,1].set_xticks( data_axes[1,1].get_xticks() )
data_axes[1,1].set_xticklabels( [int(x) for x in data_axes[1,1].get_xticks()] )
data_axes[1,1].set_xlabel("Lateral (mm)")
data_axes[1,1].set_xlim( bf_params['image_range'][0], bf_params['image_range'][1] )
data_axes[1,1].set_ylim( bf_params['image_range'][3], bf_params['image_range'][2] )
label_axes[1,1].set_axis_off()

ay_cb = fig.add_subplot( gs[1, -1] )
fig.colorbar( im, cax=ay_cb, extend='max' )
ay_cb.set_yticks( ay_cb.get_yticks() )
ay_cb.set_yticklabels( [int(x) for x in ay_cb.get_yticks()] )
ay_cb.set_ylabel( "Cystic Resolution (mm)" )

plt.rcParams['text.usetex'] = True
label_axes[0,0].text( 0.5, -0.6, r"\underline{Optimized Encoding}", transform=label_axes[0,0].transAxes, ha='center', va='center', fontsize=14 )
label_axes[0,1].text( 0.5, -0.6, r"\underline{Narrow PW Steering}", transform=label_axes[0,1].transAxes, ha='center', va='center', fontsize=14 )
label_axes[1,0].text( 0.5, -0.6, r"\underline{Wide PW Steering}", transform=label_axes[1,0].transAxes, ha='center', va='center', fontsize=14 )
label_axes[1,1].text( 0.5, -0.6, r"\underline{Hadamard Encoding}", transform=label_axes[1,1].transAxes, ha='center', va='center', fontsize=14 )
plt.rcParams['text.usetex'] = False

for ext in ['png', 'pdf']:
    fig.savefig( f"figures/figure6bottom.{ext}", bbox_inches='tight' )
