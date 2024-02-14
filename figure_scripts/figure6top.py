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

## Set up model parameters
enc_params = s.default_enc_params
bf_params = s.default_bf_params
bf_params['image_range'] = [-25, 25, 15, 55]
bf_params['roi_pads'] = [0.8, 1.2]
flipped_range = [-25, 25, 55, 15]

dataset = uu.UltrasoundDataset( f"./data/anechoic_lesion_data" )
acq_params = scipy.io.loadmat( f"./data/anechoic_lesion_data/acq_params.mat" )

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

## Store results for lesions
N_data = min( 4, len(dataset) )

bin_count = 3
gCNR_sums = torch.zeros( (4, bin_count, bin_count) )
gCNR_counts = torch.zeros( (4, bin_count, bin_count) )

for i in range( N_data ):
    print(i + 1, '/', N_data)
    [datas, locs] = [torch.tensor( x ).unsqueeze(0) for x in dataset[i]]

    for (n, model) in enumerate( [narrow_model, wide_model, hadamard_model, opt_model] ):
        prediction = model.get_image_prediction( datas, locs )[0]

        the_sum, the_counts = ui.binned_average_gCNR( prediction, locs[0], bf_params, bin_count, rpads=bf_params['roi_pads'])
        gCNR_sums[n, :, :] += the_sum
        gCNR_counts[n, :, :] += the_counts

binned_gCNRs = gCNR_sums / gCNR_counts

fig = plt.figure(figsize=(6.5, 7))
fig.subplots_adjust(wspace=0, hspace=0)

plt.rcParams["figure.autolayout"] = True

height_ratios = [0.1, 1, 0.1, 1]
width_ratios = [1, 0.1, 1, 0.1, 0.05]

gs = GridSpec(len(height_ratios), len(width_ratios), figure=fig, width_ratios=width_ratios, height_ratios=height_ratios )

data_axes = np.array( [[fig.add_subplot(gs[i, j]) for j in [0, 2]] for i in [1, 3]] )
label_axes = np.array( [[fig.add_subplot(gs[i, j]) for j in [0, 2]] for i in [0, 2]] )

gCNR_ax = fig.add_subplot( gs[1, -1] )

narrow_binned_gCNRs = binned_gCNRs[0, :, :]
wide_binned_gCNRs = binned_gCNRs[1, :, :]
hadamard_binned_gCNRs = binned_gCNRs[2, :, :]
opt_binned_gCNRs = binned_gCNRs[3, :, :]

model_names = [r"\underline{Optimized Encoding}", r"\underline{Narrow PW Steering}", r"\underline{Wide PW Steering}", r"\underline{Hadamard Encoding}"]
binned_gCNRs = [opt_binned_gCNRs, narrow_binned_gCNRs, wide_binned_gCNRs, hadamard_binned_gCNRs]

# calc max and min among all gCNRs
max_gCNR = 1.0
min_gCNR = 0.8

# Define the plasma colormap
plasma = matplotlib.cm.get_cmap('plasma')

dB_im = None

# Work on the first row of gCNRs
data_sets = [dataset[i] for i in [0, 0, 0, 0]]
for i, dax in enumerate( data_axes.flatten() ):
    datas, locs = [torch.tensor(x).unsqueeze(0) for x in data_sets[i]]
    image = models[i].get_image_prediction( datas, locs )[0]

    dax.set_xlim( flipped_range[0], flipped_range[1] )
    dax.set_ylim( flipped_range[2], flipped_range[3] )
    dB_im = dax.imshow( image, cmap='gray', extent=flipped_range, vmin=bf_params['dB_min'], vmax=0 )
    
    x_pts = np.linspace( bf_params['image_range'][0], bf_params['image_range'][1], bin_count+1 )
    z_pts = np.linspace( bf_params['image_range'][2], bf_params['image_range'][3], bin_count+1 )
    
    for x_i in range(bin_count):
        for z_i in range(bin_count):
            x_i_center = (x_pts[x_i] + x_pts[x_i+1])/2
            z_i_center = (z_pts[z_i] + z_pts[z_i+1])/2
            plt.rcParams['text.usetex'] = False

            this_gCNR = binned_gCNRs[i][x_i, z_i].item()
            this_color = plasma( (this_gCNR - min_gCNR) / (max_gCNR - min_gCNR) )

            dax.annotate(f"{binned_gCNRs[i][x_i, z_i]:.3f}", xy=(x_i_center, z_i_center), bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=this_color, linewidth=1.5),
                    color=this_color, fontsize=9, ha='center', va='center', weight='bold')
            plt.rcParams['text.usetex'] = True

    for x in x_pts:
        dax.axvline( x=x, color='k', linewidth=0.75 )
    for z in z_pts:
        dax.axhline( y=z, color='k', linewidth=0.75 )

# Do a frankly ludicrous amount of processing to tidy up the labels
data_axes[0,0].set_yticks( data_axes[0,0].get_yticks() )
data_axes[0,0].set_yticklabels([int(x) for x in data_axes[0,0].get_yticks()])
data_axes[0,0].set_xticks( data_axes[0,0].get_xticks() )
data_axes[0,0].set_xticklabels( [int(x) for x in data_axes[0,0].get_xticks()] ) 
data_axes[0,0].set_xlabel("Lateral (mm)")
data_axes[0,0].set_ylabel( "Axial (mm)" )
data_axes[0,0].set_xlim( bf_params['image_range'][0], bf_params['image_range'][1] )
data_axes[0,0].set_ylim( bf_params['image_range'][3], bf_params['image_range'][2] )
label_axes[0,0].set_axis_off()

data_axes[0,1].set_yticks( data_axes[0,1].get_yticks() )
data_axes[0,1].set_yticklabels([])
data_axes[0,1].set_xticks( data_axes[0,1].get_xticks() )
data_axes[0,1].set_xticklabels( [int(x) for x in data_axes[0,1].get_xticks()] )
data_axes[0,1].set_xlabel("Lateral (mm)")
data_axes[0,1].set_xlim( bf_params['image_range'][0], bf_params['image_range'][1] )
data_axes[0,1].set_ylim( bf_params['image_range'][3], bf_params['image_range'][2] )
label_axes[0,1].set_axis_off()

data_axes[1,0].set_yticks( data_axes[0,1].get_yticks() )
data_axes[1,0].set_yticklabels([int(x) for x in data_axes[1,0].get_yticks()])
data_axes[1,0].set_xticks( data_axes[0,1].get_xticks() )
data_axes[1,0].set_xticklabels( [int(x) for x in data_axes[1,0].get_xticks()] )
data_axes[1,0].set_xlabel("Lateral (mm)")
data_axes[1,0].set_ylabel( "Axial (mm)" )
data_axes[1,0].set_xlim( bf_params['image_range'][0], bf_params['image_range'][1] )
data_axes[1,0].set_ylim( bf_params['image_range'][3], bf_params['image_range'][2] )
label_axes[1,0].set_axis_off()

data_axes[1,1].set_yticks( data_axes[1,1].get_yticks() )
data_axes[1,1].set_yticklabels([])
data_axes[1,1].set_xticks( data_axes[1,1].get_xticks() )
data_axes[1,1].set_xticklabels( [int(x) for x in data_axes[1,1].get_xticks()] )
data_axes[1,1].set_xlabel("Lateral (mm)")
data_axes[1,1].set_xlim( bf_params['image_range'][0], bf_params['image_range'][1] )
data_axes[1,1].set_ylim( bf_params['image_range'][3], bf_params['image_range'][2] )
label_axes[1,1].set_axis_off()


# Add the gCNR colorbar
norm = matplotlib.colors.Normalize(vmin=min_gCNR, vmax=max_gCNR)
sm = plt.cm.ScalarMappable(cmap=plasma, norm=norm)
sm.set_array([])
gCNR_cbar = fig.colorbar(sm, cax=gCNR_ax, extend='min' )
gCNR_cbar.set_label( "gCNR" )
gCNR_cbar.ax.yaxis.set_ticks( [0.8, 0.85, 0.9, 0.95, 1.0] )
gCNR_cbar.ax.yaxis.set_ticklabels( [0.8, 0.85, 0.9, 0.95, 1.0] )

plt.rcParams['text.usetex'] = True
label_axes[0,0].text( 0.5, -0.6, r"\underline{Optimized Encoding}", transform=label_axes[0,0].transAxes, ha='center', va='center', fontsize=14 )
label_axes[0,1].text( 0.5, -0.6, r"\underline{Narrow PW Steering}", transform=label_axes[0,1].transAxes, ha='center', va='center', fontsize=14 )
label_axes[1,0].text( 0.5, -0.6, r"\underline{Wide PW Steering}", transform=label_axes[1,0].transAxes, ha='center', va='center', fontsize=14 )
label_axes[1,1].text( 0.5, -0.6, r"\underline{Hadamard Encoding}", transform=label_axes[1,1].transAxes, ha='center', va='center', fontsize=14 )
plt.rcParams['text.usetex'] = False

for ext in ['png', 'pdf']:
    fig.savefig( f"figures/figure6top.{ext}", bbox_inches='tight' )