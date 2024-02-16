import project_root
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import src.ultrasound_utilities as uu
import src.ultrasound_encoding as ue
import src.settings as s
from src.predictor_model import PredictorModel
import src.settings as s

import scipy
import scipy.io

import torch

if __name__ == "__main__":
    ######################### Generate data for the plot #########################
    ## Load the data to work on
    lesion_dataset = uu.UltrasoundDataset( f"./data/sample_lesion_data" )
    lesion_acq_params = scipy.io.loadmat( f"./data/sample_lesion_data/acq_params.mat" )
    lesion_datas, lesion_locs = [torch.tensor( x ).unsqueeze(0) for x in lesion_dataset[0]]

    point_dataset = uu.UltrasoundDataset( f"./data/sample_point_data" )
    point_acq_params = scipy.io.loadmat( f"./data/sample_point_data/acq_params.mat" )
    point_datas, point_locs = [torch.tensor( x ).unsqueeze(0) for x in point_dataset[0]]

    ## Set up predictor model parameters
    enc_params = s.default_enc_params
    bf_params = s.default_bf_params
    bf_params['image_range'] = [-25, 25, 15, 55]
    flipped_range = [-25, 25, 55, 15]
    
    ## Pull the sequences from files, or generate them

    # Optimized
    opt_delays = torch.tensor( np.loadtxt( f"optimal_sequences/full_parameterization/delays.csv", delimiter="," ), dtype=s.PTFLOAT  )
    opt_weights = torch.tensor( np.loadtxt( f"optimal_sequences/full_parameterization/weights.csv", delimiter="," ), dtype=s.PTFLOAT  )
    opt_sequence = (opt_delays, opt_weights)

    # Narrow FOV
    narrow_delays = ue.calc_delays_planewaves( opt_delays.shape[0], spacing=1 )
    narrow_weights = ue.calc_uniform_weights( narrow_delays.shape[0] )
    narrow_sequence = (narrow_delays, narrow_weights)

    # Wide FOV
    wide_delays = ue.calc_delays_planewaves( opt_delays.shape[0], span=75 )
    wide_weights = ue.calc_uniform_weights( wide_delays.shape[0] )
    wide_sequence = (wide_delays, wide_weights)

    ## Collect the 6 images to plot
    lesion_images = []
    point_images = []

    for i, (delays, weights) in enumerate( [narrow_sequence, wide_sequence, opt_sequence] ):
        # Define the PyTorch model with each sequence
        lesion_model = PredictorModel(delays, weights, lesion_acq_params, enc_params, bf_params )
        point_model = PredictorModel(delays, weights, point_acq_params, enc_params, bf_params )

        for model in [point_model, lesion_model]:
            model.delays.requires_grad = False
            model.weights.requires_grad = False

        # Generate the images
        lesion_images.append( lesion_model.get_image_prediction( lesion_datas, lesion_locs )[0] )
        point_images.append( point_model.get_image_prediction( point_datas, point_locs )[0] )
    
    ######################### Create the plot #########################
    fig = plt.figure(figsize=(13, 6.5))
    fig.subplots_adjust(wspace=0, hspace=0)

    width_ratios = [0.25, 1, 0.1, 1, 0.1, 1, 0.1, 0.05]
    height_ratios = [0.1, 1, 0.1, 1]
    gs = GridSpec(len(height_ratios), len(width_ratios), figure=fig, width_ratios=width_ratios, height_ratios=height_ratios )
    data_axes = np.array( [[fig.add_subplot(gs[i, j]) for j in [1, 3, 5]] for i in [1, 3]] )

    plt.rcParams["figure.autolayout"] = True

    for i in range( 3 ):
        im = data_axes[0, i].imshow( lesion_images[i], cmap='gray', extent=flipped_range, vmin=bf_params['dB_min'], vmax=0 )
        data_axes[0, i].set_yticks( data_axes[0, i].get_yticks()[1:-1] )
        if i == 0:
            data_axes[0, i].set_ylabel( "Axial (mm)" )
            data_axes[0, i].set_yticklabels( [int(x) for x in data_axes[0, i].get_yticks()] )
        else:
            data_axes[0, i].set_yticklabels([])
        data_axes[0, i].set_xticklabels([])
        data_axes[0, i].set_xlim( flipped_range[0], flipped_range[1] )
        data_axes[0, i].set_ylim( flipped_range[2], flipped_range[3] )

        # Plot all the synthetic stuff
        data_axes[1, i].imshow( point_images[i], cmap='gray', extent=flipped_range, vmin=bf_params['dB_min'], vmax=0 )
        data_axes[1, i].set_yticks( data_axes[1, i].get_yticks()[1:-1] )
        if i == 0:
            data_axes[1, i].set_ylabel( "Axial (mm)" )
            data_axes[1, i].set_yticklabels( [int(x) for x in data_axes[1, i].get_yticks()] )
        else:
            data_axes[1, i].set_yticklabels([])
        data_axes[1, i].set_xlabel( "Lateral (mm)" )
        data_axes[1, i].set_xlim( flipped_range[0], flipped_range[1] )
        data_axes[1, i].set_ylim( flipped_range[2], flipped_range[3] )

    # Plot the y-axis supertitles
    ay1 = fig.add_subplot(gs[1, 0])
    ay1.set_axis_off()

    ay2 = fig.add_subplot(gs[3, 0])
    ay2.set_axis_off()
 
    # Plot the x-axis supertitles
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_axis_off()

    ax2 = fig.add_subplot(gs[0, 3])
    ax2.set_axis_off()

    ax3 = fig.add_subplot(gs[0, 5])
    ax3.set_axis_off()
    
    plt.rcParams['text.usetex'] = True
    ay1.text( 0.0, 0.5, r"\underline{Anechoic Lesions}", transform=ay1.transAxes, rotation=90, ha='center', va='center', fontsize=14 )
    ay2.text( 0.0, 0.5, r"\underline{Point Targets}", transform=ay2.transAxes, rotation=90, ha='center', va='center', fontsize=14 )
    
    ax1.text( 0.5, 1.0, r"\underline{$1^\circ$ Spacing PW}", transform=ax1.transAxes, ha='center', va='center', fontsize=14 )
    ax2.text( 0.5, 1.0, r"\underline{$10^\circ$ Spacing PW}", transform=ax2.transAxes, ha='center', va='center', fontsize=14 )
    ax3.text( 0.5, 1.0, r"\underline{Optimized Encoding}", transform=ax3.transAxes, ha='center', va='center', fontsize=14 )
    plt.rcParams['text.usetex'] = False

    # Plot the colorbar
    ay_cb = fig.add_subplot( gs[1:, -1] )
    fig.colorbar( im, cax=ay_cb )
    ay_cb.set_ylabel( "dB" )
    ay_cb.set_yticks( ay_cb.get_yticks() )
    ay_cb.set_yticklabels( [int(x) for x in ay_cb.get_yticks()] )

    for ext in ['png', 'pdf']:
        fig.savefig( f"figures/figure4top.{ext}", bbox_inches='tight' )
