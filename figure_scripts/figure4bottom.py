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
import os

import scipy
import scipy.io

import torch

if __name__ == "__main__":
    ## Load the data to work on
    image_dataset = uu.UltrasoundImageDataset( f"./data/image_derived_data" )
    image_acq_params = scipy.io.loadmat( f"./data/image_derived_data/acq_params.mat" )
    image_datas, image_locs = [torch.tensor( x ).unsqueeze(0) for x in image_dataset[47]]

    binary_dataset = uu.UltrasoundImageDataset( f"./data/binary_image_data" )
    binary_acq_params = scipy.io.loadmat( f"./data/binary_image_data/acq_params.mat" )
    binary_datas, binary_locs = [torch.tensor( x ).unsqueeze(0) for x in binary_dataset[101]]

    ## Set up predictor model parameters
    enc_params = s.default_enc_params
    bf_params = s.default_bf_params
    bf_params['image_range'] = [-20, 20, 22, 52]
    flipped_range = [-20, 20, 52, 22]
    bf_params['hist_match'] = False
    dB_min = bf_params['dB_min']
    
    # Pull the sequences from files, or generate them

    # Optimized
    opt_delays = torch.tensor( np.loadtxt( f"optimized_sequences/full_parameterization/delays.csv", delimiter="," ), dtype=s.PTFLOAT  )
    opt_weights = torch.tensor( np.loadtxt( f"optimized_sequences/full_parameterization/weights.csv", delimiter="," ), dtype=s.PTFLOAT  )
    opt_sequence = (opt_delays, opt_weights)

    # Narrow FOV
    narrow_delays = ue.calc_delays_planewaves( opt_delays.shape[0], spacing=1 )
    narrow_weights = ue.calc_uniform_weights( narrow_delays.shape[0] )
    narrow_sequence = (narrow_delays, narrow_weights)

    # Wide FOV
    wide_delays = ue.calc_delays_planewaves( opt_delays.shape[0], span=75 )
    wide_weights = ue.calc_uniform_weights( wide_delays.shape[0] )
    wide_sequence = (wide_delays, wide_weights)

    # Make the plot
    fig = plt.figure(figsize=(13, 6.5))
    fig.subplots_adjust(wspace=0, hspace=0)

    width_ratios = [0.25, 1, 0.1, 1, 0.1, 1, 0.1, 0.05]
    height_ratios = [0.1, 1, 0.1, 1]
    gs = GridSpec(len(height_ratios), len(width_ratios), figure=fig, width_ratios=width_ratios, height_ratios=height_ratios )
    data_axes = np.array( [[fig.add_subplot(gs[i, j]) for j in [1, 3, 5]] for i in [1, 3]] )

    plt.rcParams["figure.autolayout"] = True

    for [i, (delays, weights)] in [[0, narrow_sequence], [1, wide_sequence], [2, opt_sequence]]:
        image_model = PredictorModel(delays, weights, image_acq_params, enc_params, bf_params )

        binary_model = PredictorModel(delays, weights, binary_acq_params, enc_params, bf_params )

        for model in [image_model, binary_model]:
            model.delays.requires_grad = False
            model.weights.requires_grad = False

        image_env = image_model.get_image_prediction( image_datas, image_locs )[0]
        binary_image = binary_model.get_image_prediction( binary_datas, binary_locs )[0]

        im = data_axes[0, i].imshow( image_env, cmap='gray', extent=flipped_range, vmin=bf_params['dB_min'], vmax=0 )
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
        data_axes[1, i].imshow( binary_image, cmap='gray', extent=flipped_range, vmin=bf_params['dB_min'], vmax=0 )
        data_axes[1, i].set_yticks( data_axes[1, i].get_yticks()[1:-1] )
        if i == 0:
            data_axes[1, i].set_ylabel( "Axial (mm)" )
            data_axes[1, i].set_yticklabels( [int(x) for x in data_axes[1, i].get_yticks()] )
        else:
            data_axes[1, i].set_yticklabels([])
        data_axes[1, i].set_xlabel( "Lateral (mm)" )
        data_axes[1, i].set_xlim( flipped_range[0], flipped_range[1] )
        data_axes[1, i].set_ylim( flipped_range[2], flipped_range[3] )

    plt.rcParams["text.usetex"] = True
    # Plot the y-axis supertitles
    ay1 = fig.add_subplot(gs[1, 0])
    ay1.set_axis_off()
    ay1.text( -0.2, 0.5, r"Arbitrarily"
                        "\n"
                        r"\underline{Weighted Scatterers}", transform=plt.gca().transAxes, rotation=90, ha='center', va='center', fontsize=14 )

    ay2 = fig.add_subplot(gs[3, 0])
    ay2.set_axis_off()
    ay2.text( 0.0, 0.5, r"\underline{Binarized Contrast}", transform=plt.gca().transAxes, rotation=90, ha='center', va='center', fontsize=14 )

    # Plot the x-axis supertitles
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.text( 0.5, 1.0, r"\underline{$1^\circ$ Spacing PW}", transform=plt.gca().transAxes, ha='center', va='center', fontsize=14 )
    ax1.set_axis_off()

    ax2 = fig.add_subplot(gs[0, 3])
    ax2.text( 0.5, 1.0, r"\underline{$10^\circ$ Spacing PW}", transform=plt.gca().transAxes, ha='center', va='center', fontsize=14 )
    ax2.set_axis_off()

    ax3 = fig.add_subplot(gs[0, 5])
    ax3.text( 0.5, 1.0, r"\underline{Optimized Encoding}", transform=plt.gca().transAxes, ha='center', va='center', fontsize=14 )
    ax3.set_axis_off()

    plt.rcParams["text.usetex"] = False

    # Plot the colorbar
    ay_cb = fig.add_subplot( gs[1:, -1] )
    fig.colorbar( im, cax=ay_cb )
    ay_cb.set_ylabel( "dB" )
    ay_cb.set_yticks( ay_cb.get_yticks() )
    ay_cb.set_yticklabels( [int(x) for x in ay_cb.get_yticks()] )


    for ext in ['png', 'pdf']:
        fig.savefig( f"figures/figure4bottom.{ext}", bbox_inches='tight' )
