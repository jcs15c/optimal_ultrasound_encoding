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

if __name__ == "__main__":
    ######################### Generate data for the plot #########################
    # Set up predictor model parameters
    enc_params = s.default_enc_params
    bf_params = s.default_bf_params
    bf_params['image_range'] = [-25, 25, 15, 55]
    flipped_range = [-25, 25, 55, 15]

    # Get sample encoding sequence.
    delays = ue.calc_delays_planewaves( 15, span=75 )
    weights = ue.calc_uniform_weights( 15 )

    # Load the first three kinds of data we can image, which are conventional,
    #  as well as the PyTorch models that image them
    point_dataset = uu.UltrasoundDataset( f"./data/isolated_point_data" )
    point_acq_params = scipy.io.loadmat( f"./data/isolated_point_data/acq_params.mat" )
    point_model = PredictorModel(delays, weights, point_acq_params, enc_params, bf_params)

    lowdense_dataset = uu.UltrasoundDataset( f"./data/underdeveloped_speckle_data" )
    lowdense_acq_params = scipy.io.loadmat( f"./data/underdeveloped_speckle_data/acq_params.mat" )
    lowdense_model = PredictorModel(delays, weights, lowdense_acq_params, enc_params, bf_params)

    lesion_dataset = uu.UltrasoundDataset( f"./data/anechoic_lesion_data" )
    lesion_acq_params = scipy.io.loadmat( f"./data/anechoic_lesion_data/acq_params.mat" )
    lesion_model = PredictorModel(delays, weights, lesion_acq_params, enc_params, bf_params)

    datasets = [point_dataset, lowdense_dataset, lesion_dataset]
    models = [point_model, lowdense_model, lesion_model]

    # Get the imaging targets
    unencoded_targets = []
    synthetic_targets = []

    for dataset, model in zip( datasets, models ):
        datas, locs = [ torch.tensor( x ).unsqueeze(0) for x in dataset[0]]

        unencoded_targets.append( model.get_targets( datas, locs, 'unencoded')[0] )
        synthetic_targets.append( model.get_targets( datas, locs, 'synthetic')[0] )

    ######################### Create the plot #########################
    fig = plt.figure(figsize=(13, 6.5))
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.rcParams["figure.autolayout"] = True

    width_ratios = [0.25, 1, 0.1, 1, 0.1, 1, 0.1, 0.05]
    height_ratios = [0.1, 1, 0.1, 1]
    gs = GridSpec(len(height_ratios), len(width_ratios), figure=fig, width_ratios=width_ratios, height_ratios=height_ratios )
    data_axes = np.array( [[fig.add_subplot(gs[i, j]) for j in [1, 3, 5]] for i in [1, 3]] )

    names = ['5', '1000', '~56,000']
    for i in range(3):
        # Plot all the unencoded stuff
        im = data_axes[0, i].imshow( unencoded_targets[i], cmap='gray', extent=flipped_range, vmin=bf_params['dB_min'], vmax=0 )
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
        data_axes[1, i].imshow(  synthetic_targets[i], cmap='gray', extent=flipped_range, vmin=bf_params['dB_min'], vmax=0 )
        data_axes[1, i].set_yticks( data_axes[1, i].get_yticks()[1:-1] )
        if i == 0:
            data_axes[1, i].set_ylabel( "Axial (mm)" )
            data_axes[1, i].set_yticklabels( [int(x) for x in data_axes[1, i].get_yticks()] )
        else:
            data_axes[1, i].set_yticklabels([])
        data_axes[1, i].set_xlabel( "Lateral (mm)" )
        data_axes[1, i].set_xlim( flipped_range[0], flipped_range[1] )
        data_axes[1, i].set_ylim( flipped_range[2], flipped_range[3] )

    # Add an emphasis box on the middle column
    data_axes[1, 1].add_patch( matplotlib.patches.Rectangle( (0.0, 0.0), 1.0, 1.0, transform=data_axes[1, 1].transAxes, linewidth=10, edgecolor='red', facecolor='none' ) )

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
    ay1.text( 0.0, 0.5, r"\underline{Unencoded Image}", transform=ay1.transAxes, rotation=90, ha='center', va='center', fontsize=14 )
    ay2.text( 0.0, 0.5, r"\underline{Ground Truth Contrast}", transform=ay2.transAxes, rotation=90, ha='center', va='center', fontsize=14 )
    ax1.text( 0.5, 1.0, r"\underline{5 Scatterers}", transform=ax1.transAxes, ha='center', va='center', fontsize=14 )
    ax2.text( 0.5, 1.0, r"\underline{1000 Scatterers}", transform=ax2.transAxes, ha='center', va='center', fontsize=14 )
    ax3.text( 0.5, 1.0, r"\underline{$\sim$56,000 Scatterers}", transform=ax3.transAxes, ha='center', va='center', fontsize=14 )
    plt.rcParams['text.usetex'] = False

    # Plot the colorbar
    ay_cb = fig.add_subplot( gs[1:, -1] )
    fig.colorbar( im, cax=ay_cb )
    ay_cb.set_ylabel( "dB" )
    ay_cb.set_yticks( ay_cb.get_yticks() )
    ay_cb.set_yticklabels( [int(x) for x in ay_cb.get_yticks()] )

    for ext in ['png', 'pdf']:
        fig.savefig( f"./figures/figure2a.{ext}", bbox_inches='tight' )
