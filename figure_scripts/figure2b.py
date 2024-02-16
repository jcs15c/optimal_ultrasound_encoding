import project_root
import numpy as np

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
    bf_params['hist_match'] = False
    bf_params['image_range'] = [-20, 20, 22, 52]
    flipped_range = [-20, 20, 52, 22]

    # Get sample encodign sequence
    delays = ue.calc_delays_planewaves( 15, span=75 )
    weights = ue.calc_uniform_weights( 15 )

    # Load the last two kinds of data we can image, which are image-derived
    #  as well as the PyTorch models that image them
    image_dataset = uu.UltrasoundImageDataset( f"./data/image_derived_data" )
    image_acq_params = scipy.io.loadmat( f"./data/image_derived_data/acq_params.mat" )
    image_model = PredictorModel(delays, weights, image_acq_params, enc_params, bf_params)

    bin_dataset = uu.UltrasoundImageDataset( f"./data/binary_image_data" )
    bin_acq_params = scipy.io.loadmat( f"./data/binary_image_data/acq_params.mat" )
    bin_model = PredictorModel(delays, weights, bin_acq_params, enc_params, bf_params)

    datasets = [image_dataset, bin_dataset]
    models = [image_model, bin_model]

    # Get the imaging targets
    unencoded_targets = []
    synthetic_targets = []

    for dataset, model, idx, target_type in zip( datasets, models, [47, 101], ['image_contrast', 'image_synthetic'] ):
        datas, locs = [ torch.tensor( x ).unsqueeze(0) for x in dataset[idx]]

        unencoded_targets.append( model.get_targets( datas, locs, 'unencoded')[0] )
        synthetic_targets.append( model.get_targets( datas, locs, target_type)[0] )

    ######################### Create the plot #########################
    fig = plt.figure(figsize=(9, 6.5))
    fig.subplots_adjust(wspace=0, hspace=0)

    width_ratios = [0.25, 1, 0.1, 1, 0.1, 0.05]
    height_ratios = [0.1, 1, 0.1, 1]
    gs = GridSpec(len(height_ratios), len(width_ratios), figure=fig, width_ratios=width_ratios, height_ratios=height_ratios )

    plt.rcParams["figure.autolayout"] = True

    image_idx = 47
    image_datas, image_locs = [ torch.tensor( x ).unsqueeze(0) for x in image_dataset[image_idx]]

    bin_idx = 101
    bin_datas, bin_locs = [ torch.tensor( x ).unsqueeze(0) for x in bin_dataset[bin_idx]]

    ad1 = fig.add_subplot(gs[1, 1])
    im = ad1.imshow( unencoded_targets[0], cmap='gray', extent=flipped_range, vmin=bf_params['dB_min'], vmax=0 )
    ad1.set_ylabel( "Axial (mm)" )
    ad1.set_yticks( ad1.get_yticks()[1:-1] )
    ad1.set_yticklabels( [int(x) for x in ad1.get_yticks()])
    ad1.set_xticks( [-15, -7.5, 0, 7.5, 15] )
    ad1.set_xticklabels([])

    ad2 = fig.add_subplot(gs[1, 3])
    ad2.imshow( unencoded_targets[1], cmap='gray', extent=flipped_range, vmin=bf_params['dB_min'], vmax=0 )
    ad2.set_xticks( [-15, -7.5, 0, 7.5, 15] )
    ad2.set_xticklabels([])
    ad2.set_yticks( ad2.get_yticks()[1:-1] )
    ad2.set_yticklabels([])

    ad3 = fig.add_subplot(gs[3, 1])
    ad3.imshow( synthetic_targets[0], cmap='gray', extent=flipped_range, vmin=bf_params['dB_min'], vmax=0 )
    ad3.set_xlabel( "Lateral (mm)" )
    ad3.set_ylabel( "Axial (mm)" )
    ad3.set_xticks( [-15, -7.5, 0, 7.5, 15] )
    ad3.set_xticklabels( [float(x) for x in ad3.get_xticks()] )
    ad3.set_yticks( ad3.get_yticks()[1:-1] )
    ad3.set_yticklabels( [int(x) for x in ad3.get_yticks()] )

    ad4 = fig.add_subplot(gs[3, 3])
    ad4.imshow( synthetic_targets[1], cmap='gray', extent=flipped_range, vmin=bf_params['dB_min'], vmax=0 )
    ad4.set_xlabel( "Lateral (mm)" )
    ad4.set_yticks( ad4.get_yticks()[1:-1] )
    ad4.set_yticklabels([])
    ad4.set_xticks( [-15, -7.5, 0, 7.5, 15] )
    ad4.set_xticklabels( [float(x) for x in ad4.get_xticks()] )

    ay_cb = fig.add_subplot( gs[1:, -1] )
    fig.colorbar( im, cax=ay_cb )
    ay_cb.set_ylabel( "dB" )
    ay_cb.set_yticks( ay_cb.get_yticks() )
    ay_cb.set_yticklabels( [int(x) for x in ay_cb.get_yticks()] )

    plt.rcParams['text.usetex'] = True

    # Plot the y-axis supertitles
    ay1 = fig.add_subplot(gs[1, 0])
    ay1.set_axis_off()
    ay1.text( 0.0, 0.5, r"\underline{Unencoded Image}", transform=plt.gca().transAxes, rotation=90, ha='center', va='center', fontsize=14 )

    ay2 = fig.add_subplot(gs[3, 0])
    ay2.set_axis_off()
    ay2.text( 0.0, 0.5, r"\underline{Ground Truth Contrast}", transform=plt.gca().transAxes, rotation=90, ha='center', va='center', fontsize=14 )

    # Plot the x-axis supertitles
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.text( 0.5, 1.0, r"\underline{Arbitrarily Weighted Scatterers}", transform=plt.gca().transAxes, ha='center', va='center', fontsize=14 )
    ax1.set_axis_off()

    ax2 = fig.add_subplot(gs[0, 3])
    ax2.text( 0.5, 1.0, r"\underline{Binarized Contrast}", transform=plt.gca().transAxes, ha='center', va='center', fontsize=14 )
    ax2.set_axis_off()

    plt.rcParams['text.usetex'] = False

    for ext in ['png', 'pdf']:
        fig.savefig( f"./figures/figure2b.{ext}", bbox_inches='tight' )
