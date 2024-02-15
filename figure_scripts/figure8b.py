import project_root
import torch
import src.ultrasound_encoding as ue
import src.ultrasound_imaging as ui
import src.ultrasound_utilities as uu
import src.ultrasound_experiments as ux
import src.settings as s

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import os
import scipy
import scipy.io

import numpy as np

## Set up model parameters
enc_params = s.default_enc_params
bf_params = s.default_bf_params
total_image_range = [-24, 24, 15, 64]
flipped_image_range = [-24, 24, 64, 15]
image_dims = [80, 120]

x_grid = torch.linspace(total_image_range[0], total_image_range[1], steps=image_dims[0])/1000 # In units of meters
z_grid = torch.linspace(total_image_range[2], total_image_range[3], steps=image_dims[1])/1000
image_grid = [x_grid, z_grid]

threshold = -20

opt_dataset = ux.ExperimentalDataset( "data/wire_target_data/optimized" )
narrow_dataset = ux.ExperimentalDataset( "data/wire_target_data/planewaves_15" )
middle_dataset = ux.ExperimentalDataset( "data/wire_target_data/planewaves_60" )
wide_dataset = ux.ExperimentalDataset( "data/wire_target_data/planewaves_150" )
datasets = [narrow_dataset, middle_dataset, wide_dataset, opt_dataset]

def plot_montage():
    fig = plt.figure(figsize=(15, 4.5))

    height_ratios = [1]
    width_ratios = [1, 0.1, 1, 0.1, 1, 0.1, 1, 0.1, 0.05, 0.3, 0.05]

    gs = GridSpec(len(height_ratios), len(width_ratios), figure=fig, width_ratios=width_ratios, height_ratios=height_ratios, wspace=0, hspace=0 )
    data_axes = np.array( [fig.add_subplot(gs[0]), fig.add_subplot(gs[2]), fig.add_subplot(gs[4]), fig.add_subplot(gs[6])] )
    cb_axis = fig.add_subplot( gs[8] )

    N_cols = 5
    N_rows = 6
    width = total_image_range[1] - total_image_range[0]
    height = total_image_range[3] - total_image_range[2]

    # Some extra utility methods to line up the images within the montage
    sub_image_ranges = [] # Units are m
    for i in range(N_cols):
        for j in range(N_rows):
            sub_image_ranges.append( [total_image_range[0] + i*width/N_cols, total_image_range[0] + (i+1)*width/N_cols, total_image_range[2] + j*height/N_rows, total_image_range[2] + (j+1)*height/N_rows] )

    sub_image_masks = [] # Units are pixels
    for i in range(N_cols):
        for j in range(N_rows):
            sub_image_masks.append( [i*image_dims[0]//N_cols, (i+1)*image_dims[0]//N_cols, j*image_dims[1]//N_rows, (j+1)*image_dims[1]//N_rows] )
    
    # For each point target, compute the entire image and mask out the particular region.
    # Not an especially efficient way to do this, but it works and looks good!
    for i, dataset in enumerate(datasets):
        envs = []
        masks = []

        # Want to put the same gain on each image, so compute the unscaled version first
        for k in range( len(dataset) ):
            data, acq_params = torch.tensor( dataset[k][0], dtype=s.PTFLOAT ), dataset[k][1]
            loc = torch.tensor( acq_params['locs'], dtype=s.PTFLOAT )
            delays = torch.tensor( acq_params['delays'], dtype=s.PTFLOAT )
            weights = torch.tensor( acq_params['weights'], dtype=s.PTFLOAT )

            # Find the sub image range that the loc is in
            idx = 0
            for idx in range(len(sub_image_ranges)):
                if loc[0,0]*1000 > sub_image_ranges[idx][0] and loc[0,0]*1000 < sub_image_ranges[idx][1] and \
                   loc[0,2]*1000 > sub_image_ranges[idx][2] and loc[0,2]*1000 < sub_image_ranges[idx][3]:
                    break

            H = ue.calc_H( acq_params['samples'], delays, weights )
            Hinv = ue.calc_Hinv_tikhonov(H, param=enc_params['tik_param'])

            # Decode the data
            rf_dec = ue.encode( data, Hinv )

            # Focus the data
            bf_delays = ue.calc_delays_beamforming( acq_params['rx_pos'], image_grid[0], image_grid[1] ).to( torch.device( "cuda:0" ) )
            iq_focused = ui.BeamformAD.apply(ue.hilbert( rf_dec ), acq_params['r0'], acq_params['dr'], 
                                            bf_delays, 64, image_dims[0], image_dims[1])

            env = torch.abs( iq_focused )
            envs.append( env ) 
            
            mask = torch.zeros_like( env )
            mask[sub_image_masks[idx][2]:sub_image_masks[idx][3], sub_image_masks[idx][0]:sub_image_masks[idx][1]] = 1
            masks.append( mask )

        # Get the scaling factor
        total_max = torch.amax( torch.stack( envs ) )

        # Now scale the images
        for (env, mask) in zip(envs, masks):
            image = torch.clamp( 20*torch.log10( env / torch.amax(total_max) ), bf_params['dB_min'], 0 )
            scaled_image = (image - image.min()) / (image.max() - image.min())

            rgba_image = plt.get_cmap('gray')(scaled_image)
            rgba_image[:, :, 3] = mask

            data_axes[i].imshow( rgba_image, extent=flipped_image_range )

    data_axes[0].set_yticks( data_axes[0].get_yticks() )
    data_axes[0].set_yticklabels([int(x) for x in data_axes[0].get_yticks()])
    data_axes[0].set_xticks( data_axes[0].get_xticks() )
    data_axes[0].set_xticklabels( [int(x) for x in data_axes[0].get_xticks()] )
    data_axes[0].set_xlabel("Lateral (mm)")
    data_axes[0].set_aspect('equal')
    data_axes[0].set_ylabel( "Axial (mm)" )
    data_axes[0].set_xlim( total_image_range[0], total_image_range[1] )
    data_axes[0].set_ylim( total_image_range[3], total_image_range[2] )

    # Plot narrow
    data_axes[1].set_yticks( data_axes[1].get_yticks() )
    data_axes[1].set_yticklabels([])
    data_axes[1].set_xticks( data_axes[1].get_xticks() )
    data_axes[1].set_xticklabels( [int(x) for x in data_axes[1].get_xticks()] )
    data_axes[1].set_aspect('equal')
    data_axes[1].set_xlabel("Lateral (mm)")
    # data_axes[0,1].set_ylabel( "Axial (mm)" )
    data_axes[1].set_xlim( total_image_range[0], total_image_range[1] )
    data_axes[1].set_ylim( total_image_range[3], total_image_range[2] )

    # Plot wide
    data_axes[2].set_yticks( data_axes[2].get_yticks() )
    data_axes[2].set_yticklabels([int(x) for x in data_axes[2].get_yticks()])
    data_axes[2].set_xticks( data_axes[2].get_xticks() )
    data_axes[2].set_xticklabels( [int(x) for x in data_axes[2].get_xticks()] )
    data_axes[2].set_aspect('equal')
    data_axes[2].set_xlabel("Lateral (mm)")
    # data_axes[2].set_ylabel( "Axial (mm)" )
    data_axes[2].set_xlim( total_image_range[0], total_image_range[1] )
    data_axes[2].set_ylim( total_image_range[3], total_image_range[2] )

    # Plot hadamard
    data_axes[3].set_yticks( data_axes[3].get_yticks() )
    data_axes[3].set_yticklabels([])
    data_axes[3].set_xticks( data_axes[3].get_xticks() )
    data_axes[3].set_xticklabels( [int(x) for x in data_axes[3].get_xticks()] )
    data_axes[3].set_aspect('equal')
    data_axes[3].set_xlabel("Lateral (mm)")
    # data_axes[1,1].set_ylabel( "Axial (mm)" )
    data_axes[3].set_xlim( total_image_range[0], total_image_range[1] )
    data_axes[3].set_ylim( total_image_range[3], total_image_range[2] )

    cmap = plt.get_cmap('gray')
    norm = matplotlib.colors.Normalize(vmin=bf_params['dB_min'], vmax=0)
    fig.colorbar( matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_axis )
    cb_axis.set_ylabel( "dB" )
            
    plt.rcParams['text.usetex'] = True
    data_axes[0].text( 0.5, 1.1, r"\underline{$15^\circ$ Span Planewaves}", transform=data_axes[0].transAxes, ha='center', va='center', fontsize=14 )
    data_axes[1].text( 0.5, 1.1, r"\underline{$60^\circ$ Span Planewaves}", transform=data_axes[1].transAxes, ha='center', va='center', fontsize=14 )
    data_axes[2].text( 0.5, 1.1, r"\underline{$150^\circ$ Span Planewaves}", transform=data_axes[2].transAxes, ha='center', va='center', fontsize=14 )
    data_axes[3].text( 0.5, 1.1, r"\underline{Optimized Encoding}", transform=data_axes[3].transAxes, ha='center', va='center', fontsize=14 )
    data_axes[0].text( -0.2, 0.5, r"\underline{Point Target Montage}", transform=data_axes[0].transAxes, rotation=90, ha='center', va='center', fontsize=14 )
    plt.rcParams['text.usetex'] = False

    # Plot lines between each of the sub_image_ranges
    for ax in data_axes:
        for i in range(1, N_cols):
            x_val = total_image_range[0] + i*width/N_cols 
            ax.plot( [x_val, x_val], [total_image_range[2], total_image_range[3]], color='gold', linewidth=1.5 )

        for j in range(1, N_rows):
            y_val = total_image_range[2] + j*height/N_rows 
            ax.plot( [total_image_range[0], total_image_range[1]], [y_val, y_val], color='gold', linewidth=1.5 )

    for ext in ['png', 'pdf']:
        fig.savefig( f"figures/figure8btop.{ext}", bbox_inches='tight' )

def plot_cystic_resolutions():
    opt_cystic_resolutions = np.zeros( (len( opt_dataset ), 3) )
    for i in range(len(opt_dataset)):    
        data, acq_params = opt_dataset[i]
        data = torch.tensor( data, dtype=s.PTFLOAT )
        loc = torch.tensor( acq_params['locs'], dtype=s.PTFLOAT )

        cr = ux.cystic_resolution_threshold_experimental( data, acq_params, threshold )

        opt_cystic_resolutions[i, 0] = loc[0,0]
        opt_cystic_resolutions[i, 1] = loc[0,2]
        opt_cystic_resolutions[i, 2] = cr

    narrow_cystic_resolutions = np.zeros( (len( narrow_dataset ), 3) )
    for i in range(len(narrow_dataset)):
        data, acq_params = narrow_dataset[i]
        data = torch.tensor( data, dtype=s.PTFLOAT )
        loc = torch.tensor( acq_params['locs'], dtype=s.PTFLOAT )

        cr = ux.cystic_resolution_threshold_experimental( data, acq_params, threshold )

        narrow_cystic_resolutions[i, 0] = loc[0,0]
        narrow_cystic_resolutions[i, 1] = loc[0,2]
        narrow_cystic_resolutions[i, 2] = cr

    middle_cystic_resolutions = np.zeros( (len( middle_dataset ), 3) )
    for i in range(len(middle_dataset)):
        data, acq_params = middle_dataset[i]
        data = torch.tensor( data, dtype=s.PTFLOAT )
        loc = torch.tensor( acq_params['locs'], dtype=s.PTFLOAT )

        cr = ux.cystic_resolution_threshold_experimental( data, acq_params, threshold )

        middle_cystic_resolutions[i, 0] = loc[0,0]
        middle_cystic_resolutions[i, 1] = loc[0,2]
        middle_cystic_resolutions[i, 2] = cr

    wide_cystic_resolutions = np.zeros( (len( wide_dataset ), 3) )
    for i in range(len(wide_dataset)):
        data, acq_params = wide_dataset[i]
        data = torch.tensor( data, dtype=s.PTFLOAT )
        loc = torch.tensor( acq_params['locs'], dtype=s.PTFLOAT )

        cr = ux.cystic_resolution_threshold_experimental( data, acq_params, threshold )

        wide_cystic_resolutions[i, 0] = loc[0,0]
        wide_cystic_resolutions[i, 1] = loc[0,2]
        wide_cystic_resolutions[i, 2] = cr
        
    # Get the grid that the targets are on
    unique_x = np.unique( opt_cystic_resolutions[:, 0] )
    unique_z = np.unique( opt_cystic_resolutions[:, 1] )
    x_targets, y_targets = np.meshgrid( unique_x, unique_z )
    narrow_z = np.zeros_like( x_targets )
    opt_z = np.zeros_like( x_targets )
    wide_z = np.zeros_like( x_targets )
    middle_z = np.zeros_like( x_targets )

    for i in range( len(opt_cystic_resolutions) ):
        x_i = np.where( unique_x == opt_cystic_resolutions[i, 0] )[0][0]
        z_i = np.where( unique_z == opt_cystic_resolutions[i, 1] )[0][0]
        narrow_z[z_i, x_i] = narrow_cystic_resolutions[i, 2]
        opt_z[z_i, x_i] = opt_cystic_resolutions[i, 2]
        wide_z[z_i, x_i] = wide_cystic_resolutions[i, 2]
        middle_z[z_i, x_i] = middle_cystic_resolutions[i, 2]

    # fig = plt.figure(figsize=(6, 8))
    fig = plt.figure(figsize=(15, 4.5))
    fig.subplots_adjust(wspace=0, hspace=0)

    plt.rcParams["figure.autolayout"] = True

    height_ratios = [1]
    width_ratios = [1, 0.1, 1, 0.1, 1, 0.1, 1, 0.1, 0.05, 0.3, 0.05]

    gs = GridSpec(len(height_ratios), len(width_ratios), figure=fig, width_ratios=width_ratios, height_ratios=height_ratios )
    data_axes = np.array( [fig.add_subplot(gs[0]), fig.add_subplot(gs[2]), fig.add_subplot(gs[4]), fig.add_subplot(gs[6])] )

    vmin = 0
    vmax = 5

    # Plot opt
    im = data_axes[0].scatter( narrow_cystic_resolutions[:, 0]*1000, narrow_cystic_resolutions[:, 1]*1000, s=200, c=narrow_cystic_resolutions[:, 2], cmap='viridis', vmin=vmin, vmax=vmax)
    data_axes[0].set_yticks( data_axes[0].get_yticks() )
    data_axes[0].set_yticklabels([int(x) for x in data_axes[0].get_yticks()])
    data_axes[0].set_xticks( data_axes[0].get_xticks() )
    data_axes[0].set_xticklabels( [int(x) for x in data_axes[0].get_xticks()] )
    data_axes[0].set_xlabel("Lateral (mm)")
    data_axes[0].set_aspect('equal')
    data_axes[0].set_ylabel( "Axial (mm)" )
    data_axes[0].set_xlim( total_image_range[0], total_image_range[1] )
    data_axes[0].set_ylim( total_image_range[3], total_image_range[2] )

    # Plot narrow
    im = data_axes[1].scatter( middle_cystic_resolutions[:, 0]*1000, middle_cystic_resolutions[:, 1]*1000, s=200, c=middle_cystic_resolutions[:, 2], cmap='viridis', vmin=vmin, vmax=vmax)
    data_axes[1].set_yticks( data_axes[1].get_yticks() )
    data_axes[1].set_yticklabels([])
    data_axes[1].set_xticks( data_axes[1].get_xticks() )
    data_axes[1].set_xticklabels( [int(x) for x in data_axes[1].get_xticks()] )
    data_axes[1].set_aspect('equal')
    data_axes[1].set_xlabel("Lateral (mm)")
    # data_axes[1].set_ylabel( "Axial (mm)" )
    data_axes[1].set_xlim( total_image_range[0], total_image_range[1] )
    data_axes[1].set_ylim( total_image_range[3], total_image_range[2] )

    # Plot wide
    im = data_axes[2].scatter( wide_cystic_resolutions[:, 0]*1000, wide_cystic_resolutions[:, 1]*1000, s=200, c=wide_cystic_resolutions[:, 2], cmap='viridis', vmin=vmin, vmax=vmax)
    data_axes[2].set_yticks( data_axes[2].get_yticks() )
    data_axes[2].set_yticklabels([])
    data_axes[2].set_xticks( data_axes[2].get_xticks() )
    data_axes[2].set_xticklabels( [int(x) for x in data_axes[2].get_xticks()] )
    data_axes[2].set_aspect('equal')
    data_axes[2].set_xlabel("Lateral (mm)")
    # data_axes[2].set_ylabel( "Axial (mm)" )
    data_axes[2].set_xlim( total_image_range[0], total_image_range[1] )
    data_axes[2].set_ylim( total_image_range[3], total_image_range[2] )
    # label_axes[1,0].set_axis_off()

    # Plot hadamard
    im = data_axes[3].scatter( opt_cystic_resolutions[:, 0]*1000, opt_cystic_resolutions[:, 1]*1000, s=200, c=opt_cystic_resolutions[:, 2], cmap='viridis', vmin=vmin, vmax=vmax)
    data_axes[3].set_yticks( data_axes[3].get_yticks() )
    data_axes[3].set_yticklabels([])
    data_axes[3].set_xticks( data_axes[3].get_xticks() )
    data_axes[3].set_xticklabels( [int(x) for x in data_axes[3].get_xticks()] )
    data_axes[3].set_aspect('equal')
    data_axes[3].set_xlabel("Lateral (mm)")
    # data_axes[1,1].set_ylabel( "Axial (mm)" )
    data_axes[3].set_xlim( total_image_range[0], total_image_range[1] )
    data_axes[3].set_ylim( total_image_range[3], total_image_range[2] )

    ay_cb = fig.add_subplot( gs[8] )
    fig.colorbar( im, cax=ay_cb, extend='max' )
    ay_cb.set_yticks( ay_cb.get_yticks() )
    ay_cb.set_yticklabels( [int(x) for x in ay_cb.get_yticks()] )
    ay_cb.set_ylabel( "Cystic Resolution (mm)" )
    
    plt.rcParams['text.usetex'] = True
    data_axes[0].text( 0.5, 1.1, r"\underline{$15^\circ$ Span Planewaves}", transform=data_axes[0].transAxes, ha='center', va='center', fontsize=14 )
    data_axes[1].text( 0.5, 1.1, r"\underline{$60^\circ$ Span Planewaves}", transform=data_axes[1].transAxes, ha='center', va='center', fontsize=14 )
    data_axes[2].text( 0.5, 1.1, r"\underline{$150^\circ$ Span Planewaves}", transform=data_axes[2].transAxes, ha='center', va='center', fontsize=14 )
    data_axes[3].text( 0.5, 1.1, r"\underline{Optimized Encoding}", transform=data_axes[3].transAxes, ha='center', va='center', fontsize=14 )
    data_axes[0].text( -0.2, 0.5, r"\underline{Detectability Map}", transform=data_axes[0].transAxes, rotation=90, ha='center', va='center', fontsize=14 )
    plt.rcParams['text.usetex'] = False

    for ext in ['png', 'pdf']:
        fig.savefig( f"figures/figure8bbottom.{ext}", bbox_inches='tight' )

if __name__ == "__main__":
    # plot_montage()
    plot_cystic_resolutions()