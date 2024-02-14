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

if __name__ == "__main__":
    ## Set up model parameters
    enc_params = s.default_enc_params
    bf_params = s.default_bf_params

    def get_models( acq_params, image_range, hist_match ):
        bf_params['image_range'] = image_range
        bf_params['hist_match'] = hist_match

        # Delay Optimized
        delay_opt_delays = torch.tensor( np.loadtxt( f"optimized_sequences/delay_parameterization/delays.csv", delimiter="," ), dtype=s.PTFLOAT  )
        delay_opt_weights = torch.tensor( np.loadtxt( f"optimized_sequences/delay_parameterization/weights.csv", delimiter="," ), dtype=s.PTFLOAT  )
        delay_opt_model = PredictorModel(delay_opt_delays, delay_opt_weights, acq_params, enc_params, bf_params )

        # Weight Optimized
        weight_opt_delays = torch.tensor( np.loadtxt( f"optimized_sequences/weight_parameterization/delays.csv", delimiter="," ), dtype=s.PTFLOAT  )
        weight_opt_weights = torch.tensor( np.loadtxt( f"optimized_sequences/weight_parameterization/weights.csv", delimiter="," ), dtype=s.PTFLOAT  )
        weight_opt_model = PredictorModel(weight_opt_delays, weight_opt_weights, acq_params, enc_params, bf_params )

        for model in [delay_opt_model, weight_opt_model]:
            model.delays.requires_grad = False
            model.weights.requires_grad = False

        return delay_opt_model, weight_opt_model

    isolated_point_dataset = uu.UltrasoundDataset( "./data/isolated_point_data" )
    isolated_point_acq_params = scipy.io.loadmat( f"./data/isolated_point_data/acq_params.mat" )

    underdeveloped_speckle_dataset = uu.UltrasoundDataset( "./data/underdeveloped_speckle_data" )
    underdeveloped_speckle_acq_params = scipy.io.loadmat( f"./data/underdeveloped_speckle_data/acq_params.mat" )

    anechoic_lesion_dataset = uu.UltrasoundDataset( f"./data/anechoic_lesion_data" )
    anechoic_lesion_acq_params = scipy.io.loadmat( f"./data/anechoic_lesion_data/acq_params.mat" )

    image_derived_dataset = uu.UltrasoundImageDataset( f"./data/image_derived_data" )
    image_derived_acq_params = scipy.io.loadmat( f"./data/image_derived_data/acq_params.mat" )

    binary_image_dataset = uu.UltrasoundImageDataset( f"./data/binary_image_data" )
    binary_image_acq_params = scipy.io.loadmat( f"./data/binary_image_data/acq_params.mat" )

    N_data = 50

    ## Store results for lesions
    # isolated_point_delay_opt = np.zeros( min( len( isolated_point_dataset ), N_data ) )
    # isolated_point_weight_opt = np.zeros( min( len( isolated_point_dataset ), N_data ) )

    # for i in range(min(N_data, len(isolated_point_dataset))):
    #     delay_opt_model, weight_opt_model = get_models( isolated_point_acq_params, [-25, 25, 15, 55], False )

    #     datas, locs = [torch.tensor( x ).unsqueeze(0) for x in isolated_point_dataset[i]]

    #     target = delay_opt_model.get_targets( datas, locs, 'synthetic' )

    #     isolated_point_delay_opt[i] = torch.mean( torch.square( delay_opt_model.get_image_prediction( datas, locs ) - target ) )
    #     isolated_point_weight_opt[i] = torch.mean( torch.square( weight_opt_model.get_image_prediction( datas, locs ) - target ) )

    # underdeveloped_speckle_delay_opt = np.zeros( min( len( underdeveloped_speckle_dataset ), N_data ) )
    # underdeveloped_speckle_weight_opt = np.zeros( min( len( underdeveloped_speckle_dataset ), N_data ) )

    # for i in range(min(N_data, len(underdeveloped_speckle_dataset))):
    #     delay_opt_model, weight_opt_model = get_models( underdeveloped_speckle_acq_params, [-25, 25, 15, 55], False )

    #     datas, locs = [torch.tensor( x ).unsqueeze(0) for x in underdeveloped_speckle_dataset[i]]

    #     target = delay_opt_model.get_targets( datas, locs, 'synthetic' )

    #     underdeveloped_speckle_delay_opt[i] = torch.mean( torch.square( delay_opt_model.get_image_prediction( datas, locs ) - target ) )
    #     underdeveloped_speckle_weight_opt[i] = torch.mean( torch.square( weight_opt_model.get_image_prediction( datas, locs ) - target ) )

    # anechoic_lesion_delay_opt = np.zeros( min( len( anechoic_lesion_dataset ), N_data ) )
    # anechoic_lesion_weight_opt = np.zeros( min( len( anechoic_lesion_dataset ), N_data ) )

    # for i in range(min(N_data, len(anechoic_lesion_dataset))):
    #     delay_opt_model, weight_opt_model = get_models( anechoic_lesion_acq_params,[-25, 25, 15, 55], False )
        
    #     datas, locs = [torch.tensor( x ).unsqueeze(0) for x in anechoic_lesion_dataset[i]]

    #     target = delay_opt_model.get_targets( datas, locs, 'synthetic' )

    #     anechoic_lesion_delay_opt[i] = torch.mean( torch.square( delay_opt_model.get_image_prediction( datas, locs ) - target ) )
    #     anechoic_lesion_weight_opt[i] = torch.mean( torch.square( weight_opt_model.get_image_prediction( datas, locs ) - target ) )

    image_derived_delay_opt = np.zeros( min( len( image_derived_dataset ), N_data ) )
    image_derived_weight_opt = np.zeros( min( len( image_derived_dataset ), N_data ) )
    
    for i in range(min(N_data, len(image_derived_dataset))):
        # print("Image Derived: ", i, "of", min(N_data, len(image_derived_dataset)))
        delay_opt_model, weight_opt_model = get_models( image_derived_acq_params, [-20, 20, 22, 52], False )

        datas, cmap = [torch.tensor( x ).unsqueeze(0) for x in image_derived_dataset[i]]

        target = delay_opt_model.get_targets( datas, cmap, 'image_contrast' )

        image_derived_delay_opt[i] = torch.mean( torch.square( delay_opt_model.get_image_prediction( datas, cmap ) - target ) )
        image_derived_weight_opt[i] = torch.mean( torch.square( weight_opt_model.get_image_prediction( datas, cmap ) - target ) )
        print( i, image_derived_delay_opt[i] )

    # binary_image_delay_opt = np.zeros( min( len( binary_image_dataset ), N_data ) )
    # binary_image_weight_opt = np.zeros( min( len( binary_image_dataset ), N_data ) )

    # for i in range(min(N_data, len(binary_image_dataset))):
    #     # print("Binary Image: ", i, "of", min(N_data, len(binary_image_dataset)))
    #     delay_opt_model, weight_opt_model = get_models( binary_image_acq_params, [-20, 20, 22, 52], False )

    #     datas, cmap = [torch.tensor( x ).unsqueeze(0) for x in binary_image_dataset[i]]

    #     target = delay_opt_model.get_targets( datas, cmap, 'image_synthetic' )

    #     binary_image_delay_opt[i] = torch.mean( torch.square( delay_opt_model.get_image_prediction( datas, cmap ) - target ) )
    #     binary_image_weight_opt[i] = torch.mean( torch.square( weight_opt_model.get_image_prediction( datas, cmap ) - target ) )

    print( "Delay Only, Weight Only" )
    # print( "Isolated Points: ",     np.mean( isolated_point_delay_opt ),         np.mean( isolated_point_weight_opt ) )
    # print( "Low Density Speckle: ", np.mean( underdeveloped_speckle_delay_opt ), np.mean( underdeveloped_speckle_weight_opt ) )
    # print( "Anechoic Lesions: ",    np.mean( anechoic_lesion_delay_opt ),        np.mean( anechoic_lesion_weight_opt ) )
    print( "Image Derived: ",       np.mean( image_derived_delay_opt ),          np.mean( image_derived_weight_opt ) )
    # print( "Binary Image: ",        np.mean( binary_image_delay_opt ),           np.mean( binary_image_weight_opt ) )
