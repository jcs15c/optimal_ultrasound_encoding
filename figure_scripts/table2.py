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

        # Optimized
        opt_delays = torch.tensor( np.loadtxt( f"optimized_sequences/full_parameterization/delays.csv", delimiter="," ), dtype=s.PTFLOAT  )
        opt_weights = torch.tensor( np.loadtxt( f"optimized_sequences/full_parameterization/weights.csv", delimiter="," ), dtype=s.PTFLOAT  )
        opt_model = PredictorModel(opt_delays, opt_weights, acq_params, enc_params, bf_params )

        # Narrow FOV
        narrow_delays = ue.calc_delays_planewaves( opt_delays.shape[0], spacing=1 )
        narrow_weights = ue.calc_uniform_weights( narrow_delays.shape[0] )
        narrow_model = PredictorModel(narrow_delays, narrow_weights, acq_params, enc_params, bf_params)

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

        return narrow_model, wide_model, opt_model, hadamard_model

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

    # Store results for lesions
    isolated_point_opt = np.zeros( min( len( isolated_point_dataset ), N_data ) )
    isolated_point_narrow = np.zeros( min( len( isolated_point_dataset ), N_data ) )
    isolated_point_wide = np.zeros( min( len( isolated_point_dataset ), N_data ) )
    isolated_point_hadamard = np.zeros( min( len( isolated_point_dataset ), N_data ) )

    for i in range(min(N_data, len(isolated_point_dataset))):
        narrow_model, wide_model, opt_model, hadamard_model = get_models( isolated_point_acq_params, [-25, 25, 15, 55], False )

        datas, locs = [torch.tensor( x ).unsqueeze(0) for x in isolated_point_dataset[i]]

        target = opt_model.get_targets( datas, locs, 'synthetic' )

        isolated_point_opt[i] = torch.mean( torch.square( opt_model.get_image_prediction( datas, locs ) - target ) )
        isolated_point_narrow[i] = torch.mean( torch.square( narrow_model.get_image_prediction( datas, locs ) - target ) )
        isolated_point_wide[i] = torch.mean( torch.square( wide_model.get_image_prediction( datas, locs ) - target ) )
        isolated_point_hadamard[i] = torch.mean( torch.square( hadamard_model.get_image_prediction( datas, locs ) - target ) )

    underdeveloped_speckle_opt = np.zeros( min( len( underdeveloped_speckle_dataset ), N_data ) )
    underdeveloped_speckle_narrow = np.zeros( min( len( underdeveloped_speckle_dataset ), N_data ) )
    underdeveloped_speckle_wide = np.zeros( min( len( underdeveloped_speckle_dataset ), N_data ) )
    underdeveloped_speckle_hadamard = np.zeros( min( len( underdeveloped_speckle_dataset ), N_data ) )

    for i in range(min(N_data, len(underdeveloped_speckle_dataset))):
        narrow_model, wide_model, opt_model, hadamard_model = get_models( underdeveloped_speckle_acq_params, [-25, 25, 15, 55], False )

        datas, locs = [torch.tensor( x ).unsqueeze(0) for x in underdeveloped_speckle_dataset[i]]

        target = opt_model.get_targets( datas, locs, 'synthetic' )

        underdeveloped_speckle_opt[i] = torch.mean( torch.square( opt_model.get_image_prediction( datas, locs ) - target ) )
        underdeveloped_speckle_narrow[i] = torch.mean( torch.square( narrow_model.get_image_prediction( datas, locs ) - target ) )
        underdeveloped_speckle_wide[i] = torch.mean( torch.square( wide_model.get_image_prediction( datas, locs ) - target ) )
        underdeveloped_speckle_hadamard[i] = torch.mean( torch.square( hadamard_model.get_image_prediction( datas, locs ) - target ) )

    anechoic_lesion_opt = np.zeros( min( len( anechoic_lesion_dataset ), N_data ) )
    anechoic_lesion_narrow = np.zeros( min( len( anechoic_lesion_dataset ), N_data ) )
    anechoic_lesion_wide = np.zeros( min( len( anechoic_lesion_dataset ), N_data ) )
    anechoic_lesion_hadamard = np.zeros( min( len( anechoic_lesion_dataset ), N_data ) )

    for i in range(min(N_data, len(anechoic_lesion_dataset))):
        narrow_model, wide_model, opt_model, hadamard_model = get_models( anechoic_lesion_acq_params,[-25, 25, 15, 55], False )
        
        datas, locs = [torch.tensor( x ).unsqueeze(0) for x in anechoic_lesion_dataset[i]]

        target = opt_model.get_targets( datas, locs, 'synthetic' )

        anechoic_lesion_opt[i] = torch.mean( torch.square( opt_model.get_image_prediction( datas, locs ) - target ) )
        anechoic_lesion_narrow[i] = torch.mean( torch.square( narrow_model.get_image_prediction( datas, locs ) - target ) )
        anechoic_lesion_wide[i] = torch.mean( torch.square( wide_model.get_image_prediction( datas, locs ) - target ) )
        anechoic_lesion_hadamard[i] = torch.mean( torch.square( hadamard_model.get_image_prediction( datas, locs ) - target ) )

    image_derived_opt = np.zeros( min( len( image_derived_dataset ), N_data ) )
    image_derived_narrow = np.zeros( min( len( image_derived_dataset ), N_data ) )
    image_derived_wide = np.zeros( min( len( image_derived_dataset ), N_data ) )
    image_derived_hadamard = np.zeros( min( len( image_derived_dataset ), N_data ) )

    for i in range(min(N_data, len(image_derived_dataset))):
        print("Image Derived: ", i, "of", min(N_data, len(image_derived_dataset)))
        narrow_model, wide_model, opt_model, hadamard_model = get_models( image_derived_acq_params, [-20, 20, 22, 52], False )

        datas, cmap = [torch.tensor( x ).unsqueeze(0) for x in image_derived_dataset[i]]

        target = opt_model.get_targets( datas, cmap, 'image_contrast' )

        image_derived_opt[i] = torch.mean( torch.square( opt_model.get_image_prediction( datas, cmap ) - target ) )
        image_derived_narrow[i] = torch.mean( torch.square( narrow_model.get_image_prediction( datas, cmap ) - target ) )
        image_derived_wide[i] = torch.mean( torch.square( wide_model.get_image_prediction( datas, cmap ) - target ) )
        image_derived_hadamard[i] = torch.mean( torch.square( hadamard_model.get_image_prediction( datas, cmap ) - target ) )
    
    binary_image_opt = np.zeros( min( len( binary_image_dataset ), N_data ) )
    binary_image_narrow = np.zeros( min( len( binary_image_dataset ), N_data ) )
    binary_image_wide = np.zeros( min( len( binary_image_dataset ), N_data ) )
    binary_image_hadamard = np.zeros( min( len( binary_image_dataset ), N_data ) )

    for i in range(min(N_data, len(binary_image_dataset))):
        print("Binary Image: ", i, "of", min(N_data, len(binary_image_dataset)))
        narrow_model, wide_model, opt_model, hadamard_model = get_models( binary_image_acq_params, [-20, 20, 22, 52], False )

        datas, cmap = [torch.tensor( x ).unsqueeze(0) for x in binary_image_dataset[i]]

        target = opt_model.get_targets( datas, cmap, 'image_synthetic' )

        binary_image_opt[i] = torch.mean( torch.square( opt_model.get_image_prediction( datas, cmap ) - target ) )
        binary_image_narrow[i] = torch.mean( torch.square( narrow_model.get_image_prediction( datas, cmap ) - target ) )
        binary_image_wide[i] = torch.mean( torch.square( wide_model.get_image_prediction( datas, cmap ) - target ) )
        binary_image_hadamard[i] = torch.mean( torch.square( hadamard_model.get_image_prediction( datas, cmap ) - target ) )

    print( "Optimized, Narrow, Wide, Hadamard" )
    print( "Isolated Points: ", np.mean( isolated_point_opt ), np.mean( isolated_point_narrow ), np.mean( isolated_point_wide ), np.mean( isolated_point_hadamard ) )
    print( "Low Density Speckle: ", np.mean( underdeveloped_speckle_opt ), np.mean( underdeveloped_speckle_narrow ), np.mean( underdeveloped_speckle_wide ), np.mean( underdeveloped_speckle_hadamard ) )
    print( "Anechoic Lesions: ", np.mean( anechoic_lesion_opt ), np.mean( anechoic_lesion_narrow ), np.mean( anechoic_lesion_wide ), np.mean( anechoic_lesion_hadamard ) )
    print( "Image Derived: ", np.mean( image_derived_opt ), np.mean( image_derived_narrow ), np.mean( image_derived_wide ), np.mean( image_derived_hadamard ) )
    print( "Binary Image: ", np.mean( binary_image_opt ), np.mean( binary_image_narrow ), np.mean( binary_image_wide ), np.mean( binary_image_hadamard ) )
