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
bf_params['image_range'] = [-20, 20, 22, 52]
bf_params['hist_match'] = True

def get_models( acq_params ):
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

binary_dataset = uu.UltrasoundImageDataset( "./data/binary_image_data" )
binary_acq_params = scipy.io.loadmat( f"./data/binary_image_data/acq_params.mat" )

N_data = 50

binary_opt_gCNR = np.zeros( min( len( binary_dataset ), N_data ) )
binary_narrow_gCNR = np.zeros( min( len( binary_dataset ), N_data ) )
binary_wide_gCNR = np.zeros( min( len( binary_dataset ), N_data ) )
binary_hadamard_gCNR = np.zeros( min( len( binary_dataset ), N_data ) )

for i in range(min(N_data, len(binary_dataset))):
    print( f"Binary {i}" )
    narrow_model, wide_model, opt_model, hadamard_model = get_models( binary_acq_params )

    datas, cmaps = [torch.tensor( x ).unsqueeze(0) for x in binary_dataset[i]]

    binary_opt_gCNR[i] = ui.binary_image_gCNR( opt_model.get_image_prediction( datas, cmaps )[0], cmaps[0], bf_params )
    binary_narrow_gCNR[i] = ui.binary_image_gCNR( narrow_model.get_image_prediction( datas, cmaps )[0], cmaps[0], bf_params )
    binary_wide_gCNR[i] = ui.binary_image_gCNR( wide_model.get_image_prediction( datas, cmaps )[0], cmaps[0], bf_params )
    binary_hadamard_gCNR[i] = ui.binary_image_gCNR( hadamard_model.get_image_prediction( datas, cmaps )[0], cmaps[0], bf_params )

print( "Optimized, Narrow, Wide, Hadamard" )
print( "Binary Image: ", np.mean( binary_opt_gCNR ), np.mean( binary_narrow_gCNR ), np.mean( binary_wide_gCNR ), np.mean( binary_hadamard_gCNR ) )
