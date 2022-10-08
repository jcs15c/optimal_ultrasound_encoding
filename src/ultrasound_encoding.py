import numpy as np
import torch

import scipy 
import scipy.io
import scipy.signal

import src.settings as s

from src.predictor_model import PredictorModel
from src.ultrasound_utilities import UltrasoundDataset

def get_encoded_env( delays, weights, data_folder, idx,
                     image_dims=[300, 500], image_range=[-25, 25, 15, 55], dB_min=-60, 
                     noise_param=[0.7, 12], tik_param=0.1, hist_param=[100,3]):
    """
    Compute encoded complex envelope for a given encoding sequence and RF data

    Parameters:
        delays/weights - Encoding sequence
        data_folder, idx - Location and index of RF/location data
        image_dims - Pixel resolution in [lateral, axial] direction
        image_range - maging area [x_min, x_max, z_min, z_max], in mm
        dB_min - Minimum of the dynamic range
        noise_param - [Bandwidth, SNR] parameters for adding encoding noise
        tik_param - Tikhonov regularization parameter for decoding
        hist_param - [bins, smoothing parameter] for smooth histogram calculations
        
    Returns:
        encoded_env - Encoded complex envelope for imaging
        unencoded_env - Unencoded complex envelope for imaging
    """
    # Define new set of testing data
    image_data = torch.tensor( UltrasoundDataset(data_folder)[idx][0] ).unsqueeze(0)
    image_loc = torch.tensor( UltrasoundDataset(data_folder)[idx][1] ).unsqueeze(0)
    
    acq_params = scipy.io.loadmat(f"{data_folder}/acq_params.mat")
    
    # Check that the acquisition parameters for the data is consistent
    assert acq_params['fs'].item() == s.fs
    assert acq_params['f0'].item() == s.f0
    assert acq_params['c'].item() == s.c
    assert np.allclose( acq_params['rx_pos'], s.rx_pos.numpy() )
    assert np.allclose( acq_params['tx_pos'], s.tx_pos.numpy() )
    
    num_elem = s.tx_pos.shape[0]
    hist_data = [-11.811993598937988, 5.649099826812744]
         
    x = torch.linspace(image_range[0], image_range[1], steps=int(image_dims[0]) )/1000
    z = torch.linspace(image_range[2], image_range[3], steps=int(image_dims[1]) )/1000
    
    # Set up encoding model
    predictor_model = PredictorModel(delays, weights, acq_params, [x, z], tik_param=tik_param, dB_min=dB_min, noise_param=noise_param, hist_data=hist_data )
    predictor_model.delays.requires_grad = False
    predictor_model.weights.requires_grad = False
    
    encoded_env = predictor_model.get_image_prediction( image_data, image_loc )[0]
    unencoded_env = predictor_model.get_targets( image_data, image_loc, "unencoded" )[0]
    
    return encoded_env, unencoded_env


def encode(rf, H):
    """
    Encode (or decode) RF channel data using the provided linear encoding matrix

    Parameters:
        rf - Multistatic RF data (time sample x receive channel x transmit)
        H - Linear encoding matrices (frequency x transmit x encoding)

    Returns:
        rf_enc - Encoded transmit data set (time sample x receive channel x encoding)
    """
    RF = torch.fft.rfft(rf, axis=0 )    
    RF_ENC = torch.matmul( RF, H )
    return torch.fft.irfft(RF_ENC, rf.shape[0], axis=0)

def calc_H(samples, delays, weights):
    """
    Compute linear encoding matrix H

    Parameters:
        samples - Number of axial samples
        delays - Transmit focal delays in samples (transmit event x
                                                        transmit element)
        weights - Transmit amplitude weights (transmit_event x 
                                                   transmit_element)

    Returns:
        H : Linear encoding matries (frequency x transmit_element x transmit_event)
    """
    return weights.T.view(1, weights.shape[1], weights.shape[0]) * \
            torch.exp( -2j*np.pi*torch.arange(samples//2 + 1).view(-1, 1, 1)/samples*delays.T )

def calc_Hinv_adjoint(H):
    """
    Approximate an inverse using the adjoint
    
    Parameters:
        H - Linear encoding matrices (frequency x transmit_element x transmit_event)
    
    Returns:
        Hinv - Adjoint at each frequency (frequency x transmit_event x transmit_element)
    """
    return torch.transpose(H.conj(),1,2)

def calc_Hinv_tikhonov(H,param=0.1):
    """
    Compute a pseudoinverse with tikhonov regularization
    
    Parameters:
        H - Linear encoding matrices (frequency x transmit_element x transmit_event)
    
    Returns:
        Hinv - Psuedoinverse at each frequency (frequency x transmit_event x transmit_element)
    """    
    if param == np.inf:
        return torch.transpose( H.conj(), 1, 2 ) / H.shape[1]
    
    smax = torch.linalg.norm( H, dim=(1,2), ord=2 ).view(-1, 1, 1)
    reg = param**2 * smax**2 * torch.eye(H.shape[2]).reshape(-1, H.shape[2], H.shape[2]).repeat(H.shape[0], 1, 1)
    return torch.linalg.solve(torch.matmul(torch.transpose(H.conj(),1,2), H) + reg, torch.transpose(H.conj(),1,2) )

def hilbert( x ):
    """
    Compute the analytical signal of a 3D tensor along the 0th axis
    """
    n = x.shape[0]
    u = torch.zeros( [n, 1, 1] )
    u[0,:,:] = 0.5
    u[1:n//2+1,:,:] = 1
      
    return torch.fft.ifft( torch.fft.fft( x, dim=0 ) * 2 * u, dim=0 )

def generate_encoded_noise( rf_ref, acq_params, BW, SNR, sz ):
    """
    Generate an array of bandlimited noise. Add to encoded data.
    
    Parameters:
        rf_ref - Reference RF data to normalize the encoding noise
        acq_params - Parameters for RF data
        BW - Bandwidth for the noise to be added
        SNR - Signal to Noise Ratio of noise to be added
        sz - Size of the tensor of RF data
        
    Returns:
        Additive noise in shape of sz
    """
    noise = np.random.normal( 0, 1, size=sz )
    
    # Define critical frequencies as a fraction of Nyquist frequency
    max_f = acq_params['f0'].item()*(1 + BW) / (acq_params['fs'].item() / 2)   
    min_f = acq_params['f0'].item()*(1 - BW) / (acq_params['fs'].item() / 2)

    # Create filter (with order 5) and apply it to each sequence of data
    numerator, denominator = scipy.signal.butter( 5, [min_f, max_f], btype='bandpass' )
    noise = torch.tensor( scipy.signal.filtfilt( numerator, denominator, noise, axis=1 ).copy() )

    noise /= torch.std( noise, dim=(1,2,3) ).view(-1, 1, 1, 1)
    
    # Apply a signal to noise ratio of 12 decibals
    noise *= torch.std( rf_ref, dim=(1,2) ).view(-1, 1, 1, 1) * 10**(-SNR/20)
    
    return noise.float()

def calc_delays_phasedfocused(beams,span,focus,apex=0):
    """
    Compute delays for a phased, focused transmission
    
    Parameters:
        span - Angle span (degrees)
        beams - Number of beams to transmit
        focus - Focal radius from transducer (m)
        apex - Distance behind the array to place origin of beams (m)
        tx_pos - Array element positions (element x [x,y,z]) (m)
        c - Speed of sound (m/s)
        
    Returns:
        Encoding delays
    """
    delays = torch.empty([beams,s.tx_pos.shape[0]], dtype=s.PTFLOAT)
    angles = torch.deg2rad(span * torch.linspace(-1.0, 1.0, steps=beams) / 2)
    for (i,angle) in enumerate(angles):
        foc=(torch.tan(angle) * apex + torch.sin(angle) * focus, apex + torch.cos(angle) * focus)
        r=torch.sqrt((s.tx_pos[:,0]-foc[0])**2 + (s.tx_pos[:,2]-foc[1])**2);
        r0=torch.sqrt(foc[0]**2 + foc[1]**2);
        delays[i,:]=(r0-r)/s.c; 
    return s.fs * delays

def calc_delays_planewaves(beams, span):
    """
    Compute delays for steered planewave transmission
    
    Parameters:
        span - Angle span (degrees)
        beams - Number of beams to transmit
        
    Returns:
        Encoding delays
    """
    delays = torch.empty([beams,s.tx_pos.shape[0]], dtype=s.PTFLOAT)
    angles = torch.deg2rad(span * torch.linspace(-1.0, 1.0, steps=beams) / 2)
    for (i,angle) in enumerate(angles):
        delays[i,:] = s.tx_pos[:,0] * torch.sin(angle) / s.c
    return s.fs * delays

def calc_delays_beamforming( rx_pos, x, z ):
    """
    Compute delays for each pixel in an image
    
    Parameters:
        rx_pos - Array element positions (element x [x,y,z]) (m)
        x, z - The x- and z- coordinates for each pixel
        
    Returns:
        Encoding delays
    """
    zg, xg = torch.meshgrid( z, x )
    xg = xg.flatten()
    zg = zg.flatten()
    
    bf_delays = torch.empty([len(xg), rx_pos.shape[0]], dtype=s.PTFLOAT)
    for i, pos in enumerate( rx_pos ):
        bf_delays[:, i] = torch.sqrt(torch.square(xg - pos[0]) +
                                                       pos[1]**2 +
                                     torch.square(zg - pos[2]))
        
    return bf_delays.T

def calc_uniform_weights( beams, value=1.0 ):
    """
    Compute uniform encoding weights with a given value
    
    Parameters:
        beams - The number of transmissions to generate weights for
        value - The uniform value given to each weight
        
    Returns:
        Encoding weights
    """
    return value * torch.ones( [beams, s.tx_pos.shape[0]] )
    
def calc_hadamard_weights( beams ):
    """
    Compute truncated hadamard encoding weights
    
    Parameters:
        beams - The number of transmissions to generate weights for
        
    Returns:
        Encoding weights
    """
    return torch.tensor( scipy.linalg.hadamard( s.tx_pos.shape[0] ) )[:beams].float()