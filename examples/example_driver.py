import torch
import src.ultrasound_encoding as ue
import src.ultrasound_beamforming as ub
import src.ultrasound_utilities as uu
import src.weight_constraints as wc

from encoding_optimization import optimal_image_encoding

if __name__ == "__main__":
    
    test_folder = "results"
    training_set = data/training_data"
    testing_set = data/testing_data"
    
    num_beams = 31
    init_delays = ue.calc_delays_planewaves(num_beams, span=90)
    init_weights = ue.calc_uniform_weights(num_beams)

    opt_params = "both"
    loss_func = "L2"
    WeightProj = wc.LinfBallProjection()
    target_type = "unencoded"

    training_resolution = "half"
    num_epochs = 35
    desc_alg = "Adam"
    lrate = 0.1
    training_shuffle = False
    
    image_dims = [300, 500]
    dB_min = -60
    
    noise_param = None
    tik_param = 0.1
    
    optimal_image_encoding(test_folder, training_set, testing_set, # Folder names containing data and results
                           init_delays, init_weights,  # optimization initial parameters
                           opt_params = opt_params, loss_func=loss_func, WeightProj=WeightProj, training_resolution=training_resolution,  target_type=target_type, # optimization parameters
                           num_epochs=num_epochs, desc_alg=desc_alg, lrate=lrate, training_shuffle = training_shuffle, # descent algorithm hyperparameters
                           image_dims=image_dims, dB_min=dB_min,   # image parameters
                           noise_param=noise_param, tik_param=tik_param, # Encoding parameters
                           notes="Focused beams and uniform weigths.")
