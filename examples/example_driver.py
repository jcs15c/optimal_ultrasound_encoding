import project_root
import src.ultrasound_encoding as ue
import src.ultrasound_imaging as ui
import src.ultrasound_utilities as uu
import src.sequence_constraints as sc

import torch

from encoding_optimization import optimize_encoding_sequence

if __name__ == "__main__":
    results_folder = "results"
    data_folder = "data/underdeveloped_speckle_data"
    
    num_beams = 31
    init_delays = ue.calc_delays_planewaves(num_beams, span=90)
    init_weights = ue.calc_uniform_weights(num_beams)
    
    opt_params = {'num_epochs' : 10}
    bf_params = {'image_dims': [100, 100]}

    optimize_encoding_sequence( results_folder, data_folder, init_delays, init_weights, \
                                opt_params=opt_params, bf_params=bf_params )
