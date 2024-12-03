import numpy as np
import torch

import scipy
import scipy.io

import src.ultrasound_utilities as uu
import src.ultrasound_encoding as ue
import src.ultrasound_imaging as ui
import src.ultrasound_optimization as uo

from src.predictor_model import PredictorModel
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import src.settings as s

def optimize_encoding_sequence( results_folder, data_folder, # Folder names containing data and results
                                init_delays, init_weights, # Initial Sequence
                                opt_params = None,
                                enc_params = None,
                                bf_params = None,
                                save_training_images = False, save_training_loss = False,
                                notes = None ):
    """
    Takes a number of optimization, encoding, and imaging parameters to find
    the encoding sequence that minimize given the objective function

    Meta-Parameters:
        results_folder - Path to folder in which optimization results are saved
        data_folder - Path to folder containing RF data for training/testing
        init_delays/weights - Initial encoding sequence for optimization
        save_training_images/loss - If true, store values for training
        notes - String saved alongside results in record file
    Optimization Parameters: dict containing the following fields and defaults
        trainable_params - One of ['delays', 'weights', 'both']
        training_resolution - Fraction of image dimensions used during training, or 'graduated'
        training_shuffle - Shuffle data during training phase
        LossFunc - Instance of class from ultrasound_optimization module
        DelayProj - Clipping object from `sequence_constraints.py` restricting weight values
        WeightProj - Clipping object from `sequence_constraints.py` restricting delay values
        target_type - One of ['unencoded', 'synthetic']
        num_epochs - Number of training epochs
        desc_alg - One of ['SGD', 'Adam', 'LBFGS']
        sched, lrate, momentum - Parameters for descent algorithm
    Imaging Parameters
        image_dims - Pixel resolution in [lateral, axial] direction
        image_range - Imaging area [x_min, x_max, z_min, z_max], in mm
        dB_min - Minimum of the dynamic range
        roi_pads - [inner radius, outer radius] Padding for pixels in gCNR calculations
        hist_params - Dict with 'bins' and 'sigma' (smoothing parameter) keys
        hist_match - Boolean, performs histogram matching during imaging
    Encoding Parameters
        noise_param - Dict with ['BW', 'SNR'] keys for adding encoding noise
        tik_param - Tikhonov regularization parameter for decoding
        
    Saves optimization results in `results_folder`, including
        - L2 Loss after each epoch
        - gCNR after each epoch
        - Beamformed images after each epoch
        - Imaging targets
        - Initial/Trained encoding sequence
        - Condition number for first 150 encoding matrices
        - Text file containing test parameters
    """
    # Read data parameters from file
    dataset = uu.UltrasoundDataset( data_folder )
    acq_params = scipy.io.loadmat(f"{data_folder}/acq_params.mat") 

    # Fill out the default parameters for each section
    opt_params = {**s.default_opt_params, **(opt_params or {})}
    enc_params = {**s.default_enc_params, **(enc_params or {})}
    bf_params = {**s.default_bf_params, **(bf_params or {})}

    # Fix a few parameters directly
    if bf_params['image_range'] == None:
        bf_params['image_range'] = acq_params['image_range'][0]
    
    if opt_params['loss_func'] == None:
        opt_params['loss_func'] = uo.L2_loss

    # Extract some parameters for convinience
    [full_xpx, full_zpx] = bf_params['image_dims']
    image_range = bf_params['image_range']
    num_epochs = opt_params['num_epochs']
    loss_func = opt_params['loss_func']

    # Check that the acquisition parameters for the data is consistent
    assert acq_params['fs'].item() == s.fs
    assert acq_params['f0'].item() == s.f0
    assert acq_params['c'].item() == s.c
    assert np.allclose( acq_params['rx_pos'], s.rx_pos.numpy() )
    assert np.allclose( acq_params['tx_pos'], s.tx_pos.numpy() )

    if not isinstance( opt_params['training_resolution'], str ):
        res = [opt_params['training_resolution'] for i in range(num_epochs or 1)]
    elif opt_params['training_resolution'] == "graduated":
        res = list( np.linspace( 0.25, 1.0, num_epochs)) or [1.0]

    # Get indices for each dataset
    indices = list(range(len(dataset)))
    split = int(np.floor(0.2 * len(dataset)))
    if opt_params['training_shuffle']:
        np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]

    # Write description of test to file so you don't forget what you did
    print( f"Saving results to {results_folder}..." )
    with open(results_folder + '/test_description.txt', 'w') as f:
        f.write(f"Data folder used: {data_folder}\n")
        f.write(f"Training subset used: {train_idx}\n")
        f.write(f"Testing subset used: {test_idx}\n\n")
        f.write(f"Optimization parameters:\n")
        for key, value in opt_params.items():
            f.write(f"\t{key}: {value}\n")

        f.write(f"Encoding parameters:\n")
        for key, value in enc_params.items():
            f.write(f"\t{key}: {value}\n")

        f.write(f"Imaging parameters:\n")
        for key, value in bf_params.items():
            f.write(f"\t{key}: {value}\n")

        f.write("Other notes:\n")
        f.write(f"{notes}")

    training_dataloader = DataLoader(dataset, batch_size=8, sampler=train_idx, shuffle=False)

    # Define training model and optimzer
    uncompiled_model = PredictorModel(init_delays.clone(), init_weights.clone(), acq_params, enc_params, bf_params )

    # Switch on the right parameters
    if opt_params['trainable_params'] == "delays":
        uncompiled_model.delays.requires_grad = True
        uncompiled_model.weights.requires_grad = False
    elif opt_params['trainable_params'] == "weights":
        uncompiled_model.delays.requires_grad = False
    elif opt_params['trainable_params'] == "both":
        uncompiled_model.delays.requires_grad = True
        uncompiled_model.weights.requires_grad = True
    else:
        print("Invalid choice for optimization parameters")
        return

    # training_model = torch.compile( uncompiled_model ) # Untested, only available with PyTorch >2.0
    prediction_model = uncompiled_model
    
    # Select descent algorithm
    if opt_params['desc_alg'].lower() == "sgd":
        opt = torch.optim.SGD( uncompiled_model.parameters(), lr=opt_params['lrate'], momentum=opt_params['momentum'] )
    elif opt_params['desc_alg'].lower() == "adam":
        opt = torch.optim.Adam( uncompiled_model.parameters(), lr=opt_params['lrate'] )
    elif opt_params['desc_alg'].lower() == "bfgs" or opt_params['desc_alg'].lower() == "lbfgs":
        opt = torch.optim.LBFGS( uncompiled_model.parameters(), lr=opt_params['lrate'], max_iter=4 )
    else:
        print("Invalid choice for descent algorithm")
        return
    
    # Set exponential scheduler
    if not isinstance( opt_params['sched'], str ):
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=opt_params['sched'])
    elif opt_params['sched'] == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt_params['lrate'], steps_per_epoch=len(training_dataloader), epochs=num_epochs )
    
    # Plot early to make sure it's saving correctly
    np.savetxt( results_folder + "/initial_delays.csv", init_delays, delimiter=',' )    
    uu.plot_delays(init_delays, results_folder + "/initial_delays.png", "Initial Delays")
    
    np.savetxt( results_folder + "/initial_weights.csv", init_weights, delimiter=',' )    
    uu.plot_delays_weights(init_delays, init_weights, results_folder + "/initial_weights.png", "Initial Weights")
    uu.plot_weights( init_weights, results_folder + "/initial_weigths_array.png", "Initial Amplitude Weights" )
        
    # Store array of per-iteration testing loss for each
    testing_losses_loss_func = np.zeros([len(test_idx), num_epochs+1])
    testing_losses_L2 = np.zeros([len(test_idx), num_epochs+1])
    testing_losses_gCNR = np.zeros([len(test_idx), num_epochs+1])
    
    # Store array of per-iteration training loss for each
    training_losses_loss_func = np.zeros([len(train_idx), num_epochs+1])
    training_losses_L2 = np.zeros([len(train_idx), num_epochs+1])
    training_losses_gCNR = np.zeros([len(train_idx), num_epochs+1])
    
    # Store array of condition numbers at each iteration
    num_freq = 150
    condition_numbers = np.zeros([num_freq, num_epochs+1])
    condition_numbers[:,0] = torch.linalg.cond( ue.calc_H( 2*(num_freq-1), init_delays, init_weights ) )[0]

    N_test_images = 1 # Number of test images to record

    # Acquire testing data's beamformed images for comparison at each epoch
    image_targets = torch.empty( (len( test_idx ), full_zpx, full_xpx), dtype=s.PTFLOAT )
    print("Getting testing data", end="")
    prediction_model.set_resolution( int(full_xpx), int(full_zpx) )
    for (k, idx) in enumerate( test_idx ):
        datas, locs = [torch.tensor(x).unsqueeze(0) for x in dataset[idx] ]
        with torch.no_grad():
            init_images = prediction_model.get_image_prediction( datas, locs )
            image_targets[k,:,:] = prediction_model.get_targets( datas, locs, opt_params['target_type'] )
                
            testing_losses_loss_func[k, 0] = loss_func( init_images, image_targets[k], locs, bf_params )
            testing_losses_L2[k, 0] = uo.L2_loss( init_images, image_targets[k], locs, bf_params )
            testing_losses_gCNR[k, 0] = uo.gCNR_loss( init_images, image_targets[k], locs, bf_params )

            print( f"Testing Loss {k}: {testing_losses_loss_func[k,0]}" )

            if k < N_test_images:
                uu.plot_beamformed_image( init_images[0], [full_xpx, full_zpx], image_range, results_folder + f"/testing_image{k}_target.png", "No Encoding" )
                uu.plot_beamformed_image( image_targets[k], [full_xpx, full_zpx], image_range, results_folder + f"/testing_image{k}_target.png", "Imaging Target" )
            

    if save_training_images == True or save_training_loss == True:
        # Acquire training data's beamformed images for comparison at each epoch
        training_image_targets = torch.empty( (len( train_idx ), full_zpx, full_xpx), dtype=s.PTFLOAT )
        
        print("\nGetting training data", end="")
        for (k, idx) in enumerate( train_idx ):
            
            datas, locs = [torch.tensor(x).unsqueeze(0) for x in dataset[idx] ]
            with torch.no_grad():
                init_images = prediction_model.get_image_prediction( datas, locs )
                training_image_targets[k,:,:] = prediction_model.get_targets( datas, loss_closure, opt_params['target_type'] )
                  
                if save_training_loss == True:
                    training_losses_loss_func[k, 0] = loss_func( init_images, image_targets[k], locs, bf_params )
                    training_losses_L2[k, 0] = uo.L2_loss( init_images, training_image_targets[k], locs, bf_params )
                    training_losses_gCNR[k, 0] = uo.gCNR_loss( init_images, training_image_targets[k], locs, bf_params )

                if save_training_images == True:
                    uu.plot_beamformed_image( init_images[0], [full_xpx, full_zpx], image_range, results_folder + f"/training_image{k}_init.png", "Initial Encoding" )
                    uu.plot_beamformed_image( training_image_targets[k], [full_xpx, full_zpx], image_range, results_folder + f"/training_image{k}_target.png", "No Encoding" )
            
    # Train model
    print("\n-------------Training Phase-------------")
    epoch_losses = []
    for n in range(num_epochs):
        # Do training loop over all samples
        step_losses = []
        
        # Change the training resolution
        prediction_model.set_resolution( int(full_xpx * res[n]), int(full_zpx * res[n]) )
        
        for (k, [datas, locs]) in enumerate( training_dataloader ):    
            the_loss = torch.zeros(1, dtype=s.PTFLOAT)
            def loss_closure():
                nonlocal the_loss
                opt.zero_grad()

                # Acquire targets
                targets = prediction_model.get_targets( datas, locs, opt_params['target_type'] )
                    
                # Get prediction
                predictions = prediction_model.get_image_prediction( datas, locs )
                
                # Evaluate loss
                the_loss = loss_func( predictions, targets, locs, bf_params )

                # Perform backprogation on the loss
                the_loss.backward()
                return the_loss

            # Perform optimization step on the model
            opt.step(loss_closure)
            
            if opt_params['WeightProj'] != None:
                prediction_model.apply(opt_params['WeightProj'])
            if opt_params['DelayProj'] != None:
                prediction_model.apply(opt_params['DelayProj'])

            # Show the output
            step_losses.append(the_loss.item())
            
            print( f"step {k} loss: {step_losses[-1]}")

        # Step the scheduler, adjust learning rate
        if opt_params['sched'] != None:
            scheduler.step()
        
        np.savetxt( results_folder + f"/training_delays_{n}.csv", prediction_model.delays.detach().numpy(), delimiter=',' )
        np.savetxt( results_folder + f"/training_weights_{n}.csv", prediction_model.weights.detach().numpy(), delimiter=',' )
        
        # Save a plot of the gCNR training and testing losses
        uu.plot_losses( testing_losses_loss_func[:, :n],    training_losses_loss_func[:, :n], results_folder + f"/training_losses_LossFunc_{n}.png", f"{loss_func.__name__} Loss" )
        uu.plot_losses(        testing_losses_L2[:, :n],           training_losses_L2[:, :n], results_folder +       f"/training_losses_L2_{n}.png",                    "L2 Loss" )
        uu.plot_losses(     -testing_losses_gCNR[:, :n],        -training_losses_gCNR[:, :n], results_folder +     f"/training_losses_gCNR_{n}.png",                  "gCNR Loss" )
        
        uu.plot_delays_weights(prediction_model.delays.detach().numpy(), prediction_model.weights.detach().numpy(), results_folder + f"/training_weights_{n}.png", "Learned Weights")

        condition_numbers[:,n+1] = torch.linalg.cond( ue.calc_H( 2*(num_freq-1), prediction_model.delays.clone().detach(), prediction_model.weights.clone().detach() ) )[0]
    
        # Get L2 loss, gCNR on the testing set after each epoch
        prediction_model.set_resolution( int(full_xpx), int(full_zpx) )
        for (k, idx) in enumerate( test_idx ):
            datas, locs = [torch.tensor(x).unsqueeze(0) for x in dataset[idx] ]
            with torch.no_grad():
                # Compute loss on testing data
                testing_prediction = prediction_model.get_image_prediction( datas, locs )[0]

                testing_losses_loss_func[k, n+1] = loss_func( init_images, image_targets[k], locs, bf_params )
                testing_losses_L2[k, n+1] = uo.L2_loss( init_images, image_targets[k], locs, bf_params )
                testing_losses_gCNR[k, n+1] = uo.gCNR_loss( init_images, image_targets[k], locs, bf_params )
            
                uu.plot_beamformed_image( testing_prediction, [full_xpx, full_zpx], image_range, results_folder + f"/testing_image{k}_epoch{n}.png", "Learning Encoding" )
                
        if save_training_images == True or save_training_loss == True:
            # Get L2 loss, gCNR on the training set after each epoch
            prediction_model.set_resolution( int(full_xpx * res[n]), int(full_zpx * res[n]) )
            for (k, idx) in enumerate( train_idx ):
                datas, locs = [torch.tensor(x).unsqueeze(0) for x in dataset[idx] ]
                with torch.no_grad():
                    # Compute loss on training data, but at full resolution
                    training_predictions = prediction_model.get_image_prediction( datas, locs )
                    
                    if save_training_loss == True:
                        training_losses_loss_func[k, n+1] = loss_func( training_predictions, image_targets[k], locs, bf_params )
                        training_losses_L2[k, n+1] = uo.L2_loss( training_predictions, training_image_targets[k], locs, bf_params )
                        training_losses_gCNR[k, n+1] = uo.gCNR_loss( training_predictions, training_image_targets[k], locs, bf_params )

                    if save_training_images == True:
                        uu.plot_beamformed_image( training_predictions[0], [full_xpx, full_zpx], image_range, results_folder + f"/training_image{k}_init.png", "Initial Encoding" )
 
        print( f"epoch: {n}, average loss: {np.mean( step_losses )}" )
        epoch_losses.append( np.mean( step_losses ) )

    # Save as much data as we have available
    uu.plot_delays(prediction_model.delays.detach().numpy(), results_folder + "/trained_delays.png", "Learned Delays")
    uu.plot_delays_weights(prediction_model.delays.detach().numpy(), prediction_model.weights.detach().numpy(), results_folder + "/trained_weights.png", "Learned Weights")
    uu.plot_weights( prediction_model.weights.detach().numpy(), results_folder + "/trained_weigths_array.png", "Learned Amplitude Weights" )
    
    np.savetxt( results_folder + "/trained_delays.csv", prediction_model.delays.detach().numpy(), delimiter=',' )
    np.savetxt( results_folder + "/trained_weights.csv", prediction_model.weights.detach().numpy(), delimiter=',' )
    
    np.savetxt( results_folder + "/training_losses_epoch.csv", np.array( epoch_losses ), delimiter=',' )

    np.savetxt( results_folder + "/testing_losses_L2.csv", testing_losses_L2, delimiter=',' )
    np.savetxt( results_folder + "/testing_losses_gCNR.csv", testing_losses_gCNR, delimiter=',' )
    
    np.savetxt( results_folder + "/training_losses_L2.csv", training_losses_L2, delimiter=',' )
    np.savetxt( results_folder + "/training_losses_gCNR.csv", training_losses_gCNR, delimiter=',' )

    np.savetxt( results_folder + "/condition_numbers.csv", condition_numbers, delimiter=',' )
