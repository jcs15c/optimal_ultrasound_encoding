import numpy as np
import torch

import scipy
import scipy.io

import src.ultrasound_utilities as uu
import src.ultrasound_encoding as ue
import src.ultrasound_imaging as ui

from src.predictor_model import PredictorModel
from torch.utils.data import DataLoader

import src.settings as s

def optimal_encoding_sequence(test_folder, training_set, testing_set, # Folder names containing data and results
                           init_delays, init_weights,  # optimization initial parameters
                           opt_params = "both", loss_func="L2", WeightProj=None, training_resolution="half",  target_type="unencoded", # optimization parameters
                           num_epochs=15, desc_alg="SGD", lrate=0.1, sched=None, momentum=0, training_shuffle = True, # descent algorithm hyperparameters
                           image_dims=[300, 500], image_range=[-25, 25, 15, 55], dB_min=-60,   # image parameters
                           noise_param=[0.7, 12], tik_param=0.1, # Encoding parameters
                           hist_param=[100,3], gCNR_pad=[1.0, 1.0], notes=None,
                           save_training_images = False, save_training_loss = False): # less important/fixed parameters
    """
    Takes a number of encoding, imaging, and optimization parameters to find
    the encoding sequence that minimize given the objective function

    Meta-Parameters:
        test_folder - Location to save optimization results
        training/testing_set - Folders containing RF data for training/testing
        save_training_images/loss - If true, store value for training
        notes - String to save notes in record file
    Optimization Parameters:
        init_delays/weights - Initial encoding sequence for optimization
        opt_params - One of ['delays', 'weights', 'both']
        loss_func - One of ['L2', 'gCNR']
        WeightProj - Clipper object from `weight_constraints.py` restricting weight values
        training_resolution - One of ['full', 'half', 'graduated']
        target_type - One of ['unencoded', 'synthetic']
        num_epochs - Number of training epochs
        desc_alg - One of ['SGD', 'Adam', 'LBFGS']
        sched, lrate, momentum - Parameters for descent algorithm
        training_shuffle - Shuffle data during training phase
    Imaging Parameters
        image_dims - Pixel resolution in [lateral, axial] direction
        image_range - Imaging area [x_min, x_max, z_min, z_max], in mm
        dB_min - Minimum of the dynamic range
        gCNR_pad - [inner radius, outer radius] Padding for pixels in gCNR calculation
        hist_param - [bins, smoothing parameter] for smooth histogram calculations
    Encoding Parameters
        noise_param - [Bandwidth, SNR] parameters for adding encoding noise
        tik_param - Tikhonov regularization parameter for decoding
        
    Returns:
        Saves optimization results in `test_folder`, including
            - L2 Loss after each epoch
            - gCNR after each epoch
            - Beamformed images after each epoch
            - Imaging targets
            - Initial/Trained encoding sequence
            - Condition number for first 150 encoding matrices
            - Text file containing test parameters
    """
    print( f"Saving results to {test_folder}..." )
    
    # Create datasets for training and testing
    training_dataset = uu.UltrasoundDataset( training_set )
    training_dataloader = DataLoader(training_dataset, batch_size=len( training_dataset ) // 8, shuffle=training_shuffle)
    training_acq_params = scipy.io.loadmat(f"{training_set}/acq_params.mat") 

    testing_dataset = uu.UltrasoundDataset( testing_set )   
    testing_acq_params = scipy.io.loadmat(f"{testing_set}/acq_params.mat") 
        
    # Check that the acquisition parameters for the data is consistent
    assert training_acq_params['fs'].item() == testing_acq_params['fs'].item() == s.fs
    assert training_acq_params['f0'].item() == testing_acq_params['f0'].item() == s.f0
    assert training_acq_params['c'].item() == testing_acq_params['c'].item() == s.c
    assert (training_acq_params['rx_pos'] == testing_acq_params['rx_pos']).all() \
        and np.allclose( training_acq_params['rx_pos'], s.rx_pos.numpy() )
    assert (training_acq_params['tx_pos'] == testing_acq_params['tx_pos']).all() \
        and np.allclose( training_acq_params['tx_pos'], s.tx_pos.numpy() )
    
    # We train at a lower resolution than we test for efficiency,
    #   so get list of resolution scales
    if training_resolution == "full":
        res = [1 for i in range(num_epochs or 1)]
    if training_resolution == "half":
        res = [0.5 for i in range(num_epochs or 1)]
    if training_resolution == "graduated":
        res = list( np.linspace( 0.25, 1.0, num_epochs)) or [1.0]
    
    # Define full image parameters for testing (number of pixels)
    xpx = image_dims[0]
    zpx = image_dims[1]

    # Define testing resolution (1D grid)
    x_te = torch.linspace(image_range[0], image_range[1], steps=xpx)/1000
    z_te = torch.linspace(image_range[2], image_range[3], steps=zpx)/1000
    
    # Define training resolution (1D grid)
    x_tr = torch.linspace(image_range[0], image_range[1], steps=int(xpx*res[0]) )/1000
    z_tr = torch.linspace(image_range[2], image_range[3], steps=int(zpx*res[0]) )/1000    

    # Define "default" histogram data from standard spackel pattern
    hist_data = [-11.811993598937988, 5.649099826812744]

    # Define training model and optimzer
    training_model = PredictorModel(init_delays.clone(), init_weights.clone(), training_acq_params, [x_tr, z_tr], 
                                    tik_param=tik_param, dB_min=dB_min, hist_data=hist_data, noise_param=noise_param )

    # Switch on the right parameters
    if opt_params == "delays":
        training_model.delays.requires_grad = True
        training_model.weights.requires_grad = False
    elif opt_params == "weights":
        training_model.delays.requires_grad = False
    elif opt_params == "both":
        training_model.delays.requires_grad = True
        training_model.weights.requires_grad = True
    else:
        print("Invalid choice for optimization parameters")
        return

    # Select descent algorithm
    if desc_alg.lower() == "sgd":
        opt = torch.optim.SGD( training_model.parameters(), lr=lrate, momentum=momentum )
    elif desc_alg.lower() == "adam":
        opt = torch.optim.Adam( training_model.parameters(), lr=lrate )
    elif desc_alg.lower() == "bfgs" or desc_alg.lower() == "lbfgs":
        opt = torch.optim.LBFGS( training_model.parameters(), lr=lrate, max_iter=4 )
    else:
        print("Invalid choice for descent algorithm")
        return
    
    # Set exponential scheduler
    if sched != None:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=sched)
   
    # Plot early to make sure it's saving correctly
    np.savetxt( test_folder + "/initial_delays.csv", init_delays, delimiter=',' )    
    uu.plot_delays(init_delays, test_folder + "/initial_delays.png", "Initial Delays")
    
    np.savetxt( test_folder + "/initial_weights.csv", init_weights, delimiter=',' )    
    uu.plot_delays_weights(init_delays, init_weights, test_folder + "/initial_weights.png", "Initial Weights")
    uu.plot_weights( init_weights, test_folder + "/initial_weigths_array.png", "Initial Amplitude Weights" )
        
    # Define encoding models at high resolution for testing
    testing_model = PredictorModel(init_delays.clone(), init_weights.clone(), testing_acq_params, [x_te, z_te], 
                                   tik_param=tik_param, dB_min=dB_min, hist_data=hist_data, noise_param=noise_param )
        
    # Store array of per-iteration testing loss for each
    testing_losses_L2 = np.zeros([len(testing_dataset), num_epochs+1])
    testing_losses_gCNR = np.zeros([len(testing_dataset), num_epochs+1])
    
    # Store array of per-iteration training loss for each
    training_losses_L2 = np.zeros([len(training_dataset), num_epochs+1])
    training_losses_gCNR = np.zeros([len(training_dataset), num_epochs+1])
    
    # Store array of condition numbers at each iteration
    num_freq = 150
    condition_numbers = np.zeros([num_freq, num_epochs+1])
    condition_numbers[:,0] = torch.linalg.cond( ue.calc_H( 2*(num_freq-1), init_delays, init_weights ) )[0]
    
    # Acquire testing data's beamformed images for comparison at each epoch
    image_targets = torch.empty( (len( testing_dataset ), zpx, xpx), dtype=s.PTFLOAT )
    print("Getting testing data", end="")
    for (k, [data, loc]) in enumerate( testing_dataset ):
        
        print(".", end="")
        with torch.no_grad():
            data = torch.tensor( data ).unsqueeze(0)
            loc = torch.tensor( loc ).unsqueeze(0)
            
            image_targets[k,:,:] = testing_model.get_targets( data, loc, target_type )
            
            init_image = testing_model.get_image_prediction( data, loc )[0]
                
            testing_losses_L2[k, 0] = torch.mean( torch.linalg.norm( init_image - image_targets[k]) )
            testing_losses_gCNR[k, 0] = ui.mean_gCNR( init_image.unsqueeze(0), loc, [x_te, z_te], hist_param=hist_param, rpad=gCNR_pad )

            uu.plot_beamformed_image( init_image, image_dims, image_range, test_folder + f"/testing_image{k}_init.png", "Initial Encoding" )
            uu.plot_beamformed_image( image_targets[k], image_dims, image_range, test_folder + f"/testing_image{k}_target.png", "No Encoding" )
            
    if save_training_images == True or save_training_loss == True:
        # Acquire training data's beamformed images for comparison at each epoch
        training_image_targets = torch.empty( (len( training_dataset ), zpx, xpx), dtype=s.PTFLOAT )
        
        print("\nGetting training data", end="")
        for (k, [data, loc]) in enumerate( training_dataset ):
            
            print(".", end="")
            with torch.no_grad():
                data = torch.tensor( data ).unsqueeze(0)
                loc = torch.tensor( loc ).unsqueeze(0)
                
                training_image_targets[k,:,:] = testing_model.get_targets( data, loc, target_type )
                
                init_image = testing_model.get_image_prediction( data, loc )[0]
                  
                if save_training_loss == True:
                    training_losses_L2[k, 0] = torch.mean( torch.linalg.norm( init_image - training_image_targets[k]) )
                    training_losses_gCNR[k, 0] = ui.mean_gCNR( init_image.unsqueeze(0), loc, [x_te, z_te], hist_param=hist_param, rpad=gCNR_pad )   
                    
                if save_training_images == True:
                    uu.plot_beamformed_image( init_image, image_dims, image_range, test_folder + f"/training_image{k}_init.png", "Initial Encoding" )
                    uu.plot_beamformed_image( training_image_targets[k], image_dims, image_range, test_folder + f"/training_image{k}_target.png", "No Encoding" )
            
    # Train model
    print("\n-------------Training Phase-------------")
    epoch_losses = []
    for n in range(num_epochs):
        # Do training loop over all samples
        step_losses = []
        
        # Change the training resolution
        x = torch.linspace(image_range[0], image_range[1], steps=int( xpx * res[n] ))/1000
        z = torch.linspace(image_range[2], image_range[3], steps=int( zpx * res[n] ))/1000
        training_model.change_resolution( [x, z] )
        
        for (k, [datas, locs]) in enumerate( training_dataloader ):    
            the_loss = torch.zeros(1, dtype=s.PTFLOAT)
            def loss_closure():
                nonlocal the_loss
                opt.zero_grad()

                # Acquire targets
                targets = training_model.get_targets( data, loc, target_type )
                    
                # Get prediction
                predictions = training_model.get_image_prediction( datas, locs )
                
                # Evaluate loss
                if loss_func.lower() == "gcnr": # we want to maximize gCNR
                    the_loss = -ui.mean_gCNR( predictions, locs, [x, z], hist_param=hist_param, rpad=gCNR_pad )
                else:
                    the_loss = torch.mean( torch.linalg.norm( predictions - targets, axis=(1,2) ) )
                
                # Perform backprogation on the loss
                the_loss.backward()
                return the_loss

            # Perform optimization step on the model
            opt.step(loss_closure)
            
            if WeightProj != None:
                training_model.apply(WeightProj)
            
            # Show the output
            step_losses.append(the_loss.item())
            
            print( f"step {k} loss: {step_losses[-1]}")

        # Step the scheduler, adjust learning rate
        if sched != None:
            scheduler.step()
        
        # Adjust high-res models with learned delays or weights
        testing_model.delays = torch.nn.Parameter( training_model.delays.clone() )
        testing_model.weights = torch.nn.Parameter( training_model.weights.clone() )
        
        condition_numbers[:,n+1] = torch.linalg.cond( ue.calc_H( 2*(num_freq-1), training_model.delays.clone().detach(), training_model.weights.clone().detach() ) )[0]
    
        # Get L2 loss, gCNR on the testing set after each epoch
        for (k, [data, loc]) in enumerate( testing_dataset ):
            with torch.no_grad():
                data = torch.tensor( data ).unsqueeze(0)
                loc = torch.tensor( loc ).unsqueeze(0)
                
                # Compute loss on testing data
                testing_prediction = testing_model.get_image_prediction( data, loc )[0]
            
                testing_losses_L2[k, n+1] = torch.mean( torch.linalg.norm( testing_prediction - image_targets[k]) )
                testing_losses_gCNR[k, n+1] = ui.mean_gCNR( testing_prediction.unsqueeze(0), loc, [x_te, z_te], hist_param=hist_param, rpad=gCNR_pad )   
            
                uu.plot_beamformed_image( testing_prediction, image_dims, image_range, test_folder + f"/testing_image{k}_epoch{n}.png", "Learning Encoding" )
                
        if save_training_images == True or save_training_loss == True:
            # Get L2 loss, gCNR on the training set after each epoch
            for (k, [data, loc]) in enumerate( training_dataset ):
                with torch.no_grad():
                    data = torch.tensor( data ).unsqueeze(0)
                    loc = torch.tensor( loc ).unsqueeze(0)
                    
                    # Compute loss on training data, but at full resolution
                    training_prediction = testing_model.get_image_prediction( data, loc )[0]
                
                    if save_training_loss == True:
                        training_losses_L2[k, n+1] = torch.mean( torch.linalg.norm( training_prediction - training_image_targets[k]) )
                        training_losses_gCNR[k, n+1] = ui.mean_gCNR( training_prediction.unsqueeze(0), loc, [x_te, z_te], hist_param=hist_param, rpad=gCNR_pad )   
                    
                    if save_training_images == True:
                        uu.plot_beamformed_image( training_prediction, image_dims, image_range, test_folder + f"/testing_image{k}_epoch{n}.png", "Learning Encoding" )
                
        print( f"epoch: {n}, average loss: {np.mean( step_losses )}" )
        epoch_losses.append( np.mean( step_losses ) )

    # Save as much data as we have available
    uu.plot_delays(training_model.delays.detach().numpy(), test_folder + "/trained_delays.png", "Learned Delays")
    uu.plot_delays_weights(training_model.delays.detach().numpy(), training_model.weights.detach().numpy(), test_folder + "/trained_weights.png", "Learned Weights")
    uu.plot_weights( training_model.weights.detach().numpy(), test_folder + "/trained_weigths_array.png", "Learned Amplitude Weights" )
    
    np.savetxt( test_folder + "/trained_delays.csv", training_model.delays.detach().numpy(), delimiter=',' )
    np.savetxt( test_folder + "/trained_weights.csv", training_model.weights.detach().numpy(), delimiter=',' )
    
    np.savetxt( test_folder + "/training_losses_epoch.csv", np.array( epoch_losses ), delimiter=',' )

    np.savetxt( test_folder + "/testing_losses_L2.csv", testing_losses_L2, delimiter=',' )
    np.savetxt( test_folder + "/testing_losses_gCNR.csv", testing_losses_gCNR, delimiter=',' )
    
    np.savetxt( test_folder + "/training_losses_L2.csv", training_losses_L2, delimiter=',' )
    np.savetxt( test_folder + "/training_losses_gCNR.csv", training_losses_gCNR, delimiter=',' )

    np.savetxt( test_folder + "/condition_numbers.csv", condition_numbers, delimiter=',' )
    
    # Write description of test to file so you don't forget what you did
    with open(test_folder + '/test_description.txt', 'w') as f:
        f.write(f"Original test name: {test_folder}\n")
        f.write(f"Training dataset used: {training_set}\n")
        f.write(f"Testing dataset used: {testing_set}\n")
        f.write(f"Optimization parameters: {opt_params}\n")
        f.write(f"Target type: {target_type}\n")

        f.write(f"Number of training epochs: {num_epochs}\n")
        f.write(f"Training shuffle: {training_shuffle}\n")
        f.write(f"Descent algorithm: {desc_alg}\n")
        f.write(f"Loss function: {loss_func}\n")
        if WeightProj is not None:
            WeightProj = WeightProj.__class__.__name__
        f.write(f"Amplitude weight projection: {WeightProj}\n")
 
        f.write(f"Training resolution warmup: {training_resolution}\n\n")
        f.write(f"Learning rate: {lrate}\n")
        f.write(f"Scheduler? {sched}\n")
        f.write(f"Momentum: {momentum}\n")
        
        f.write(f"Image pixel size: {image_dims}\n")
        f.write(f"Noise parameters [BW, SNR]: {noise_param}\n")
        f.write(f"Tikhonov Regularization parameter: {tik_param}\n")
        
        f.write(f"Dynamic range minimum: {dB_min}\n")
        f.write(f"Smooth histogram parameters: {hist_param}\n")
        f.write("Other notes:\n")
        f.write(f"{notes}")