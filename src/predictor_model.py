import torch

import src.ultrasound_encoding as ue
import src.ultrasound_imaging as ui
import src.settings as s

class PredictorModel(torch.nn.Module):
    """
    Model to turn RF data into a beamformed image
    
    Initialization inputs:
        > init_delays/weights : Initial encoding sequence
        > acq_params : Parameters for the acquisition of RF data
        > image_grid : [xpts, zpts] Coordinates of each pixel in image (mm)
        > tik_param : Tikhonov regularization parameter
        > dB_min : Minimum of the dynamic range
        > noise_param : [bandwidth, SNR] parameters for encoding noise
        > hist_data : [mean, std] parameters for background spackle pattern
    """
    def __init__(self, init_delays, init_weights, acq_params, enc_params, bf_params ):
        super().__init__()
        # Store NN parameters
        self.delays = torch.nn.Parameter( init_delays )
        self.weights = torch.nn.Parameter( init_weights )
        
        # Store encoding and imaging parameters
        self.enc_params = enc_params
        self.bf_params = bf_params

        # Store beamforming parameters
        self.acq_params = acq_params
        self.r0 = (acq_params['data_span'][0,0] / acq_params['fs'][0,0] * acq_params['c'][0,0])
        self.dr = (acq_params['c'][0,0] / acq_params['fs'][0,0])
        self.set_resolution( bf_params['image_dims'][0], bf_params['image_dims'][1] )

    def set_resolution( self, xpx, zpx ):
        """
        Changes the resolution of the images produced by the model
        """
        
        # Store delays used for pixel-by-pixel beamforming
        x = torch.linspace( self.bf_params['image_range'][0], self.bf_params['image_range'][1], xpx )/1000
        z = torch.linspace( self.bf_params['image_range'][2], self.bf_params['image_range'][3], zpx )/1000
        self.bf_delays = ue.calc_delays_beamforming( self.acq_params['rx_pos'], x, z )

        if torch.cuda.is_available():
            self.bf_delays = self.bf_delays.to( torch.device( "cuda:0" ) )
        else:
            print( "GPU unavailable" )

        self.bf_params['image_dims'] = [xpx, zpx]
        
    def get_image_prediction(self, datas, locs ):
        """
        Evaluate the imaging model for given RF data and target locations
        """
        predictions = torch.empty( (datas.shape[0], self.bf_params['image_dims'][1], self.bf_params['image_dims'][0]), dtype=s.PTFLOAT )
        
        # Generate encoding and decoding matrices
        H = ue.calc_H( datas.shape[1], self.delays, self.weights)
        Hinv = ue.calc_Hinv_tikhonov(H, param=self.enc_params['tik_param'])
        
        # Generate encoding noise, if applicable
        if self.enc_params['noise_params'] != None:
            noise_instances = ue.generate_encoded_noise( torch.sum( datas, dim=3 ), self.acq_params, self.enc_params['noise_params'], \
                                                   [datas.shape[0], datas.shape[1], datas.shape[2], H.shape[2]] )
        else:
            noise_instances = torch.zeros( [datas.shape[0], datas.shape[1], datas.shape[2], H.shape[2]] )
            
        for k in range(datas.shape[0]):                    
            # Encode beams
            rf_enc = ue.encode(datas[k], H) 
                
            # Add noise instance
            rf_enc = rf_enc + noise_instances[k]
            
            # Decode beams
            rf_dec = ue.encode(rf_enc, Hinv)

            # Perform beamforming step
            iq_focused = ui.BeamformAD.apply(ue.hilbert( rf_dec ), self.r0, self.dr, self.bf_delays, 
                                             self.delays.shape[1], self.bf_params['image_dims'][0], self.bf_params['image_dims'][1])
            
            # Store resulting image in array. Add tiny variable to avoid log of zeros
            predictions[k,:,:] = torch.abs(iq_focused) + 1e-15
            
        # Perform histogram matching to fix brightness issues
        if self.bf_params['hist_match'] == True:
            rescaled_pred = ui.partial_histogram_matching( 20*torch.log10( predictions / torch.amax(predictions, dim=(1,2)).view(-1, 1, 1) ), locs, 
                                                            self.bf_params, s.hist_data[0], s.hist_data[1])
        else:
            rescaled_pred = 20*torch.log10( predictions / torch.amax(predictions, dim=(1,2)).view(-1, 1, 1) )

        # Clamp data to correct decibel range
        return torch.clamp(rescaled_pred, min=self.bf_params['dB_min'], max=0 )
       
    def get_data_prediction( self, datas, noise_instances=None ):
        """
        Evaluate the data model for given RF data and target locations
        """
        encoded_datas = torch.empty( (datas.shape[0], datas.shape[1], datas.shape[2], datas.shape[3]), dtype=s.PTFLOAT )
        
        # Generate encoding and decoding matrices
        H = ue.calc_H( datas.shape[1], self.delays, self.weights)
        Hinv = ue.calc_Hinv_tikhonov(H, param=self.enc_params['tik_param'])
        
        # Generate encoding noise, if applicable
        if self.noise_param != None:
            noise_instances = ue.generate_encoded_noise( torch.sum( datas, dim=3 ), self.acq_params, self.enc_params['noise_params'], \
                                                   [datas.shape[0], datas.shape[1], datas.shape[2], H.shape[2]] )        
        else:
            noise_instances = torch.zeros( [datas.shape[0], datas.shape[1], datas.shape[2], H.shape[2]] )
      
        for k in range(datas.shape[0]):                    
            # Encode beams
            rf_enc = ue.encode(datas[k], H) 
                
            # Add noise instance
            rf_enc = rf_enc + noise_instances[k]
            
            # Decode beams
            rf_dec = ue.encode(rf_enc, Hinv)

            encoded_datas[k,:,:,:] = rf_dec
            
        return encoded_datas
    
    def get_targets( self, datas, locs, target_type ):
        """
        Generate target images for given RF data, target locations, 
        and target type (unencoded or synthetic)
        """
        targets = torch.empty( (datas.shape[0], self.bf_params['image_dims'][1], self.bf_params['image_dims'][0]), dtype=s.PTFLOAT )
        
        if target_type.lower() == "unencoded":
            for k in range(datas.shape[0]):
                targets[k,:,:] = torch.abs( ui.BeamformAD.apply(ue.hilbert( datas[k] ), self.r0, self.dr, self.bf_delays, 
                                              self.delays.shape[1], self.bf_params['image_dims'][0], self.bf_params['image_dims'][1]) ) + 1e-15
            
            if self.bf_params['hist_match'] == True:
                rescaled_pred = ui.partial_histogram_matching( 20*torch.log10( targets / torch.amax(targets, dim=(1,2)).view(-1, 1, 1) ), locs, 
                                                            self.bf_params, s.hist_data[0], s.hist_data[1])
            else:
                rescaled_pred = 20*torch.log10( targets / torch.amax(targets, dim=(1,2)).view(-1, 1, 1) )

            return torch.clamp(rescaled_pred, min=self.bf_params['dB_min'], max=0 )
        elif target_type.lower() == "synthetic":
            masks = ui.create_target_mask( locs, self.bf_params ).float()
            return s.hist_data[0] * (1 - masks) + s.hist_data[1] * masks
        else:
            print( "Invalid target type" )
            return
