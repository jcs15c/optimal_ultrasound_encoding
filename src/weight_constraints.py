import numpy as np
import torch

"""
Define several projection operators to add constraints on our weights,
or else in the presence of noise, the optimal solution is to make each
arbitrarily large
"""

class L1BallProjection( object ):
    """
    Project weights of class to an L1 unit ball
    """
    def __call__(self, module):
        if hasattr( module, "weights" ):
            w = module.weights.data
            eps = w.numel() # Radius of L1 ball

            # Only project onto the ball if we need to
            if torch.norm( w, 1 ) > eps:                
                original_shape = w.shape
                x = w.clone().flatten()
                mu, _ = torch.sort( torch.abs(x), descending=True )
                cumsum = torch.cumsum( mu, dim=0 )
                arange = torch.arange( 1, x.numel() + 1 )
                rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=0)
                theta = (cumsum[rho - 1] - eps) / rho
                proj = (torch.abs(x) - theta.unsqueeze(0)).clamp(min=0)
                x = proj * torch.sign(x)
                w = x.view(original_shape)
                module.weights.data = w
                
class L1BallIntRounding( object ):
    """
    Project weights of class to an L1 unit ball, 
    then round to the nearest integer
    """
    def __call__(self, module):
        if hasattr( module, "weights" ):
            w = module.weights.data
            eps = w.numel() # Radius of L1 ball

            # Only project onto the ball if we need to
            if torch.norm( w, 1 ) > eps:                
                original_shape = w.shape
                x = w.clone().flatten()
                mu, _ = torch.sort( torch.abs(x), descending=True )
                cumsum = torch.cumsum( mu, dim=0 )
                arange = torch.arange( 1, x.numel() + 1 )
                rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=0)
                theta = (cumsum[rho - 1] - eps) / rho
                proj = (torch.abs(x) - theta.unsqueeze(0)).clamp(min=0)
                x = proj * torch.sign(x)
                w = x.view(original_shape)
                
            module.weights.data = torch.round(w)
                
class L2BallProjection( object ):
    """
    Project weights of class to an L2 unit ball
    """
    def __call__(self, module):
        if hasattr( module, "weights" ):
            w = module.weights.data
            w.mul_( max(1.0, np.sqrt(w.numel()) / torch.norm( w.flatten(), 2 ) ) )
            module.weights.data = w

class L2SquaredNormProjection( object ):
    """
    Project weights of class to an L2-squared ball
    """
    def __call__(self, module):
        if hasattr( module, "weights" ):
            w = module.weights.data
            w.div_( (torch.norm( w.flatten(), 2 )**2).expand_as(w) )
            w.mul_( float(w.numel()) )        
            
class LinfBallProjection( object ):
    """
    Project weights of class to an L-infinity ball
    """
    def __call__(self, module):
        if hasattr( module, "weights" ):
            w = module.weights.data
            w = w.clamp(-1, 1)
            module.weights.data = w
            
class VerasonicsProjection( object ):
    """
    Restrict weights to those readable by our Verasonics hardware,
    with |weights| < 0.2 clamped to zero, and rounded to 2 decimals otherwise
    """
    def __call__(self, module):
        if hasattr (module, "weights" ):
            w = module.weights.data
            w[abs(w) < .2] = 0
            w = torch.round(w, decimals = 2)
            module.weights.data = w
