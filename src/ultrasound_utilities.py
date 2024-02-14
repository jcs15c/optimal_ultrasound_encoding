import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt

import pandas as pd

import src.settings as s

class UltrasoundDataset( torch.utils.data.Dataset ):
    """
    Return RF data and locations within a given folder
    """
    def __init__(self, data_directory):
        """
        Load filenames for each piece of data
        """
        self.data_dir = data_directory
        self.data_names = pd.read_csv( data_directory + '/data_filenames.csv' )
        self.loc_names = pd.read_csv( data_directory + '/loc_filenames.csv' )
        
    def __len__(self):
        return len(self.data_names)
    
    def __getitem__(self, idx):
        """
        Return RF data and locationd data
        """
        return np.load( self.data_dir + '/' + self.data_names.iloc[idx, 0]).astype( s.NPFLOAT ), \
               np.load( self.data_dir + '/' + self.loc_names.iloc[idx, 0]).astype( s.NPFLOAT )

class UltrasoundImageDataset( torch.utils.data.Dataset ):
    """
    Return RF data and locations within a given folder
    """
    def __init__(self, data_directory):
        """
        Load filenames for each piece of data
        """
        self.data_dir = data_directory
        self.data_names = pd.read_csv( data_directory + '/data_filenames.csv' )
        self.loc_names = pd.read_csv( data_directory + '/cmap_filenames.csv' )
        
    def __len__(self):
        return len(self.data_names)
    
    def __getitem__(self, idx):
        """
        Return RF data and locationd data
        """
        return np.load( self.data_dir + '/' + self.data_names.iloc[idx, 0]).astype( s.NPFLOAT ), \
               np.load( self.data_dir + '/' + self.loc_names.iloc[idx, 0]).astype( s.NPFLOAT )
    
    def get_named_item(self, idx):
        return np.load( self.data_dir + '/' + self.data_names.iloc[idx, 0]).astype( s.NPFLOAT ), \
               np.load( self.data_dir + '/' + self.loc_names.iloc[idx, 0]).astype( s.NPFLOAT ), \
               self.data_names.iloc[idx, 0].replace('_data', '').replace('.npy', '')

class SoftHistogram(torch.nn.Module):
    """
    Define the smooth histogram function for use in automatic differentiation
    """
    def __init__(self, bins, min, max, sigma):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.sigmoid(self.sigma * (x + self.delta/2)) - torch.sigmoid(self.sigma * (x - self.delta/2))
        x = x.sum(dim=1)
        return x

def plot_beamformed_image( env, image_dims, image_range, filename, title ):
    """
    Given complex envelope, plot the image.
    
    Parameters:
        env - 
        image_dims - Pixel resolution in [lateral, axial] direction
        image_range - Imaging area [x_min, x_max, z_min, z_max], in mm
        filename - Filename to save the image to
        title - Title of the plot
        
    Returns:
        Saves the image to `filename`
    """
    plt.cla()
    x = np.linspace(image_range[0], image_range[1], num=image_dims[0], endpoint=True)/1000
    z = np.linspace(image_range[2], image_range[3], num=image_dims[1], endpoint=True)/1000
    plt.clf()
    plt.imshow(env, cmap='gray',
           extent=[x[0]*1e3, x[-1]*1e3, z[-1]*1e3, z[0]*1e3],
           vmin=env.min(), vmax=0)
    plt.axis('equal')
    plt.title( title )
    plt.colorbar()
    plt.savefig( filename )

def plot_delays(delays, filename, title):
    """
    Plot a set of delays
    
    Parameters:
        delays - The delays to plot
        filename - Filename to save the image to
        title - Title of the plot
        
    Returns:
        Saves the image to `filename`
    """
    plt.clf()
    plt.plot( delays.T )
    plt.xlabel( "Element" )
    plt.ylabel( "Delay (samples)" )
    plt.title( title )
    plt.savefig( filename )

def plot_weights( weights, filename, title ):
    """
    Plot a set of weights in a grid
    
    Parameters:
        weights - The weights to plot
        filename - Filename to save the image to
        title - Title of the plot
        
    Returns:
        Saves the image to `filename`
    """
    plt.clf()
    plt.imshow( weights, cmap='bwr', norm=matplotlib.colors.TwoSlopeNorm(0) )
    plt.colorbar()

    plt.xlabel( "Element" )
    plt.ylabel( "Delay (samples)" )
    plt.title( title )
    plt.tight_layout()
    plt.savefig( filename )

def plot_delays_weights( delays, weights, filename, title ):
    """
    Plot a set of weights overlaid on the delays 
    
    Parameters:
        delays - The delays to plot
        weights - The delays to plot
        filename - Filename to save the image to
        title - Title of the plot
        
    Returns:
        Saves the image to `filename`
    """
    plt.clf()
    plt.plot( delays.T, 'k', alpha=0.1 )
    xs = np.stack( [np.arange(delays.T.shape[0]) for i in range(delays.T.shape[1]) ] ).T
    plt.scatter( xs, delays.T, c=weights.T, edgecolor='none', cmap='bwr', norm=matplotlib.colors.TwoSlopeNorm(0) )
    plt.colorbar()

    plt.xlabel( "Element" )
    plt.ylabel( "Delay (samples)" )
    plt.title( title )
    plt.tight_layout()
    plt.savefig( filename )


def plot_losses( testing_losses, training_losses, filename, title ):
    """
    Plot a set of weights overlaid on the delays 
    
    Parameters:
        delays - The delays to plot
        weights - The delays to plot
        filename - Filename to save the image to
        title - Title of the plot
        
    Returns:
        Saves the image to `filename`
    """
    plt.clf()
    
    plt.plot( testing_losses.T, 'r', alpha=0.1 )
    plt.plot( np.mean( testing_losses, axis=0 ), color='red', linewidth=2, label="Average Loss on Testing Data" )
    
    plt.plot( training_losses.T, 'b', alpha=0.1 )
    plt.plot( np.mean( training_losses, axis=0 ), color='blue', linewidth=2, label="Average Loss on Training Data" )
    plt.xlabel( "Epoch" )
    plt.ylabel( f"{title}" )
    plt.legend()
    plt.title( title )
    plt.tight_layout()
    plt.savefig( filename )

