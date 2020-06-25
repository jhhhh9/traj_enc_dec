"""Class that handles the reading of files"""

import numpy as np 

class FileReader():
    """Class that handles the reading of files"""
    
    def read_npy(self, path):
        """
        Reads .npy files 
        
        Inputs:
            path: (string) The path to the .npy files 
            
        Returns: 
            A numpy array containing the data of the .npy files 
        """
        return np.load(path, allow_pickle = True)
        