"""This module handles the different keras data generators"""

import copy 
import keras 
import numpy as np 

from traj_processor import TrajProcessor

class KerasFitGenerator(keras.utils.Sequence):
    """Generator for the training and validation"""
    def __init__(self, X, y, topk_weights, batch_size):
        self.X = copy.deepcopy(X)
        self.y = copy.deepcopy(y) 
        self.topk_weights = topk_weights
        self.batch_size = batch_size
        
    def __len__(self):
        return int(np.floor(self.X.shape[0] / self.batch_size))
        
        
    def on_epoch_end(self):
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
        self.X = self.X[indices]
        self.y = self.y[indices]
        
    
    def __getitem__(self, index):
        X = self.X[index*self.batch_size:(index+1)*self.batch_size]
        y = self.y[index*self.batch_size:(index+1)*self.batch_size]
        
        
        # Preprocessing the data 
        # First, pad X so that it's no longer a jagged array 
        X = self.__pad_jagged_array(X) 
        
        # Splits y into three 
        # y_traj consists of the trajectory after the topk lookup 
        # Shape is (num_traj, traj_len, k)
        traj_len = X.shape[2]
        y_traj = y[:,0]
        y_traj = self.__lookup_topk(y_traj, self.topk_weights)
        y_traj = self.__pad_nan(y_traj, traj_len)
        
        # y_s_patt consists of the trajectory spatial pattern 
        # Shape is (num_traj, traj_len, 1) 
        y_s_patt = y[:,1]
        y_s_patt = self.__pad_nan(y_s_patt, traj_len)
        
        # y_s_patt consists of the trajectory temporal pattern 
        # Shape is (num_traj, traj_len, 1)  
        y_t_patt = y[:,2]
        y_t_patt = self.__pad_nan(y_t_patt, traj_len) 
        
        # Concatenate y_traj, y_s_patt, and y_t_patt
        y = np.concatenate([y_traj, y_s_patt, y_t_patt], axis = 2)
        return X, y
        

    def __lookup_topk(self, in_array, topk_weights):
        """
        Given a numpy array consisting of all trajectories, where each 
        trajectory point is represented with a cell ID, perform a lookup to 
        get the top-k weights of the cells and return as a new numpy array 
        
        Args:
            in_array: (numpy array) Jagged array of shape 
                      (num_traj, traj_len, 1), which represents the 
                       trajectories to perform the lookup with. 
                       
        Returns:
            Array of shape (num_traj, traj_len, k) where k represents the 
            weight of each cell to its k-nearest cells 
        """
        new_array =[np.array([topk_weights[x[0]] for x in y]) for y in in_array]
        new_array = np.array(new_array)
        return new_array
        

    def __pad_nan(self, in_array, pad_len):
        """
        Given an array, pad every array in axis 1 (i.e. 2nd dimension) to the 
        length of the longest axis-1-array from the whole input_array. The 
        padding value is nan, the type of the elements is float and post-padding 
        is used. 
        
        Args:
            in_array: (numpy array) 4D numpy array. All the values within
                       the array must be a type in which arithmetic addition can 
                       be applied to. 
            pad_len: (integer or None) The length to pad each trajectory to. If 
                      None is provided, pad to the maximum trajectory length. 
        
        Returns:
            in_array after the padding. The padding turns a jagged array to a 
            non-jagged array, which can now be fed to the deep neural network 
            model. 
        """
        # Get some important variables from in_array shapes 
        num_data = in_array.shape[0]
        if pad_len is None:
            pad_len = max([len(x) for x in in_array])
        k = in_array[0].shape[-1]
        
        # Do the padding by creating an array of nan in the intended shape 
        # Then, we just copy the relevant values form in_array 
        final = np.empty((num_data, pad_len, k))
        final[:,:,:] = np.nan
        for i in range(len(in_array)):
            for j, row in enumerate(in_array[i]):
                final[i][j, :len(row)] = row 
        return final 


    def __pad_jagged_array(self, in_array):
        """
        Given an array, pad every array in axis 1 (i.e. 2nd dimension) to the 
        length of the longest axis-1-array from the whole input_array. The 
        padding value is 0, the type of the elements is float and post-padding 
        is used. 
        
        Args:
            in_array: (numpy array) 4D numpy array. All the values within
                       the array must be a type in which arithmetic addition can 
                       be applied to. 
        
        Returns:
            in_array after the padding. The padding turns a jagged array to a 
            non-jagged array, which can now be fed to the deep neural network 
            model. 
        """
        # Get some important variables from in_array shapes 
        num_data = in_array.shape[0]
        num_data_inner = in_array.shape[1]
        max_len = max([len(y) for x in in_array for y in x])
        
        # Do the padding by creating an array of zeroes in the intended shape 
        # Then, we can perform addition to fill the relevant values in this 
        # array with the values from in_array 
        final = np.zeros((num_data,num_data_inner,max_len,1))
        for i in range(len(in_array)):
            for j, row in enumerate(in_array[i]):
                final[i][j, :len(row)] += row 
        return final 
        
        
class KerasPredictGenerator(keras.utils.Sequence):
    """Generator for the prediction""" 
    def __init__(self, X, batch_size):
        self.X = X
        self.batch_size = batch_size 
        
        
    def __len__(self):
        return int(np.floor(self.X.shape[0] / self.batch_size))
        
        
    def on_epoch_end(self):
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
        self.X = self.X[indices]
        
    
    def __getitem__(self, index):
        X = self.X[index*self.batch_size:(index+1)*self.batch_size]
        # YOU WERE HERE 
        # EACH ITEM IN X IS AN ARRAY OF SIZE 2 CONTAINING THE TRAJ ID (STRING)
        # AND THEN THE TRAJECTORY ITSELF. 
        # GATHER ONLY THE TRAJECTORIES, CALL __PAD_JAGGED_ARRAY AND RETURN. 
        # AFTERWARDS, FINISH THE PREDICTION FUNCTION IN MODEL_PROCESSOR
        print(X.shape) 
        input("++++++++")
        return X 
        
        
    def __pad_jagged_array(self, in_array):
        """
        Given an array, pad every array in axis 1 (i.e. 2nd dimension) to the 
        length of the longest axis-1-array from the whole input_array. The 
        type of the elements is float and post-padding is used. 
        
        Args:
            in_array: (numpy array) 4D numpy array. All the values within
                       the array must be a type in which arithmetic addition can 
                       be applied to. 
        
        Returns:
            in_array after the padding. The padding turns a jagged array to a 
            non-jagged array, which can now be fed to the deep neural network 
            model. 
        """
        # Get important variables from in_array shapes 
        num_data = in_array.shape[0]
        num_data_inner = in_array.shape[1]
        max_len = max([len(y) for x in in_array for y in x])
        
        # Do the padding by creating an array of zeroes in the intended shape 
        # Then, we can perform addition to fill the relevant values in this 
        # array with the values from in_array 
        final = np.zeros((num_data,num_data_inner,max_len,1))

        for i in range(len(in_array)):
            for j, row in enumerate(in_array[i]):
                final[i][j, :len(row)] += row 
        return final 