"""This module handles the different keras data generators"""

import keras 
import numpy as np 

class KerasFitGenerator(keras.utils.Sequence):
    """Generator for the training and validation"""
    def __init__(self, X, y, batch_size):
        self.X = self.__pad_jagged_array(X)
        self.y = self.__pad_jagged_array(y)
        self.batch_size = batch_size
        assert self.X.shape[0] == self.y.shape[0], ("X and y have different " +
                                                    "number of data")
        
        
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
        return X, y 
        
        
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
        
        
class KerasPredictGenerator(keras.utils.Sequence):
    """Generator for the prediction""" 
    def __init__(self, X, batch_size):
        self.X = self.__pad_jagged_array(X)
        self.batch_size = batch_size 
        
        
    def __len__(self):
        return int(np.floor(self.X.shape[0] / self.batch_size))
        
        
    def on_epoch_end(self):
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
        self.X = self.X[indices]
        
    
    def __getitem__(self, index):
        X = self.X[index*self.batch_size:(index+1)*self.batch_size]
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