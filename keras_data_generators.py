"""This module handles the different keras data generators"""

import copy 
from tensorflow.keras.utils import Sequence
import numpy as np 

from traj_processor import TrajProcessor

class KerasFitGenerator(Sequence):
    """Generator for the training and validation"""
    def __init__(self, X, y, topk_weights, batch_size):
        self.X = copy.deepcopy(X)
        self.y = copy.deepcopy(y) 
        self.topk_weights = topk_weights
        self.batch_size = batch_size
        
        # Shuffle the dataset so that when we pick the negative samples later 
        # it will also be randomized 
        # 打乱data
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
        self.X = self.X[indices]
        self.y = self.y[indices]
        
        
    def __len__(self):
        return int(np.ceil(self.X.shape[0] / self.batch_size))
        
        
    def on_epoch_end(self):
        # 每一个epoch都打乱一次
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
        self.X = self.X[indices]
        self.y = self.y[indices]
        
    
    def __getitem__(self, index):
        # Get current batch 
        # 获取当前batch数据
        batch_end = (index+1)*self.batch_size
        if batch_end > len(self.X):
            batch_end = len(self.X)
        X = self.X[index*self.batch_size:batch_end]
        y = self.y[index*self.batch_size:batch_end]
        
        # Get next batch. If next batch is not a full batch, get the first 
        # 定义下一个batch的数据
        next_batch_end = (index+2)*self.batch_size
        if next_batch_end > len(self.X):
            # 从头获取 [128,4]
            X_next = self.X[:self.batch_size]
        else:
            X_next_start = (index + 1) * self.batch_size
            X_next_end = (index + 2) * self.batch_size
            X_next = self.X[X_next_start : X_next_end]

        # We use the next batch to get a sample of negative trajectories. 
        # This works the same as randomizing the negative trajectories simply 
        # because we randomize the data after every epoch.
        # 对于三元组loss，需要负样本，负样本就是下一个batch的数据
        # [128,]
        X_neg = X_next[:len(X),0] # 只取第0维
        # [128,1]
        X_neg = X_neg.reshape(len(X_neg), 1)
        # [128,3]
        X_1 = np.concatenate((X[:,:2], X_neg), axis=1) # X正样本只取前两维，并且和负样本拼接
        # [128,2]
        X_2 = X[:,2:]   # 后2维
        # [128,5]
        X = np.concatenate((X_1, X_2), axis=1) # 相当于中间插入了负样本
        
        # Preprocessing the data 
        # First, pad X so that it's no longer a jagged array 
        X = self.__pad_jagged_array(X) 
        
        # Splits y into three 
        # y_traj consists of the trajectory after the topk lookup 
        # Shape is (num_traj, traj_len, k)
        traj_len = X.shape[2] # 上一步求得的maxlen
        # 取出第一维度
        # y:[batch_size,3]
        # y_traj:[batch_size,]
        y_traj = y[:,0]
        y_traj = self.__lookup_topk(y_traj, self.topk_weights)
        y_traj = self.__pad_nan(y_traj, traj_len)
        
        # y_s_patt consists of the trajectory spatial pattern 
        # Shape is (num_traj, traj_len, 1) 
        y_s_patt = y[:,1]
        # todo: 这些是不同模式
        y_s_patt = self.__pad_nan(y_s_patt, traj_len)
        
        # y_s_patt consists of the trajectory temporal pattern 
        # Shape is (num_traj, traj_len, 1)  
        y_t_patt = y[:,2]
        y_t_patt = self.__pad_nan(y_t_patt, traj_len) 
        
        # Concatenate y_traj, y_s_patt, and y_t_patt
        # 拼接返回
        y = np.concatenate([y_traj, y_s_patt, y_t_patt], axis = 2)
        return X, y
        

    """
        #OLD GETITEM
        def __getitem__(self, index):
        batch_end = (index+1)*self.batch_size
        if batch_end > len(self.X):
            batch_end = len(self.X)
        X = self.X[index*self.batch_size:batch_end]
        y = self.y[index*self.batch_size:batch_end]
        
        
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
    """


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
        # 将每个id的topk找出来并且每个序列变成[[...10个],[...10个],..轨迹长度]
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
        k = in_array[0].shape[-1] # 邻居数量
        
        # Do the padding by creating an array of nan in the intended shape 
        # Then, we just copy the relevant values form in_array 
        # [batch_size, maxlen, k]
        final = np.empty((num_data, pad_len, k))
        final[:,:,:] = np.nan
        # final[batch_in_idx][轨迹序列中每个点][当前点的邻居数量] = 邻居data
        # 不满足k个的，别的是nan
        
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
        max_len = max([len(y) for x in in_array for y in x]) # 取所有维度数据的最长
        
        # Do the padding by creating an array of zeroes in the intended shape 
        # Then, we can perform addition to fill the relevant values in this 
        # array with the values from in_array 
        # 【batchsize, 5, maxlen, 1】
        final = np.zeros((num_data,num_data_inner,max_len,1))
        
        # final[current in batch id][维度][:data的长度] += data
        for i in range(len(in_array)):
            for j, row in enumerate(in_array[i]):
                final[i][j, :len(row)] += row 
        return final 
        
        
class KerasPredictGenerator(Sequence):
    """Generator for the prediction""" 
    def __init__(self, X, batch_size, traj_len):
        self.X = X
        self.X = np.array([x[1] for x in self.X]) 
        self.X = self.__pad_jagged_array(self.X, traj_len) 
        self.X = self.X[:,:,0]
        self.batch_size = batch_size 
        
        
    def __len__(self):
        return int(np.ceil(self.X.shape[0] / self.batch_size))
        
        
    def on_epoch_end(self):
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
        self.X = self.X[indices]
        
    
    def __getitem__(self, index):
        batch_end = (index+1)*self.batch_size
        if batch_end > len(self.X):
            batch_end = len(self.X)
            
        X = self.X[index*self.batch_size:batch_end]
        return X 
        
        
    def __pad_jagged_array(self, in_array, traj_len):
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
        
        # Do the padding by creating an array of zeroes in the intended shape 
        # Then, we can perform addition to fill the relevant values in this 
        # array with the values from in_array 
        final = np.zeros((num_data,traj_len,1))
        for j, row in enumerate(in_array):
            final[j, :len(row)] += row 
        return final 