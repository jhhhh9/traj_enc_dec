"""
This module handles tasks related to the Keras models except the creation. 
Tasks such as the training and testing of the models are done here. 
"""

from keras import models 
from scipy.spatial import KDTree 
from tensorflow.keras import callbacks 
import copy 
import numpy as np 
import tensorflow as tf 
import tensorflow.keras.backend as K 

class ModelProcessor():
    """
    This class handles the general Keras model operations such as training and 
    prediction.
    """
    def model_train(self, model, epochs, train_generator, val_generator, 
                    triplet_margin, output_directory, patience, model_path):
        """
        Trains the provided model and save the best performing model. 
        
        Args:
            model: (keras model) The model to be trained 
            epochs: (int) The number of epochs for the training 
            train_generator: (keras generator) The data generator for the 
                              training.
            val_generator: (keras generator) The data generator for the 
                            validation. 
            triplet_margin: (float) Margin for the triplet loss
            output_directory: (string) Output directory to save the best model 
            checkpoint_model: (boolean) Whether or not the model checkpointing 
                               is used. 
            patience: (integer) Terminate training if the model's performance 
                       does not improve after this many epochs. 
            model_path: (string) The directory where the model checkpoint will 
                         be output to. 
        """
        # Create the callbacks. 
        # EarlStopping
        all_callbacks = []
        early_stopping = callbacks.EarlyStopping(monitor = 'val_loss', 
                                                 min_delta = 0, 
                                                 patience = patience,
                                                 restore_best_weights = True)
        all_callbacks.append(early_stopping)
        
        # ModelCheckpoint 
        model_checkpoint = callbacks.ModelCheckpoint(filepath = model_path,
                                                     monitor = "val_loss",
                                                     save_best_only = True, 
                                                     save_weights_only=False,
                                                     mode='min')
        all_callbacks.append(model_checkpoint)

        # Compile and train 
        model.compile(optimizer = "adam", 
                      loss = [self.repr_loss(triplet_margin),
                              self.point2point_loss,
                              self.patt_loss])
        model.fit(train_generator, validation_data = val_generator,
                  epochs = epochs, callbacks = all_callbacks) 


    def model_evaluate(self, model, all_q, all_gt, ks, use_mean_rank,
                       predict_batch_size):
        """
        Evaulate the model's performance.
        
        Args:
            model: (keras model) The model to be used for the prediction 
            all_q: (numpy array) Array containing all query data 
            all_gt: (numpy array) Array containing all ground truth data 
            ks: (list of integers) Top-k's for the prediction
            use_mean_rank: (boolean) Whether or not to report the mean predicted
                            rank. 
            predict_batch_size: (integer) The batch size for the prediction 
        
        Returns: 
            The hit rates for all given k 
        """
        # Remove duplicate ground truths and create dictionaries to find out 
        # the matching ground truth given a query. We need this because we 
        # cannot use the trajectory IDs in the KDtree we're going to create; 
        # we can only use the position of the trajectories we feed into the 
        # KDTree. So, we need to find a way to map the ID to the positions. 
        id_to_pos_dict = {}
        gt_dedup = []
        i = 0
        for gt in all_gt:
            if gt[0] not in id_to_pos_dict:
                id_to_pos_dict[gt[0]] = i
                gt_dedup.append(gt[1])
                i += 1
        
        # Smothing out the jagged q and gt traj arrays 
        q_array = np.array([x[1] for x in all_q])
        q_ids = [x[0] for x in all_q]
        gt_dedup = np.array(gt_dedup)
        max_gt_len = max([len(x) for x in gt_dedup])
        max_q_len = max([len(x) for x in q_array])
        max_len = max([max_gt_len, max_q_len])
        gt_dedup = self.__pad_jagged_array(gt_dedup, max_len)
        q_array = self.__pad_jagged_array(q_array, max_len)
        
        # Perform the prediction for the ground truth   
        if predict_batch_size == 0:
            prediction_gt = model.predict(gt_dedup)
        else:
            prediction_gt = model.predict(gt_dedup,batch_size=predict_batch_size)
        
        # Flatten the representation for each trajectory in  gt 
        gt_shape = prediction_gt.shape
        prediction_gt = prediction_gt.reshape((gt_shape[0], 
                                               gt_shape[1] * gt_shape[2]))
        
        # Builds the KDtree out of the ground truth trajectories 
        gt_tree = KDTree(prediction_gt)
        
        # Iterate through each of the query trajectories to query the KDTree 
        # First case is the top-k ranking. 
        if len(ks) > 0:
            all_k_hit = []
            all_rank = []
            for i in range(len(q_array)):
                # Get the q's representation 
                print("Evaluating trajectory %d out of %d." % (i+1,len(q_array)))
                one_q = model.predict(np.array([q_array[i]])) 
                one_q_shape = one_q.shape
                one_q = one_q.reshape((one_q_shape[0], 
                                       one_q_shape[1] * one_q_shape[2]))
                                       
                # Get the ID and find the position of the corresponding ground 
                # truth trajectory in the KDTree 
                q_id = q_ids[i]
                gt_pos = id_to_pos_dict[q_id]
                
                if use_mean_rank:
                    q_knn = gt_tree.query(one_q, k = len(prediction_gt))[1][0]
                else:
                    q_knn = gt_tree.query(one_q, k = max(ks))[1][0]
                
                # Check if the top-k hits are achieved 
                k_hit = np.array([gt_pos in q_knn[:x] for x in ks])
                all_k_hit.append(k_hit)
                
                # Get the actual rank if use_mean_rank is true 
                if use_mean_rank:
                    rank = np.where(q_knn==gt_pos)[0][0] + 1
                    all_rank.append(rank) 
            all_k_hit = np.array(all_k_hit) 
            all_hit_rates = np.sum(all_k_hit, axis=0) / all_k_hit.shape[0]
            mean_rank = None 
            if use_mean_rank:
                mean_rank = sum(all_rank)/len(all_rank) 
        return [all_hit_rates, mean_rank]


    def load_model(self, model_path, triplet_margin):
        """
        Loads the saved model 
        
        Args:
            model_path: (string) Path to the saved model 
        
        Returns:
            The loaded Keras model 
        """
        def triplet_loss(y_true, y_pred):
            anc = y_pred[:,0,:,:]
            pos = y_pred[:,1,:,:]
            neg = y_pred[:,2,:,:]
            pos_dist = K.sqrt(K.sum(K.square(anc-pos), axis=-1,keepdims=False))
            neg_dist = K.sqrt(K.sum(K.square(anc-neg), axis=-1,keepdims=False))
            const = K.constant(margin, dtype='float32')
            dist = pos_dist - neg_dist + const 
            dist = K.maximum(dist, 0)
            dist_sum = K.mean(dist, 1)
            return dist_sum
        
        repr_loss = self.repr_loss(triplet_margin)
        p2p = self.point2point_loss
        patt_loss = self.patt_loss
        model = models.load_model(model_path, 
                                  custom_objects={'repr_loss':repr_loss,
                                                  'point2point_loss':p2p,
                                                  'patt_loss':patt_loss,
                                                  'triplet_loss':triplet_loss})
        return model 


    def repr_loss(self, margin):
        """
        The representation loss takes the 
        
        Args:
            y_true: (whatever) Supposed to be the ground truth values, but this 
                     is not used in the representation loss as this loss 
                     relies entirely on the model output. 
            y_pred: (keras tensor) Keras tensor of shape 
        """
        def triplet_loss(y_true, y_pred):
            """
            Args:
                y_true: (whatever) Supposed to be the ground truth values, but 
                         this is not used in the representation loss as this 
                         loss relies entirely on the model output. 
                y_pred: (keras tensor) Keras tensor of shape 
                        (batch_size,3,traj_len, gru_cell_size * directions)
            """
            # Split y_pred, consisting of the output from the model 
            # 'y_pred' shape (batch_size,3,traj_len, gru_cell_size * directions)
            # 'anc' shape (batch_size, traj_len, gru_cell_size * directions). 
            # 'pos' shape (batch_size, traj_len, gru_cell_size * directions). 
            # 'neg' shape (batch_size, traj_len, gru_cell_size * directions). 
            anc = y_pred[:,0,:,:]
            pos = y_pred[:,1,:,:]
            neg = y_pred[:,2,:,:]
            
            # Form the loss function 
            # 'posdist' shape (batch_size, traj_len)
            # 'negdist' shape (batch_size, traj_len)
            pos_dist = K.sqrt(K.sum(K.square(anc-pos), axis=-1,keepdims=False))
            neg_dist = K.sqrt(K.sum(K.square(anc-neg), axis=-1,keepdims=False))
            const = K.constant(margin, dtype='float32')
            dist = pos_dist - neg_dist + const 
            dist = K.maximum(dist, 0)
            dist_sum = K.mean(dist, 1)
            return dist_sum
        return triplet_loss 
        
    
    def point2point_loss(self, y_true, y_pred):
        """
        Calculates the loss function that compares the predicted vs true value 
        on a trajectory-point-per-trajectory-point basis. 
        
        Args:
            y_true: Actual point2point input 
            y_pred: Predicted point2point prediction 
        """
        # 'y_pred' shape (batch_size, trg_traj_len, k+1)
        # 'y_weights' shape (batch_size, traj_len, k). 
        y_pred_weights = y_pred[:,:,:-1]
        
        # 'y_true' shape (batch_size, traj_len, k+2)
        y_true_weights = y_true
        
        # Create the boolean mask to filter out nan values in y_true_weights 
        bool_finite = tf.math.is_finite(y_true_weights) 
        
        # Apply mask to both y_true_weights and y_pred_weights
        y_pred_weights = tf.boolean_mask(y_pred_weights, bool_finite)
        y_true_weights = tf.boolean_mask(y_true_weights, bool_finite)
        
        # TODO: IMPLEMENT NLL LOSS 
        dist = K.sqrt(K.mean(K.square(y_true_weights - y_pred_weights), 
                       axis=-1,keepdims=False))  
        #dist = K.mean(K.categorical_crossentropy(y_true_weights, y_pred_weights,
        #                                  axis = -1))
        return dist 
        
        
    def patt_loss(self, y_true, y_pred):
        """
        Calculates the loss function that compares the predicted vs true value
        spatial and temporal features on a pattern-by-pattern basis. 
        
        Args:
            y_true: Actual pattern input 
            y_pred: Predicted pattern prediction 
        """
        # 'y_true_patt' shape (batch_size, trg_traj_len, k+2)
        # 'y_true_patt_s' shape (batch_size, traj_len, 1). 
        # 'y_true_patt_t' shape (batch_size, traj_len, 1). 
        y_true_patt_s = y_true[:,:,-2]
        y_true_patt_t = y_true[:,:,-1]
        
        # 'y_pred_patt' shape (batch_size, trg_traj_len, 2)
        # 'y_pred_patt_s' shape (batch_size, traj_len, 1). 
        # 'y_pred_patt_t' shape (batch_size, traj_len, 1). 
        y_pred_patt_s = y_pred[:,:,0]
        y_pred_patt_t = y_pred[:,:,1] 
        
        # Create the boolean mask to filter out nan values in
        bool_finite_s = tf.math.is_finite(y_true_patt_s) 
        bool_finite_t = tf.math.is_finite(y_true_patt_t) 
        
        
        # Apply mask to both y_true_weights and y_pred_weights
        y_true_patt_s = tf.boolean_mask(y_true_patt_s, bool_finite_s)
        y_true_patt_t = tf.boolean_mask(y_true_patt_t, bool_finite_t)
        y_pred_patt_s = tf.boolean_mask(y_pred_patt_s, bool_finite_s)
        y_pred_patt_t = tf.boolean_mask(y_pred_patt_t, bool_finite_t)
        
        # Calculate spatial and temporal loss 
        dist_s = K.mean(K.abs(y_true_patt_s - y_pred_patt_s))
        dist_t = K.mean(K.abs(y_true_patt_t - y_pred_patt_t))
        dist_st = dist_s + dist_t
        return dist_st
        
        
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