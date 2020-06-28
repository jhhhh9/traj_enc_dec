"""
This module handles tasks related to the Keras models except the creation. 
Tasks such as the training and testing of the models are done here. 
"""

import copy 
import tensorflow as tf 
import tensorflow.keras.backend as K 

class ModelProcessor():
    """
    This class handles the general Keras model operations such as training and 
    prediction.
    """
    def model_train(self, model, epochs, train_generator, val_generator, 
                    triplet_margin):
        """
        Trains the provided model.
        
        Args:
            model: (keras model) The model to be trained 
            epochs: (int) The number of epochs for the training 
            train_generator: (keras generator) The data generator for the 
                              training.
            val_generator: (keras generator) The data generator for the 
                            validation. 
            triplet_margin: (float) Margin for the triplet loss 
        """
        model.compile(optimizer = "adam", 
                      loss = [self.repr_loss(triplet_margin),
                              self.point2point_loss,
                              self.patt_loss])
        model.fit(train_generator, validation_data = val_generator,
                  epochs = epochs) 


    def model_predict(self, model, predict_generator):
        """
        Use the provided model to do predictions
        
        Args:
            model: (keras model) The model to be used for the prediction 
            predict_generator: (keras generator) The data generator for the 
                                prediction data
            
        Returns 
            A numpy array containing the output prediction 
        """
        print("MODEL PREDICT NOT IMPLEMENTED YET")
        assert False 
        predictions = model.predict(predict_generator)
        print(predictions.shape)
        # TEST HERE 
        input("====")
        return predictions

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
        y_true_weights = y_true[:,:,:-2]
        
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
        
        # 'y_pred_patt' shape (batch_size, trg_traj_len, 3)
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
        
        
    def model_predict(self, model, data_generator):
        """
        Predict 
        """
        return None 
        #prediction = model.predict_generator(data_generator)