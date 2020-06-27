"""
This module handles tasks related to the Keras models except the creation. 
Tasks such as the training and testing of the models are done here. 
"""

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
        #model.compile(optimizer = "sgd", 
        #              loss = [self.repr_loss(triplet_margin)])
        model.compile(optimizer = "sgd", 
                      loss = [self.point2point_loss])
        ## YOU WERE HERE 
        ## FINISH AND TEST THE LOSSES ONE-BY-ONE-BY-O 
        ## ONCE ALL ARE TESTED. USE THEM ALL AT ONCE. 
        ## CHECK IF ANY WARNINGS PERSIST ONCE YOU USED ALL THREE LOSSES 
        model.fit(train_generator, validation_data = val_generator,
                  epochs = epochs) 


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
            dist_sum = K.sum(dist, 1)
            return dist_sum
        return triplet_loss 


    def point2point_loss(self, y_true, y_pred):
        """
        """
        # 'y_pred' shape (batch_size, trg_traj_len, k+1)
        # 'y_weights' shape (batch_size, traj_len, k). 
        y_pred_weights = y_pred[:,:,:-1]
        
        # 'y_true' shape (batch_size, traj_len, k)
        y_true_weights = y_true[:,:,:-2]
        
        # Create the boolean mask to filter out nan values in y_true_weights 
        bool_finite = tf.math.is_finite(y_true_weights)
        
        # Apply mask to both y_true_weights and y_pred_weights
        y_pred_weights = tf.boolean_mask(y_pred_weights, bool_finite)
        y_true_weights = tf.boolean_mask(y_true_weights, bool_finite)
        
        # TEST START 
        dist = K.sqrt(K.sum(K.square(y_true_weights - y_pred_weights), 
                       axis=-1,keepdims=False))  
        return dist 
        # TEST END 
        
        
    def patt_loss(self, y_true, y_pred):
        """
        """
        return None 
        
        
    def model_predict(self, model, data_generator):
        """
        Predict 
        """
        return None 
        #prediction = model.predict_generator(data_generator)