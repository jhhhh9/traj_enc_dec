"""
This module handles tasks related to the Keras models except the creation. 
Tasks such as the training and testing of the models are done here. 
"""

import tensorflow.keras.backend as K 

class ModelProcessor():
    """
    This class handles the general Keras model operations such as training and 
    prediction.
    """
    def model_train(self, model, train_generator, val_generator, 
                    triplet_margin):
        """
        Trains the provided model.
        
        Args:
            model: (keras model) The model to be trained 
            train_generator: (keras generator) The data generator for the 
                              training.
            val_generator: (keras generator) The data generator for the 
                            validation. 
            triplet_margin: (float) Margin for the triplet loss 
        """
        model.compile(optimizer = "sgd", 
                      loss = [self.repr_loss(triplet_margin)])
        ## YOU WERE HERE 
        ## TRIPLET MARGIN LOSS DONE... I THINK, NEEDS CHECKING. 
        ## TEST IT FIRST, SEE IF IT WORKS FINE. 
        ## THEN, DESIGN THE OTHER 2 LOSS FUNCTIONS AND THEN TEST THEM ONE-BY-ONE
        model.fit(train_generator, validation_data = val_generator,
                  use_multiprocessing = True, workers = 4) 


    def repr_loss(self, margin):
        """
        The representation loss takes the 
        
        Args:
            y_true
            y_pred
        """
        def triplet_loss(y_true, y_pred):
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


    def point2point_loss(self, y_true, y_prd):
        """
        """
        return None 
        
        
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