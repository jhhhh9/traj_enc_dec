"""
This module handles tasks related to the Keras models except the creation. 
Tasks such as the training and testing of the models are done here. 
"""

class ModelProcessor():
    """
    This class handles the general Keras model operations such as training and 
    prediction.
    """
    def model_train(self, model):
        """
        Trains a model 
        """
        return None 
        
        
    def model_predict(self, data_generator, model):
        """
        Predict 
        """
        prediction = model.predict_generator(data_generator)   