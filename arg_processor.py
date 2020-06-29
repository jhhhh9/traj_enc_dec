"""This module handles the processing of the input arguments to the program"""

import ast 
import configparser
import os 

class ArgProcessor():
    """Class that handles the .ini arguments"""
    
    def __init__(self, ini_path):
        """
        Reads the arguments from the input .ini file and checks their validity
        
        Args:
            ini_path: The path to the input .ini file 
        """
        # Read the .ini file 
        config = configparser.ConfigParser()
        config.read(ini_path)
        
        # Reads the arguments 
        self.training_x_path = config['DIRECTORY']['TrainingXPath']
        self.training_y_path= config['DIRECTORY']['TrainingYPath']
        self.validation_x_path = config['DIRECTORY']['ValidationXPath']
        self.validation_y_path = config['DIRECTORY']['ValidationYPath']
        self.test_gt_path = config['DIRECTORY']['TestGTPath']
        self.test_q_path = config['DIRECTORY']['TestQPath']
        self.topk_id_path = config['DIRECTORY']['TopKIDPath']
        self.topk_weights_path = config['DIRECTORY']['TopKWeightsPath']
        self.output_directory = config['DIRECTORY']['OutputDirectory']
        
        self.batch_size = int(config['TRAINING']['BatchSize'])
        self.triplet_margin = float(config['TRAINING']['TripletMargin'])
        self.epochs = int(config['TRAINING']['Epochs'])
        
        self.gru_cell_size = int(config['MODEL']['GRUCellSize'])
        self.num_gru_layers = int(config['MODEL']['NumGruLayers'])
        self.gru_dropout_ratio = float(config['MODEL']['GRUDropoutRatio'])
        self.embedding_size = int(config['MODEL']['EmbeddingSize'])
        self.embedding_vocab_size = int(config['MODEL']['EmbeddingVocabSize'])
        self.traj_repr_size = int(config['MODEL']['TrajReprSize'])
        self.bidirectional = bool(config['MODEL']['Bidirectional'])
        self.use_attention = bool(config['MODEL']['UseAttention'])
        
        self.ks = ast.literal_eval(config['PREDICTION']['KS'])
        self.ks.sort()
        
        # Check if all the inputs files are valid files 
        if not os.path.isfile(self.training_x_path):
            raise IOError("'" + self.training_x_path + "' is not a valid file")
        if not os.path.isfile(self.training_y_path):
            raise IOError("'" + self.training_y_path + "' is not a valid file")
        if not os.path.isfile(self.validation_x_path):
            raise IOError("'" + self.validation_x_path + "' is not a valid file")
        if not os.path.isfile(self.validation_y_path):
            raise IOError("'" + self.validation_y_path + "' is not a valid file")
        if not os.path.isfile(self.test_gt_path):
            raise IOError("'" + self.test_gt_path + "' is not a valid file")
        if not os.path.isfile(self.test_q_path):
            raise IOError("'" + self.test_q_path + "' is not a valid file")
        if not os.path.isfile(self.topk_id_path):
            raise IOError("'" + self.topk_id_path + "' is not a valid file")
        if not os.path.isfile(self.topk_weights_path):
            raise IOError("'" + self.topk_weights_path + "' is not a valid file")
        if not os.path.isdir(self.output_directory):
            print("Output director does not exist. Creating...")
            os.makedirs(self.output_directory)
            
        # Check numerical features 
        if self.batch_size <= 0:
            raise ValueError("BatchSize must be greater than 0")
        if self.gru_cell_size <= 0:
            raise ValueError("GRUCellSize must be greater than 0")
        if self.num_gru_layers <= 0:
            raise ValueError("NumGruLayers must be greater than 0")
        if self.gru_dropout_ratio <= 0 or self.gru_dropout_ratio >= 1:
            raise ValueError("GRUDropoutRatio must be between 0 and 1 exclusive")
        if self.traj_repr_size <= 0:
            raise ValueError("EmbeddingSize must be greater than 0")
        if self.embedding_vocab_size <= 0:
            raise ValueError("EmbeddingVocabSize must be greater than 0")
        for k in self.ks:
            if not isinstance(k, int):
                raise ValueError("The values in KS must all be integers")
            if k < 1:
                raise ValueError("K must not be 0 or negative") 
        