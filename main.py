"""Main class. Assigns tasks to other classes within the project"""

from argparse import ArgumentParser 
import datetime 
import numpy as np 
import time 

from arg_processor import ArgProcessor
from dnn_model import STSeqModel
from file_reader import FileReader
from keras_data_generators import KerasFitGenerator, KerasPredictGenerator
from log_writer import LogWriter 
from model_processor import ModelProcessor
from resource_manager import set_gpu_resource
from traj_processor import TrajProcessor 

def main(): 
    # Read the ini file path argument 
    parser = ArgumentParser(description='inputs')
    parser.add_argument('--config', dest = 'config',
                        help='The path to the .ini config file. FORMAT: ' + 
                             'a string.')
    ini_path = parser.parse_args().config
    
    # Pass to ArgProcessor to read and process arguments 
    arg_processor = ArgProcessor(ini_path)
    is_training = arg_processor.is_training
    is_evaluating = arg_processor.is_evaluating
    
    # Set the GPU resource
    gpu_used = arg_processor.gpu_used 
    gpu_memory = arg_processor.gpu_memory
    set_gpu_resource(gpu_used, gpu_memory)
    
    file_reader = FileReader()
    model_processor = ModelProcessor()  
    
    topk_id = file_reader.read_npy(arg_processor.topk_id_path)
    topk_weights = file_reader.read_npy(arg_processor.topk_weights_path)
    log_writer = LogWriter()
    if is_training: 
        # Reads the input .npy files for the data and the topk trajectories 
        print("Reading training data...")
        training_x = file_reader.read_data(arg_processor.training_x_path)
        training_y = file_reader.read_data(arg_processor.training_y_path)
        validation_x = file_reader.read_data(arg_processor.validation_x_path)
        validation_y = file_reader.read_data(arg_processor.validation_y_path) 
        topk_weights = file_reader.read_data(arg_processor.topk_weights_path)
        
        # Create the fit generator 
        print("Creating data generators...")
        batch_size = arg_processor.batch_size
        train_gen = KerasFitGenerator(training_x, training_y, topk_weights, 
                                      batch_size) 
        val_gen = KerasFitGenerator(validation_x, validation_y, topk_weights,
                                    batch_size)
        
        # Create the model 
        print("Creating model...")
        embedding_size = arg_processor.embedding_size
        traj_repr_size = arg_processor.traj_repr_size
        gru_cell_size = arg_processor.gru_cell_size
        num_gru_layers = arg_processor.num_gru_layers
        gru_dropout_ratio = arg_processor.gru_dropout_ratio
        bidirectional = arg_processor.bidirectional
        use_attention = arg_processor.use_attention
        model_path = arg_processor.model_path
        k = topk_weights.shape[1]
        embedding_vocab_size = arg_processor.embedding_vocab_size
        if embedding_vocab_size is None:
            embedding_vocab_size = topk_weights.shape[0] + 1
        stseqmodel = STSeqModel(embedding_vocab_size, embedding_size,
                                traj_repr_size, gru_cell_size, num_gru_layers, 
                                gru_dropout_ratio, bidirectional, 
                                use_attention, k)
        
        # Train the model 
        print("Training model...")
        triplet_margin = arg_processor.triplet_margin
        epochs = arg_processor.epochs 
        output_directory = arg_processor.output_directory
        patience = arg_processor.patience
        train_start = time.time()
        model_processor.model_train(stseqmodel.model, epochs,  train_gen, val_gen, 
                                    triplet_margin, output_directory, patience,
                                    model_path)
        train_time = time.time() - train_start 
        log_writer.write_train_results(output_directory, training_x, training_y, 
                                       validation_x, validation_y, topk_id, 
                                       topk_weights, train_time)


    if is_evaluating:
        # Perform prediction on the model
        # Get the data 
        print("Reading test dat...")
        test_gt = file_reader.read_npy(arg_processor.test_gt_path)
        test_q = file_reader.read_npy(arg_processor.test_q_path)
        
        
        # Load the model 
        print("Loading model...")
        model_path = arg_processor.model_path
        triplet_margin = arg_processor.triplet_margin
        pred_model = model_processor.load_model(model_path, triplet_margin)
        encoder = pred_model.get_layer('model_1')
        ks = arg_processor.ks
        use_mean_rank = arg_processor.use_mean_rank
        predict_batch_size = arg_processor.predict_batch_size
        
        print("Evaluating model...")
        predict_start = time.time()
        results = model_processor.model_evaluate(encoder, test_q, test_gt, ks,
                                                 use_mean_rank, 
                                                 predict_batch_size)
        predict_time = time.time() - predict_start
        
        # Write the results to a file 
        output_directory = arg_processor.output_directory
        log_writer.write_eval_results(output_directory, test_gt, test_q, 
                                     predict_time, ks, results)
        # Also make a copy of the input .ini file 
    log_writer.copy_ini_file(ini_path, output_directory)
    
if __name__ == "__main__":
    start_dt = datetime.datetime.now()
    start_t = time.time() 
    print("START DATETIME")
    print(start_dt)
    main()
    end_dt = datetime.datetime.now()
    end_t = time.time()
    print("END DATETIME")
    print(end_dt)
    print("Total time: " + str(end_t - start_t))