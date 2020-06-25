"""Main class. Assigns tasks to other classes within the project"""

from argparse import ArgumentParser 
import datetime 
import time 

from arg_processor import ArgProcessor
from dnn_model import STSeqModel
from file_reader import FileReader
from keras_data_generators import KerasFitGenerator, KerasPredictGenerator
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
    
    # Reads the input .npy files for the data and the topk trajectories 
    file_reader = FileReader()
    training_x = file_reader.read_npy(arg_processor.training_x_path)
    training_y = file_reader.read_npy(arg_processor.training_y_path)
    #validation_x = file_reader.read_npy(arg_processor.validation_x_path)
    #validation_y = file_reader.read_npy(arg_processor.validation_y_path) 
    #test_gt = file_reader.read_npy(arg_processor.test_gt_path)
    #test_q = file_reader.read_npy(arg_processor.test_q_path)
    topk_id = file_reader.read_npy(arg_processor.topk_id_path)
    topk_weights = file_reader.read_npy(arg_processor.topk_weights_path)

    """
    # Do the topk lookup to transform the cell IDs to the top-k nearest cell 
    # weights 
    traj_processor = TrajProcessor()
    training_x = traj_processor.all_traj_to_topk_x(training_x, topk_id, 
                                                   topk_weights)
    training_y = traj_processor.all_traj_to_topk_y(training_y, topk_id,
                                                   topk_weights)
    print(training_x.shape)
    print(training_y.shape)
    assert False 
    """
    
    # Create the fit generator 
    batch_size = arg_processor.batch_size
    fitgen = KerasFitGenerator(training_x, training_y, batch_size)
    
    # Creates the model 
    embedding_vocab_size = arg_processor.embedding_vocab_size
    embedding_size = arg_processor.embedding_size
    traj_repr_size = arg_processor.traj_repr_size
    gru_cell_size = arg_processor.gru_cell_size
    num_gru_layers = arg_processor.num_gru_layers
    gru_dropout_ratio = arg_processor.gru_dropout_ratio
    bidirectional = arg_processor.bidirectional
    use_attention = arg_processor.use_attention
    xshape = fitgen.X.shape
    k = topk_weights.shape[1]
    stseqmodel = STSeqModel(embedding_vocab_size, embedding_size,traj_repr_size,
                            gru_cell_size, num_gru_layers, gru_dropout_ratio, 
                            bidirectional, use_attention, xshape, k)
    
    
    # TEST START 
    predgen = KerasPredictGenerator(training_x, batch_size) 
    #model_processor = ModelProcessor() 
    #a = model_processor.model_predict(predgen, stseqmodel.model)
    a = stseqmodel.model.predict(predgen)
    # YOU WERE HERE 
    # PREDICTION WORKS. NOW DESIGN THE TRAINING  
    for x in a:
        print(x.shape)
    # TEST END  
    
    
    
    
    
    

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