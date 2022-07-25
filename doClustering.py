import datetime, time
from argparse import ArgumentParser 
from arg_processor import ArgProcessor
from resource_manager import set_gpu_resource
from file_reader import FileReader
from model_processor import ModelProcessor

def main():
    # Read the ini file path argument 
    parser = ArgumentParser(description='inputs')
    parser.add_argument('--config', dest='config', default="arg.ini",
                        help='The path to the .ini config file. FORMAT: ' +
                             'a string.')
    ini_path = parser.parse_args().config

    # Pass to ArgProcessor to read and process arguments 
    arg_processor = ArgProcessor(ini_path)
    # Set the GPU resource
    gpu_used = arg_processor.gpu_used 
    gpu_memory = arg_processor.gpu_memory
    set_gpu_resource(gpu_used, gpu_memory)

    file_reader = FileReader()
    model_processor = ModelProcessor()  
    
    # Load the model 
    print("Loading model...")
    model_path = arg_processor.model_path
    triplet_margin = arg_processor.triplet_margin
    cluster_model = model_processor.load_model(model_path, triplet_margin)
    
    # Load the data and label
    training_x = file_reader.read_data(arg_processor.test_x_path) # 
    training_label = file_reader.read_data("data/1_test_label.npy") # 类别
    training_x = training_x[::16]
    training_label = training_label[::16]
    
    encoder = cluster_model.get_layer('model_1')  #
    # encoder.model()
    predict_batch_size = arg_processor.predict_batch_size
    centroids, error_total, nmi, ari, inertia_start, inertia_end, n_iter, labels = \
        model_processor.model_evaluate_cluster(encoder, training_x, training_label, predict_batch_size, 16, save=False, epoch=arg_processor.model_path)
        # model_processor.model_evaluate_cluster(encoder, training_x, training_label, predict_batch_size, 19)
    print(
        "cluster error_total:{} nmi:{} ari:{} inertia_start:{} inertia_end:{} n_iter:{}******".format(
            error_total, round(nmi, 4), round(ari, 4), round(inertia_start, 4), round(inertia_end, 4), n_iter))

    # old_label = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
    # old_label = [0, 0, 1, 8, 2, 2, 9, 3, 4, 4, 10, 5, 6, 6, 7, 11]
    # old_label = [0, 0, 0, 0, 1, 1, 1, 1,
    #              2, 2, 2, 2, 3, 3, 3, 3,
    #              4, 4, 4, 4,
    #              5, 5, 6, 6,
    #              7, 7, 8, 8,
    #              9, 9, 9, 9, 10, 10, 10, 10,
    #              11, 11, 11, 11, 12, 12, 12, 12,
    #              13, 13, 14, 14,
    #              15, 15, 15, 15, 16, 16, 16, 16,
    #              17, 17, 17, 17, 18, 18, 18, 18]
    # # old_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # old_label = [0, 0, 1, 1,
    #              2, 2, 3, 3,
    #              4, 4, 5, 5,
    #              6, 6, 7, 7,
    #              8, 8, 9, 9,
    #              10, 10, 11, 11,
    #              12, 12, 13, 13,
    #              14, 14, 15, 15,
    #              16, 16, 17, 17,
    #              18, 18, 19, 19]
    old_label = [0, 0, 1, 1,
                 2, 2, 3, 3,
                 4, 4, 5, 5,
                 6, 6, 7, 7,
                 8, 8, 9, 9,
                 10, 10, 11, 11,
                 # 12, 12, 13, 13,
                 12, 12, 13, 13,
                 14, 14, 15, 15]
    with open("data/train_" + str(arg_processor.gpu_used[0]+1) + ".csv", 'w') as wf:
        for idx, label in enumerate(labels):
            
            # if i in data_id:
            wf.write(str(idx+1)+","+str(training_label[idx])+","+str(label)+"\n")
            # else:
            #     wf.write(str(i+1)+","+str(ori_labels_dict[i])+","+str(-1)+"\n")
    print("exp path is: {}".format(arg_processor.model_path))
    
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
