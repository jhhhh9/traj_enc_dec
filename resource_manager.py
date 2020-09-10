import os 

def set_gpu_resource(gpus, gpu_memory_mb):
    """
    Set which GPUs to use and also the maximum memory to use for each 
    
    Args:
        gpus: A list of gpus to be used 
        gpu_memory_mb: Upper limit of megabytes of memory for each GPU 
    """
    
    if len(gpus) == 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        gpus_ = []
        for gpu in gpus:
            if gpu in [0,1]:
                gpus_.append(gpu)
            else:
                print("Warning! " + str(gpu) + " is not a valid GPU no. This \
                       input will be ignored.")
       
        gpu_str = ""
        for gpu_ in gpus_:
            gpu_str += str(gpu_)
            gpu_str += ","
        gpu_str = gpu_str[:-1]
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        import tensorflow as tf
        all_device_gpus = tf.config.experimental.list_physical_devices('GPU')
        for device_gpu in all_device_gpus:
            tf.config.experimental.set_virtual_device_configuration(device_gpu,
                [tf.config.experimental.VirtualDeviceConfiguration \
                (memory_limit=gpu_memory_mb)]) 