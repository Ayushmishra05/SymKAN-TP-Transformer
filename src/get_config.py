def get_config():
    return{
        "batch_size" :  8, 
        "num_epochs" : 20, 
        "lr" : 0.0001,
        "num_layers" : 6,
        "seq_len" : 44,
        "d_model" : 512, 
        "model_folder" : "weights", 
        "model_filename" : "tmodel_", 
        "preload" : None, 
        "num_heads" : 8,
        "d_ff" : 2048,
        "dropout" : 0.1,
        "vocab_size" : 5000
    }

