# traj_enc_dec

Call the script by providing an .ini file containing the arguments listed below (case sensitive):

```
[MODE]
IsTraining              = <Boolean. Determines if you want to train a new model>
IsEvaluating            = <Boolean. Determines if you want to evaluate an existing model>

[DIRECTORY]
TrainingXPath           = <String. Path to the X data (input data) for the training>
TrainingYPath           = <String. Path to the Y data (ground truth data) for the training>
ValidationXPath         = <String. Path to the X data (input data) for the validation>
ValidationYPath         = <String. Path to the Y data (ground truth data) for the validation>
TestGTPath              = <String. Path to the ground truth data for the testing. Refer to the paper for the meaning of ground truth and query>
TestQPath               = <String. Path to the query data for the testing>
TopKIDPath              = <String. Path to the Top-k ID path. Output by the data processor>
TopKWeightsPath         = <String. Path to the Top-k weights path. Output by the data procesor>
OutputDirectory         = <String. Path to where the training and/or evaluation logs and results will be written to>

[TRAINING]
ModelPath               = <String. In training mode, this is the path for the model to be saved to. In evaluating mode, this is the past the model will be loaded from>
BatchSize               = <Integer. Batch size for the training>
TripletMargin           = <Float. Margin for the triplet loss>
Epochs                  = <Integer. How many epochs for the training>
Patience                = <Integer. Relevant to early stopping. If the val_loss does not decrease after this amount of steps, stop the training>
LossWeights             = <List. List of three integers determining the weights of the three losses. Default value is [1,1,1]>

[MODEL]
GRUCellSize             = <Integer. Size of the GRU cell>
NumGRULayers            = <Integer. Number of GRU layers>
GRUDropoutRatio         = <Float. Dropout ratio for the GRU layers>
EmbeddingSize           = <Integer. Size of the cell embedding for the trajectory points>
EmbeddingVocabSize      = <Integer or None. The size of the embedding vocabulary, i.e. how many cells are there in the dataset. It is heavily recommended to leave this a None>
TrajReprSize            = <Integer. Trajectory representation size>
Bidirectional           = <Boolean. Whether or not to use bidirectional GRU>
UseAttention            = <Boolean. Unused. Leagve it as True>

[PREDICTION]
KS                      = <List. List of integers for the top-k hit rate experiment (omitted from paper)>
PredictBatchSize        = <Integer. Batch size for the prediction>
UseMeanRank             = <Boolean. Whether or not to perform the mean rank experiment. This is the result reported in the paper)

[GPU]
GPUUsed                 = <List. List of integers determining which GPU to use for the model. Indexing starts from 0>
GPUMemory               = <Integer. How many Megabytes to be used for the model> 
```     