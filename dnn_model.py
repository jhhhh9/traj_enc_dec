"""This class handles the creation of the EncoderDecoder model"""

from tensorflow.keras.layers import Attention, Bidirectional, Concatenate, Dense
from tensorflow.keras.layers import Embedding, GRU, Input, Lambda, Masking 
from tensorflow.keras.layers import Permute, TimeDistributed
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K 
import numpy as np 

class StackedGRU():
    """
    Handles the Encoder part of the overall encoder-decoder model. The Encoder 
    takes as input the trajectory after the embedding is done and outputs the 
    hidden state of each layer as well as the feature vector representation of 
    each trajectory after being passed through the GRU layers contained within. 
    """
    
    def __init__(self, embedding_size, gru_cell_size, num_gru_layers, 
                 gru_dropout_ratio, bidirectional, encoder_h0):
        """
        Initializes the model. 
        
        Inputs:
            input: Shape is (batch_size, in_traj_len, embedding_size) This is 
                   the trajectory after each of its spatiotemporal cells have
                   been embedded. in_traj_len depends on the input tensor 
        Outputs 
            hn: Shape is (batch_size,num_gru_layers * num_directions,
                gru_cell_size). This is the hidden state of each of the GRU 
                layers 
            output: Shape is (batch_size, in_traj_len, gru_cell_size * 
                    num_directions). This is the encoded input trajectory. 
        Where: 
            - batch_size: Size of input batch 
            - in_traj_len: Length of the input trajectory 
            - embedding_size: The size of the embedding 
            - num_gru_layers: The number of GRU layers
            - num_directions: This is 1 if bidirectional is False. 2 if True. 
            - gru_cell_size: The size of the GRU cells 
            
        Arguments: 
            embedding_size: (integer) The size of the spatiotemporal cells embedding
            gru_cell_size: (integer) Size of the gru cels 
            num_gru_layers: (integer) How many GRus to use 
            gru_dropout_ratio: (float) The dropout ratio for the GRUs 
            bidirectional: (boolean) Whether or not to use bidirectional GRUs 
            encoder_h0: (tensor) Initial state of the GRU 
        """
        # (batch_size, in_traj_len, embedding_size)
        # in_traj_len is not static 
        inputs = Input((None, embedding_size))

        hn = []
        if bidirectional:
            gru = Bidirectional(GRU(gru_cell_size, 
                                    dropout = gru_dropout_ratio,
                                    return_sequences = True))\
                               (inputs, initial_state = encoder_h0)
            hn.append(gru) 
            for i in range(num_gru_layers-1):
                gru = Bidirectional(GRU(gru_cell_size,
                                        return_sequences = True,
                                        dropout = gru_dropout_ratio))(gru)
                hn.append(gru)
        else:
            gru = GRU(gru_cell_size, return_sequences = True, 
                      dropout = gru_dropout_ratio)\
                      (inputs, initial_state=encoder_h0)
            hn.append(gru) 
            for i in range(num_gru_layers-1):
                gru = GRU(gru_cell_size, return_sequences = True,
                          dropout = gru_dropout_ratio)(gru)
                hn.append(gru)
        model = Model(inputs = inputs, outputs = gru) 
        self.model = model 
        
    
class Encoder():
    """
    Handles the Encoder part of the overall encoder-decoder model. The Encoder 
    takes as input the trajectory after the embedding is done and outputs the 
    hidden state of each layer as well as the feature vector representation of 
    each trajectory after being passed through the GRU layers contained within. 
    """
    
    def __init__(self, in_traj_len, embedding, gru):
        """
        Initializes the model. 
        
        Args:
            in_traj_len: (integer) Length of trajectory 
            embedding: (keras model) The embedding layer 
            gru: (keras model) The stacked GRU model 
        """
        inputs = Input((in_traj_len,))
        embedded = embedding(inputs) 
        outputs = gru(embedded)
        model = Model(inputs = inputs, outputs = outputs) 
        self.model = model 


class Decoder():
    """
    Handles the Decoder part of the overall encoder-decoder model. The Decoder 
    first it produces the feature vector representation of the target 
    trajectory. Then, an attention layer is applied to this feature vector 
    representation and the source trajectory feature vector representation. 
    The output of this attention module is used as the Decoder 
    output.
    """
    
    def __init__(self, src_traj_len, src_feature_size, trg_traj_len,
                 use_attention, embedding, gru):
        """
        Initialize the model 
        
        Inputs:
            inputs_1: Shape is (batch_size, src_traj_len, src_feature_size).
                      This is the learned representation of the source 
                      trajectory, which is the output from the Encoder 
            inputs_2: Shape is (batch_size, trg_traj_len). This is the target 
                      trajectory. 
                      
        Outputs:
            outputs: Shape is (batch_size, trg_traj_len, src_feature_size)
        
        Args:
            src_traj_len: (integer) The length of the source trajectory 
            src_feature_size: (integer) Feature vector size of each sequence in 
                               the target trajectory. 
            trg_traj_len: (integer) The length of the target trajectory 
            use_attention: (boolean) Whether or not to use the attention module 
            embedding: (keras model) The embedding layer 
            gru: (keras model) The GRU model 
        """
        # embedded_src (batch_size, src_traj_len, src_feature_size)
        # This input is for the encoded source trajectory 
        embedded_src = Input((src_traj_len, src_feature_size))
        
        # inputs_trg (batch_size, trg_traj_len)
        # This input is for the target trajectory 
        inputs_trg = Input((trg_traj_len,))
        
        # inputs_trg (batch_size, trg_traj_len)
        # embedded_trg (batch_size, trg_traj_len, src_feature_size)
        # Since the embedding layer used is the same as the target, the 
        # output feature size is the same as src_feature_size
        embedded_trg = embedding(inputs_trg)
        embedded_trg = gru(embedded_trg)
        
        # embedded_src (batch_size, src_traj_len, src_feature_size)
        # embedded_trg (batch_size, trg_traj_len, src_feature_size)
        # outputs (batch_size, trg_traj_len, src_feature_size)
        # TODO: OPTION TO NOT USE ATTENTION 
        outputs = Attention()([embedded_trg, embedded_src])
        self.model = Model(inputs = [embedded_src, inputs_trg],outputs=outputs)


class PatternDecoder():
    """
    The pattern decoder. It first takes the target pattern and produces a 
    feature vector representation so that the dimension matches the source 
    input, which is the source trajectory feature vector representation. Then,
    the two are fed into an attention layer to produce the prediction 
    """
    def __init__(self, src_traj_len, src_feature_size, trg_traj_len,
                 use_attention):
        """
        Initialize the model 
        
        Inputs:
            inputs_1: Shape is (batch_size, src_traj_len, src_feature_size).
                      This is the learned representation of the source 
                      trajectory, which is the output from the Encoder 
            inputs_2: Shape is (batch_size, trg_traj_len, 2). This is the 
                      target trajectory pattern features, which includes the 
                      spatial and temporal features. 
                      
        Outputs:
            outputs: Shape is (batch_size, trg_traj_len, src_feature_size)
        
        Args:
            src_traj_len: (integer) The length of the source trajectory 
            src_feature_size: (integer) Feature vector size of each sequence in 
                               the target trajectory. 
            trg_traj_len: (integer) The length of the target trajectory 
            use_attention: (boolean) Whether or not to use the attention module 
        """
        # embedded_src (batch_size, src_traj_len, src_feature_size)
        # This input is for the encoded source trajectory 
        embedded_src = Input((src_traj_len, src_feature_size))
        
        # inputs_trg (batch_size, trg_traj_len, 2)
        # This input is for the target spatiotemporal trajectory pattern
        inputs_trg = Input((trg_traj_len, 2))
        
        # inputs_trg (batch_size, trg_traj_len, 2)
        # embedded_trg (batch_size, trg_traj_len, src_feature_size)
        # Learns features based on inputs_trg. 
        embedded_trg = TimeDistributed(Dense(src_feature_size))(inputs_trg)
        
        # embedded_src (batch_size, src_traj_len, src_feature_size)
        # embedded_trg (batch_size, trg_traj_len, src_feature_size)
        # outputs (batch_size, trg_traj_len, src_feature_size)
        # TODO: OPTION TO NOT USE ATTENTION 
        outputs = Attention()([embedded_trg, embedded_src])
        self.model = Model(inputs = [embedded_src, inputs_trg],outputs=outputs)


class EncoderDecoder():
    """
    The EncoderDecoder model is the final model to be used, which is an 
    amalgamation of the Encoder, Decoder, and other components contained 
    within the other classes in this module.
    """
    
    def __init__(self, src_traj_len, trg_traj_len, trg_patt_len,
                 embed_vocab_size, embedding_size, gru_cell_size, 
                 num_gru_layers, gru_dropout_ratio, bidirectional, 
                 use_attention, encoder_h0 = None):
        """
        Initializes the model. 
        
        Inputs: 
            input: Shape is (batch_size, src_traj_len). This is the input 
                   trajectory where each item in the sequence is the 
                   spatiotemporal cell IDs 
        Outputs:
            output: Shape is (batch_size, trg_traj_len, embedding_size). It 
                    contains the feature vector representation of the predicted 
                    output trajectory. 
        Where: 
            - batch_size: Size of input batch 
            - src_traj_len: Length of the source (i.e. query) trajectory 
            - trg_traj_len: Length of the output (i.e. ground truth) trajectory
            - k: Stands for the top-k nearest neighbors' weights. 
        
        Args:
            src_traj_len: (integer) Length of source trajectory 
            trg_traj_len: (integer) Length of target trajectory 
            embed_vocab_size: (integer) Size of the embedding layer vocabulary
            embedding_size: (integer) Size of the embedding layer output 
            gru_cell_size: (integer) Size of the gru cels 
            num_gru_layers: (integer) How many GRUs to use 
            gru_dropout_ratio: (float) Dropout ratio for the GRUs 
            bidirectional: (boolean) Whether or not bidirectional GRU is used 
            use_attention: (boolean) Whether or not the attention model is used 
            encoder_h0: (tensor) Initial state of the encoder 
        """
        # inputs_1 (batch_size, src_traj_len) 
        # This input is for the source trajectory 
        inputs_1 = Input((src_traj_len,))
        
        # inputs_2 (batch_size, trg_traj_len)
        # This input is for the target trajectory 
        inputs_2 = Input((trg_traj_len,))
        
        # inputs_3 (batch_size, trg_patt_len)
        # This input is for the target pattern 
        inputs_3 = Input((trg_patt_len,2))
        
        # Encoder part
        self.sgru = StackedGRU(embedding_size, gru_cell_size, 
                               num_gru_layers, gru_dropout_ratio, bidirectional, 
                               encoder_h0).model
        self.embedding = Embedding(embed_vocab_size, embedding_size)
        self.encoder = Encoder(src_traj_len, self.embedding, self.sgru)
        # inputs (batch_size, src_traj_len) 
        # encoded (batch_size, src_traj_len, gru_cell_size * directions) 
        # directions = 2 if bidirectional, else 1.
        encoded = self.encoder.model(inputs_1)
        
        # Decoder part 
        traj_repr_size = gru_cell_size
        if bidirectional:
            traj_repr_size *= 2
        self.decoder = Decoder(src_traj_len, traj_repr_size, trg_traj_len,
                               use_attention, self.embedding, self.sgru)
        # encoded (batch_size, src_traj_len, gru_cell_size * directions) 
        # inputs_2 (batch_size, trg_traj_len)
        # output_point (batch_size, trg_traj_len, gru_cell_size * directions)
        output_point = self.decoder.model([encoded, inputs_2])
        
        # Pattern decoder part 
        # Pattern decoder 
        self.patt_decoder = PatternDecoder(src_traj_len, traj_repr_size, 
                                           trg_patt_len, use_attention)
        # encoded (batch_size, src_traj_len, embedding_size) 
        # inputs_2 (batch_size, trg_traj_len)  
        # output_patt (batch_size, trg_patt_len, embedding_size)
        output_patt = self.patt_decoder.model([encoded, inputs_3])
        
        # Finished model 
        self.model = Model(inputs = [inputs_1, inputs_2, inputs_3], 
                           outputs = [output_point, output_patt])
        

class STSeqModel():
    """This class handles the SpatioTemporal Sequence2Sequence """
    
    
    def __init__(self, embed_vocab_size, embedding_size, traj_repr_size,
                 gru_cell_size, num_gru_layers, gru_dropout_ratio, 
                 bidirectional, use_attention, xshape, k):
        """
        Creates the model
        
        Args: 
            gru_cell_size: (integer) Size of every LSTM in the model 
            traj_repr_size: (integer) The size of the trajectory vector 
                             representation 
            xshape: (numpy array) The size of the input numpy arrays 
        """
        # 'inputs' shape (batch_size, num_inner_data, traj_len, 1)
        # num_inner_data should be 5, representing: 
        # - ground truth trajectory, 
        # - query trajectory
        # - negative trajectory
        # - ground truth length
        # - ground truth pattern length 
        inputs = Input((xshape[1], xshape[2], xshape[3]))
        
        ## Lambda layers to split the inputs. 
        # 'gt' shape (batch_size, traj_len, 1). 
        # Represents ground truth trajectory.
        gt = Lambda(lambda x:x[:,0,:,:])(inputs)
        gt = Masking(mask_value = 0)(gt) 
        
        # 'q' shape (batch_size, traj_len, 1). 
        # Represents the query trajectory. 
        q = Lambda(lambda x:x[:,1,:,:])(inputs)
        q = Masking(mask_value = 0)(q) 
        
        # 'neg' shape (batch_size, traj_len, 1). 
        # Represents the negative trajectory. 
        neg = Lambda(lambda x:x[:,2,:,:])(inputs)
        neg = Masking(mask_value = 0)(neg) 
        
        # 'gt_patt_s' shape (batch_size, traj_len, 1).
        gt_patt_s = Lambda(lambda x:x[:,3,:,:])(inputs)
        gt_patt_s = Masking(mask_value = 0)(gt_patt_s) 
        
        # 'gt_patt_t' shape (batch_size, traj_len, 1).
        gt_patt_t = Lambda(lambda x:x[:,4,:,:])(inputs)
        gt_patt_t = Masking(mask_value = 0)(gt_patt_t) 
        
        # 'gt_patt_st' shape (batch_size, traj_len, 2)
        gt_patt_st = Concatenate(axis=2)([gt_patt_s, gt_patt_t])
        
        # 'gt_len' shape (batch_size, traj_len, 1). 
        # Represents the actual ground truth trajectory length. 
        gt_len = Lambda(lambda x:x[:,5,:,:])(inputs)
        
        # 'gt_patt_len' shape (batch_size, traj_len, 1). 
        # Represents the ground truth pattern length. 
        gt_patt_len = Lambda(lambda x:x[:,6,:,:])(inputs)
        
        # EncoderDecoder model 
        assert gt_patt_s.shape[0] == gt_patt_t.shape[0] 
        src_traj_len = q.shape[1]
        trg_traj_len = gt.shape[1] 
        trg_patt_len = gt_patt_s.shape[1]
        
        # 'q' shape (batch_size, traj_len, 1). 
        # 'gt' shape (batch_size, traj_len, 1). 
        # 'gt_patt_st' shape (batch_size, traj_len, 2)
        # 'traj_repr' shape (batch_size,trg_traj_len,gru_cell_size * directions)
        # 'patt_repr' shape (batch_size,trg_traj_len,gru_cell_size * directions)
        # directions = 2 if bidirectional, else 1
        encoder_decoder = EncoderDecoder(src_traj_len, trg_traj_len,
                                         trg_patt_len, embed_vocab_size, 
                                         embedding_size, gru_cell_size,
                                         num_gru_layers, gru_dropout_ratio,
                                         bidirectional, use_attention)
        [traj_repr, patt_repr] = encoder_decoder.model([q, gt, gt_patt_st])
        
        # 'traj_repr' shape (batch_size, trg_traj_len, k)
        traj_repr = TimeDistributed(Dense(k))(traj_repr)
        
        # 'patt_repr' shape (batch_size, trg_traj_len, 2)
        patt_repr = TimeDistributed(Dense(2))(patt_repr)
        
        # Encoder part 
        encoder = encoder_decoder.encoder 
        # 'q' shape (batch_size, traj_len, 1). 
        # 'gt' shape (batch_size, traj_len, 1). 
        # 'neg' shape (batch_size, traj_len, 1). 
        # 'enc_q' shape (batch_size, traj_len, gru_cell_size * directions). 
        # 'enc_gt' shape (batch_size, traj_len, gru_cell_size * directions). 
        # 'enc_neg' shape (batch_size, traj_len, gru_cell_size * directions). 
        # directions = 2 if bidirectional, else 1 
        enc_q = encoder.model(q)
        enc_gt = encoder.model(gt)
        enc_neg = encoder.model(neg) 
        
        # Three loss functions needed, so we need three outputs 
        # First is the representation loss which takes the encoder outputs 
        # 'enc_q' shape (batch_size, traj_len, gru_cell_size * directions). 
        # 'enc_gt' shape (batch_size, traj_len, gru_cell_size * directions). 
        # 'enc_neg' shape (batch_size, traj_len, gru_cell_size * directions). 
        # 'out_repr' shape (batch_size, 3, traj_len, gru_cell_size * directions)
        out_repr = K.stack([enc_q, enc_gt, enc_neg], axis=1)
        
        # Second is the point-to-point spatiotemporal loss 
        # 'traj_repr' shape (batch_size, trg_traj_len, k)
        # 'gt_len' shape (batch_size, traj_len, 1). 
        # 'out_traj' shape (batch_size, traj_len, k+1)
        out_traj = Concatenate()([traj_repr, gt_len]) 
        
        # Third is the pattern loss 
        # 'patt_repr' shape (batch_size, trg_traj_len, 2)
        # 'gt_patt_len' shape (batch_size, traj_len, 1). 
        # 'out_patt' shape (batch_size, traj_len, 3). 
        out_patt = Concatenate()([patt_repr, gt_patt_len])
         
        # Create model 
        model = Model(inputs = inputs, outputs = [out_repr, out_traj, out_patt])
        self.model = model  