import trax
import pickle
from collections import defaultdict
import numpy as np
import random as rnd
from trax.fastmath import numpy as fastnp
from trax import layers as tl
from trax.supervised import training



def load_vocab(path="vocab.pkl"):
    with open(path, "rb") as f:
        vocab_regular = pickle.load(f)

    # Restore as defaultdict with OOV handling
    vocab = defaultdict(lambda: 0)
    vocab.update(vocab_regular)
    return vocab

vocab = load_vocab()  


# dat6a_generator function to returnj tuple of 2 arrays with each has batch_size questions
def data_generator(Q1, Q2, batch_size, pad = 1, shuffle = True):
    # initialize inputs to empty lists
    input1 = []
    input2 = []
    
    # initialize index to 0
    index = 0
    
    # get questions indexes
    q_indexes = [*range(len(Q1))]
    
    # if shuffle is True shuffle q_indexes
    if shuffle:
        rnd.shuffle(q_indexes)

    while True:
        if index >= len(Q1):
            # set index to 0
            index = 0
            
            # shuffle data if shuffle set to True
            if shuffle:
                rnd.shuffle(q_indexes)

        # get questions from Q1, Q2 by using index from q_indexes
        q1 = Q1[q_indexes[index]]
        q2 = Q2[q_indexes[index]]
        # increment index by 1
        index += 1
        # append q1 and q2 to inputs
        input1.append(q1)
        input2.append(q2)
        
        # if inputs reached to batch_size
        if len(input1) == batch_size:
            # get max question length in both inputs then we take the max of two of them
            max_len = max(len(max(input1, key = len)), len(max(input2, key = len)))
            # ceil max_len to power of 2
            max_len = 2 ** int(np.ceil(np.log2(max_len)))
            
            # initialize b1 and b2 to empty lists for storing padded questions
            b1 = []
            b2 = []
            
            # get q1, q2 from inputs
            for q1, q2 in zip(input1, input2):
                # pad q1 until it reaches max_len
                q1 = q1 + [pad] * (max_len - len(q1))
                # pad q2 until it reaches max_len
                q2 = q2 + [pad] * (max_len - len(q2))
                # append q1 and q2 to b1 and b2
                b1.append(q1)
                b2.append(q2)
                
            # return b1 and b2 as tuple of numpy arrays
            yield (np.array(b1), np.array(b2))
            
            # reset batches
            input1, input2 = [], []

       
    
# Siamese() model
def siamese(vocab_size = len(vocab), d_model = 128, mode = "train"):
    # normalize function (L2 norm)
    def normalize(x):
        return x / fastnp.sqrt(fastnp.sum(x * x, axis = -1, keepdims = True))
        
    # q_processor is a subnetwork used to process the question
    q_processor = tl.Serial(
        # convert input tensor to embedded tensor of shape [vocab_size, d_model]
        tl.Embedding(vocab_size, d_model),
        # fed embedded tensors to LSTM layer
        tl.LSTM(d_model),
        # take the average value across columns
        tl.Mean(axis = 1),
        # normalize thd averaged matrix
        tl.Fn('Normalize', lambda x: normalize(x))
        # returns [batch_size, d_model]
    )
    
    # model has 2 subnetworks share parameters in parallel
    model = tl.Parallel(q_processor, q_processor)
    # return model
    return model