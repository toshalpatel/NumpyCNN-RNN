from nn.layers import *
from nn.model import Model
from nn.initializers import *

def MyModel_SentimentNet(word_to_idx):
    
    vocab_size = len(word_to_idx)

    model = Model()
    model.add(Linear2D(vocab_size, 200, name='embedding', initializer=Gaussian(std=0.01)))
    model.add(BiRNN(in_features=200, units=50, initializer=Gaussian(std=0.01)))
    model.add(Linear2D(100, 50, name='linear1', initializer=Gaussian(std=0.01)))
    model.add(TemporalPooling()) # defined in layers.py
    model.add(Linear2D(50, 2, name='linear2', initializer=Gaussian(std=0.01)))

#    model.add(Linear2D(vocab_size, 200, name='embedding', initializer=Gaussian(std=0.01)))
#    model.add(GRU(in_features=200, units=50, initializer=Gaussian(std=0.01)))

#    model.add(Linear2D(50, 50, name='embedding', initializer=Gaussian(std=0.01)))
#    model.add(GRU(in_features=200, units=50, initializer=Gaussian(std=0.01)))
#    model.add(TemporalPooling())
#    model.add(Linear(200,2,initializer=Gaussian(std=0.01)))
    return model
