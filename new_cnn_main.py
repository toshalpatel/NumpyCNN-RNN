import matplotlib.pyplot as plt
import numpy as np

from models.MyModel_FashionMNIST import MyModel_FashionMNIST
from nn.loss import SoftmaxCrossEntropy, L2
from nn.optimizers import Adam, SGD
from data.datasets import Fashion_MNIST
np.random.seed(0)

Fashion_mnist = Fashion_MNIST()
Fashion_mnist.load()

from nn.optimizers import RMSprop, Adam

model = MyModel_FashionMNIST()
loss = SoftmaxCrossEntropy(num_class=10)

# define your learning rate sheduler
def func(lr, iteration):
    if iteration % 1000 ==0:
        return lr*0.5
    else:
        return lr

adam = Adam(lr=0.001, decay=0,  sheduler_func=None, bias_correction=True)
#sgd = SGD(lr=0.01, beta=0.9)
l2 = L2(w=0.001) # L2 regularization with lambda=0.001
model.compile(optimizer=adam, loss=loss, regularization=l2)

import time
start = time.time()
train_results, val_results, test_results = model.train(
    Fashion_mnist, 
    train_batch=50, val_batch=1000, test_batch=1000, 
    epochs=5, 
    val_intervals=-1, test_intervals=1000, print_intervals=100)
print('cost:', time.time()-start)


