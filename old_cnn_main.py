import matplotlib.pyplot as plt
import numpy as np

from models.Fas_MNISTNet import Fashion_MNISTNet
from models.MyFashMNIST_CNN import MyFashMNIST_CNN
from nn.loss import SoftmaxCrossEntropy, L2
from nn.optimizers import Adam
from data.datasets import Fashion_MNIST
np.random.seed(5242)

Fashion_mnist = Fashion_MNIST()
Fashion_mnist.load()
#print(Fashion_mnist.dtype)
print(Fashion_mnist)

idx = np.random.randint(Fashion_mnist.num_train, size=4)
print('\nFour examples of training images:')
img = Fashion_mnist.x_train[idx][:,0,:,:]


from nn.optimizers import RMSprop, Adam

#model = Fashion_MNISTNet()
model = MyFashMNIST_CNN()
loss = SoftmaxCrossEntropy(num_class=10)

# define your learning rate sheduler
def func(lr, iteration):
    if iteration % 1000 ==0:
        return lr*0.5
    else:
        return lr

adam = Adam(lr=0.001, decay=0,  sheduler_func=None, bias_correction=True)
l2 = L2(w=0.001) # L2 regularization with lambda=0.001
model.compile(optimizer=adam, loss=loss, regularization=l2)

import time
start = time.time()
train_results, val_results, test_results = model.train(
    Fashion_mnist, 
    train_batch=50, val_batch=1000, test_batch=1000, 
    epochs=2, 
    val_intervals=-1, test_intervals=900, print_intervals=100)
print('cost:', time.time()-start)


