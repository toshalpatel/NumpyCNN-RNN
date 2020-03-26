from data import datasets
#from models.SentimentNet import SentimentNet
from models.MyModel_SentimentNet import MyModel_SentimentNet
from nn.loss import SoftmaxCrossEntropy, L2
from nn.optimizers import Adam
import numpy as np

np.random.seed(0)

dataset = datasets.Sentiment()
#model = SentimentNet(dataset.dictionary)
model = MyModel_SentimentNet(dataset.dictionary)
loss = SoftmaxCrossEntropy(num_class=2)

adam = Adam(lr=0.01, decay=0,
            sheduler_func=lambda lr, it: lr*0.5 if it%1000==0 else lr)
model.compile(optimizer=adam, loss=loss, regularization=L2(w=0.001))
train_results, val_results, test_results = model.train(
        dataset, 
        train_batch=20, val_batch=100, test_batch=100, 
        epochs=28, 
        val_intervals=-1, test_intervals=20, print_intervals=5)

