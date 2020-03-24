from data import datasets
from models.SentimentNet import SentimentNet
from nn.loss import SoftmaxCrossEntropy, L2
from nn.optimizers import Adam
import numpy as np
np.random.seed(5242)

dataset = datasets.Sentiment()
model = SentimentNet(dataset.dictionary)
loss = SoftmaxCrossEntropy(num_class=2)

adam = Adam(lr=0.01, decay=0,
            sheduler_func=lambda lr, it: lr*0.5 if it%1000==0 else lr)
model.compile(optimizer=adam, loss=loss, regularization=L2(w=0.001))
train_results, val_results, test_results = model.train(
        dataset, 
        train_batch=20, val_batch=100, test_batch=100, 
        epochs=5, 
        val_intervals=-1, test_intervals=25, print_intervals=5)

# import matplotlib.pyplot as plt
# plt.figure(figsize=(18, 8))
# plt.subplot(2, 2, 1)
# plt.title('Training loss')
# plt.plot(train_results[:,0], train_results[:,1])
# plt.subplot(2, 2, 3)
# plt.title('Training accuracy')
# plt.plot(train_results[:,0], train_results[:,2])
# plt.subplot(2, 2, 2)
# plt.title('Testing loss')
# plt.plot(test_results[:,0], test_results[:, 1])
# plt.subplot(2, 2, 4)
# plt.title('Testing accuracy')
# plt.plot(test_results[:, 0], test_results[:,2])
# plt.plot(test_results[:, 0], test_results[:,2])
