from nn.layers import *
from nn.model import Model


def MyFashionModel_CNN():
    conv1_params = {
        'kernel_h': 3,
        'kernel_w': 3,
        'pad': 2, #same
        'stride': 1,
        'in_channel': 1,
        'out_channel': 32
    }
    conv2_params = {
        'kernel_h': 3,
        'kernel_w': 3,
        'pad': 2, #same
        'stride': 1,
        'in_channel': 32,
        'out_channel': 32
    }
    conv3_params = {
        'kernel_h': 3,
        'kernel_w': 3,
        'pad': 2, #same
        'stride': 1,
        'in_channel': 32,
        'out_channel': 64
    }
    conv4_params = {
        'kernel_h': 3,
        'kernel_w': 3,
        'pad': 2, #same
        'stride': 1,
        'in_channel': 64,
        'out_channel': 128
    }
    pool1_params = {
        'pool_type': 'max',
        'pool_height': 2,
        'pool_width': 2,
        'stride': 2,
        'pad': 0
    }


    model = Model()
    model.add(Conv2D(conv1_params, name='conv1',initializer=Gaussian(std=0.001)))
    model.add(ReLU(name='relu1'))
    # model.add(Dropout(rate=0.25, name='dropout1'))

    model.add(Conv2D(conv2_params, name='conv2',
                          initializer=Gaussian(std=0.001)))
    model.add(ReLU(name='relu2'))
    model.add(Dropout(rate=0.25, name='dropout2'))

    model.add(Conv2D(conv3_params, name='conv3',
                          initializer=Gaussian(std=0.001)))
    model.add(ReLU(name='relu3'))

    model.add(Pool2D(pool1_params, name='pooling1'))
    model.add(Dropout(rate=0.25, name='dropout3'))

    model.add(Conv2D(conv4_params, name='conv4',initializer=Gaussian(std=0.001)))
    model.add(ReLU(name='relu4'))
    model.add(Dropout(rate=0.25, name='dropout4'))

    model.add(Flatten(name='flatten'))
    model.add(Linear(25088, 512, name='fclayer1', #512
                      initializer=Gaussian(std=0.01)))
    model.add(ReLU(name='relu5'))
    model.add(Dropout(rate=0.5, name='dropout5'))

    model.add(Linear(512, 256, name='fclayer1', #128
                      initializer=Gaussian(std=0.01)))
    model.add(ReLU(name='relu5'))
    model.add(Dropout(rate=0.5, name='dropout5'))

    model.add(Linear(256, 10, name='fclayer2',
                      initializer=Gaussian(std=0.01)))
    return model
