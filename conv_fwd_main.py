
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from nn.layers import Conv2D
from utils.tools import rel_error

from keras import Sequential
from keras.layers import Conv2D as k_Conv2D

input = np.random.uniform(size=(10, 3, 30, 30))
params = { 
    'kernel_h': 4,
    'kernel_w': 4,
    'pad': 2,
    'stride': 2,
    'in_channel': input.shape[1],
    'out_channel': 64,
}
conv = Conv2D(params)
out = conv.forward(input)
print(out.shape)
keras_conv = Sequential([
    k_Conv2D(filters=params['out_channel'],
            kernel_size=(params['kernel_h'], params['kernel_w']),
            strides=(params['stride'], params['stride']),
            padding='same',
            data_format='channels_first',
            input_shape=input.shape[1:]),
])
keras_conv.layers[0].set_weights([conv.weights.transpose((2,3,1,0)), conv.bias])

keras_out = keras_conv.predict(input, batch_size=input.shape[0])
print(keras_out.shape)
print('Relative error (<1e-6 will be fine): ', rel_error(out, keras_out))
