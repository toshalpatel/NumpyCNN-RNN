import numpy as np
from nn.layers import GRUCell
from utils.check_grads_rnn import check_grads_layer

N, D, H = 3, 10, 4
x = np.random.uniform(size=(N, D))
prev_h = np.random.uniform(size=(N, H))
in_grads = np.random.uniform(size=(N, H))

gru_cell = GRUCell(in_features=D, units=H)

check_grads_layer(gru_cell, [x, prev_h], in_grads)
