#!/usr/bin/env python

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

X = np.arange(1000).reshape([-1,10])
y = np.arange(100)

dense = Dense(1, input_dim=10)

model = Sequential([
    dense
])

print(np.sum(dense.get_weights()[0]))

model.trainable = False
# dense.trainable=False
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, nb_epoch=1000)

print(np.sum(dense.get_weights()[0]))