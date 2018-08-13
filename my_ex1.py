import keras
import numpy

x = numpy.array([1,2,3,4,5])
y = x * 2 + 1

model = keras.models.Sequential()
model.add(keras.layers.Dense(1, input_shape=(1,)))

model.compile(optimizer='SGD', loss='mse')

model.fit(x[:2], y[:2], epochs=1000, verbose=0)

print('Targets:', y[2:])
print('Predictions:', model.predict(x[2:]).flatten())

