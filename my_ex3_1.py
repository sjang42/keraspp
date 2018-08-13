from keras import layers, models
from datasets import mnist_data


class DNN(models.Model):
    def __init__(self, input_size, hidden_sizes, output_size):
        hidden1 = layers.Dense(hidden_sizes[0])
        hidden2 = layers.Dense(hidden_sizes[1])
        output = layers.Dense(output_size)

        relu = layers.Activation('relu')
        softmax = layers.Activation('softmax')

        x = layers.Input(shape=(input_size,))
        h1 = relu(hidden1(x))
        h2 = relu(hidden2(h1))
        y = softmax(output(h2))

        super().__init__(x, y)
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


input_size = 784
hidden_sizes = [400, 100]
output_size = 10

model = DNN(input_size, hidden_sizes, output_size)
(X_train, y_train), (X_test, y_test) = mnist_data()

history = model.fit(x=X_train, y=y_train, batch_size=100, epochs=15, verbose=2)
performance_test = model.evaluate(X_test, y_test, batch_size=100)
print('Test Loss and Accuracy ->', performance_test)
