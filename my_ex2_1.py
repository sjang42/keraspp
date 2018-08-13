from keras import layers, models
from keras import datasets
from keras.utils import np_utils
import matplotlib.pyplot as plt


class ANN_models_class(models.Model):
    def __init__(self, input_size, hidden_size, output_size):
        hidden = layers.Dense(hidden_size)
        output = layers.Dense(output_size)
        relu = layers.Activation('relu')
        softmax = layers.Activation('softmax')

        x = layers.Input(shape=(input_size, ))
        h = relu(hidden(x))
        y = softmax(output(h))

        super().__init__(x, y)
        self.compile(loss='categorical_crossentropy',
                     optimizer='adam', metrics=['accuracy'])


def mnist_data():
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)

    L, W, H = X_train.shape
    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)
    print(X_train.shape)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return (X_train, Y_train), (X_test, Y_test)


def plot_loss(history, title=None):
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])

    if title is not None:
        plt.title(title)

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)

    # plt.show()


input_size = 784
hidden_size = 100
output_size = 10

model = ANN_models_class(input_size, hidden_size, output_size)
(X_train, Y_train), (X_test, Y_test) = mnist_data()


history = model.fit(x=X_train, y=Y_train, batch_size=100, epochs=15, validation_split=0.2)
performance_test = model.evaluate(X_test, Y_test, batch_size=100)
print('Test Loss and Accuracy ->', performance_test)

plot_loss(history)
plt.show()
