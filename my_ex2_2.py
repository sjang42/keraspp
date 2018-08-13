from keras import layers, models
from keras import datasets
from sklearn import preprocessing
import matplotlib.pyplot as plt


def boston_housing_data():
    (X_train, y_train), (X_test, y_test) = datasets.boston_housing.load_data()
    scaler = preprocessing.MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return (X_train, y_train), (X_test, y_test)


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


class ANN_models_regression(models.Model):
    def __init__(self, input_size, hidden_size, output_size):
        hidden = layers.Dense(hidden_size)
        relu = layers.Activation('relu')
        output = layers.Dense(output_size)

        x = layers.Input(shape=(input_size,))
        h = relu(hidden(x))
        y = output(h)

        super().__init__(x, y)
        self.compile(loss='mse', optimizer='sgd')


input_size = 13
hidden_size = 5
output_size = 1

model = ANN_models_regression(input_size, hidden_size, output_size)

(X_train, y_train), (X_test, y_test) = boston_housing_data()
history = model.fit(X_train, y_train, epochs=100, batch_size=100, validation_split=0.2, verbose=2)

performance_test = model.evaluate(X_test, y_test, batch_size=100)
print('\nTest Loss -> {}'.format(performance_test))

y_predict = model.predict(X_test, batch_size=100)

plot_loss(history)
plt.show()
