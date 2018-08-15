from datasets import MnistDataset
from keras import layers, models
import keras

class CNN(models.Model):
    def __init__(self, input_shape, num_classes):
        conv2d_32 = layers.Conv2D(32, kernel_size=(3, 3))
        conv2d_64 = layers.Conv2D(64, kernel_size=(3, 3))
        max_pool = layers.MaxPool2D(pool_size=(2, 2))
        dropout_025 = layers.Dropout(0.25)

        flatten = layers.Flatten()
        dense_128 = layers.Dense(128)
        dropout_050 = layers.Dropout(0.5)

        relu = layers.Activation('relu')
        relu = layers.ReLU()
        softmax = layers.Activation('softmax')
        softmax = layers.Softmax()

        dense_out = layers.Dense(num_classes)

        x = layers.Input(shape=input_shape)
        h1 = relu(conv2d_32(x))
        h2 = relu(conv2d_64(h1))
        h3 = max_pool(h2)
        h3_dropout = dropout_025(h3)

        h4 = flatten(h3_dropout)
        h5 = relu(dense_128(h4))
        h5_dropout = dropout_050(h5)

        y = softmax(dense_out(h5_dropout))

        super().__init__(x, y)
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


mnist_dataset = MnistDataset()
model_cnn = CNN(mnist_dataset.input_shape, mnist_dataset.num_class)

batch_size = 128
epochs = 10


history = model_cnn.fit(mnist_dataset.x_train, mnist_dataset.y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
score = model_cnn.evaluate(mnist_dataset.x_test, mnist_dataset.y_test)

print('Loss and Accuracy -> ', score)
