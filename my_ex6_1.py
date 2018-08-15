from keras import models, layers
from datasets import mnist_data


class AE(models.Model):
    def __init__(self, x_nodes, z_dim=36):
        x_shape = (x_nodes,)
        hidden = layers.Dense(z_dim)
        out = layers.Dense(x_nodes)
        relu = layers.ReLU()
        sigmoid = layers.Activation('sigmoid')

        x = layers.Input(shape=x_shape)
        z = relu(hidden(x))
        y = sigmoid(out(z))

        super().__init__(x, y)

        self.x_nodes = x_nodes
        self.z_dim = z_dim
        self.x = x
        self.z = z
        self.out = out
        self.sigmoid = sigmoid

        self.compile(optimizer='adadelta', loss = 'binary_crossentropy', metrics=['accuracy'])

    def Encoder(self):
        return models.Model(self.x, self.z)

    def Decoder(self):
        z_shape = (self.z_dim,)
        z = layers.Input(shape=z_shape)
        y = self.sigmoid(self.out(z))
        return models.Model(z, y)


###########################
# 학습 효과 분석
###########################
from keraspp.skeras import plot_loss, plot_acc
import matplotlib.pyplot as plt


###########################
# AE 동작 확인
###########################
def show_ae(autoencoder):
    encoder = autoencoder.Encoder()
    decoder = autoencoder.Decoder()

    encoded_imgs = encoder.predict(X_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    n = 10
    plt.figure(figsize=(20, 6))
    for i in range(n):

        ax = plt.subplot(3, n, i + 1)
        plt.imshow(X_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n)
        plt.stem(encoded_imgs[i].reshape(-1))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()



x_nodes = 784
z_dim = 36

ae = AE(x_nodes, z_dim)
(X_train, _), (X_test, _) = mnist_data()
history = ae.fit(X_train, X_train, epochs=10, batch_size=256, shuffle=True, validation_data=(X_test, X_test))
show_ae(ae)

