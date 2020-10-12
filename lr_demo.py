import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
x = np.random.random((10000))
y = x * 1.5 + 3


mu, sigma = 0, 0.1
noise = np.random.normal(mu, sigma, 10000)

y += noise

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

class Linear(keras.layers.Layer):
    def __init__(self):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(1,), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(1,), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        return inputs * self.w + self.b


x_input = keras.Input(shape=(1,), dtype='float32')
liner = Linear()
out = liner(x_input)
model = keras.models.Model(inputs=x_input, outputs=out)
model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(learning_rate=0.2), metrics=keras.metrics.MeanSquaredError())

model.summary()
model.fit(x=x, y=y, epochs=10, batch_size=216, validation_split=0.2, verbose=1)
print(liner.w)
print(liner.b)