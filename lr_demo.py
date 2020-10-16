import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
x = np.random.random((10000))
y = x * 7 - 3


mu, sigma = 0, 0.1
noise = np.random.normal(mu, sigma, 10000)

y += noise
x=x.astype(np.float32)
y=y.astype(np.float32)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

class Linear(keras.layers.Layer):
    def __init__(self):
        super(Linear, self).__init__()


    def build(self, input_shape):
        w_init = tf.random_normal_initializer(stddev=4)
        self.w = tf.Variable(
            initial_value=w_init(shape=(1,), dtype="float32"),
            trainable=True,
            name='w'
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(1,), dtype="float32"), trainable=True,
            name='b'
        )

    def call(self, inputs):
        return inputs * self.w + self.b


train_dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(100).batch(1024)
x_input = keras.Input(shape=(1,), dtype='float32')
liner = Linear()
out = liner(x_input)
model = keras.models.Model(inputs=x_input, outputs=out)

print(liner.w)
print(liner.b)

metric = keras.metrics.MeanSquaredError()

print(model.trainable_variables)
optimizer = keras.optimizers.Adam(learning_rate=1)
for i in range(20):
    print(f"epoch {i+1}")
    epoch_loss_avg = tf.keras.metrics.Mean()
    for x_in, y_true in train_dataset:
        with tf.GradientTape() as tape:
            y_pred = model(x_in)
            loss = keras.losses.mean_squared_error(y_true=y_true, y_pred=y_pred)
        grad = tape.gradient(loss, model.trainable_variables)
        epoch_loss_avg.update_state(loss)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
    print(f"loss: {epoch_loss_avg.result()}")

features, labels = next(iter(train_dataset))
print(features[:5])
print(labels[:5])
print(model(features[:5]))
# model.compile(loss=keras.losses.mean_squared_error, optimizer=tf.keras.optimizers.Adam(learning_rate=1),
#               metrics=[keras.metrics.mean_squared_error])
# model.fit(x=x, y=y, batch_size=128, validation_split=0.2, epochs=10)

print(liner.w)
print(liner.b)