import tensorflow_addons as tfa
import tensorflow.keras as keras
import tensorflow as tf

crf_log_likelihood = tfa.text.crf_log_likelihood
crf_decode = tfa.text.crf_decode


class CRF(keras.layers.Layer):
    def __int__(self):
        super(CRF, self).__init__()
        self.initializer = tf.keras.initializers.RandomNormal()
        self.softmax_w = tf.Variable("softmax_w", [200, 14],
                                     initializer=self.initializer)
        self.softmax_b = tf.Variable("softmax_b", [14], initializer=self.initializer)

    def build(self, input_shape):
        # self.trans_params = self.add_weight(shape=(14, 14), initializer='random_normal',
        #                                     trainable=True)
        self.softmax_w = self.add_weight(shape=(200, 14),
                                     initializer='random_normal')
        self.softmax_b = self.add_weight(shape=(14,), initializer='random_normal')

    def call(self, inputs, **kwargs):
        print(inputs.shape)
        logits = tf.matmul(inputs, self.softmax_w) + self.softmax_b
        logits = tf.reshape(logits, [-1, 300, 14])
        return logits



