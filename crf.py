import tensorflow_addons as tfa
import tensorflow.keras as keras
import tensorflow as tf

crf_log_likelihood = tfa.text.crf_log_likelihood
crf_decode = tfa.text.crf_decode


class CRF(keras.layers.Layer):
    def __init__(self, num_tags, hidden_dim):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.hidden_dim = hidden_dim
        w_init = tf.random_normal_initializer(stddev=4)
        self.softmax_w = tf.Variable(name="softmax_w",
                                     initial_value=w_init(shape=(hidden_dim, num_tags), dtype="float32"),
                                     dtype='float32', trainable=True)
        self.softmax_b = tf.Variable(name="softmax_b",
                                     initial_value=w_init(shape=(num_tags,), dtype="float32"),
                                     dtype='float32', trainable=True)
        self.trans_param = tf.Variable(name="trans_param",
                                       initial_value=w_init(shape=(num_tags, num_tags), dtype="float32"),
                                       dtype='float32', trainable=True)


    def call(self, inputs, **kwargs):
        logits = tf.matmul(inputs, self.softmax_w) + self.softmax_b
        logits = tf.reshape(logits, [-1, inputs.shape[1], self.num_tags])
        return logits



