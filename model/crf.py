import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa

class CRF(keras.Model):
    def __init__(self, num_tags):
        super(CRF, self).__init__()
        initializer = tf.random_uniform_initializer(
            minval=-0.05, maxval=0.05, seed=None
        )
        self.trans_para = tf.Variable(initializer(shape=[num_tags, num_tags], dtype=tf.float32), trainable=True)

    def call(self, unary_score):
        return unary_score

    def nnl_loss(self, inputs):
        nnl_loss, self.trans_para = tfa.text.crf_decode(inputs[0], self.trans_para, inputs[1])




