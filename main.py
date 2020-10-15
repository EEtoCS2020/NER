import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
from data_util import DataManager
from crf import CRF
import numpy as np

embedding_dim=100
hidden_dim=100
max_sequence_length=300
batch_size = 64
dm = DataManager('./data/example_datasets1/', embedding_dim=embedding_dim, max_sequence_length=max_sequence_length)
querys_train, labels_train, querys_val, labels_val = dm.getTrainingSet(0.9)
num_classes = len(dm.label2id)

train_dataset = tf.data.Dataset.from_tensor_slices((querys_train, labels_train)).batch(batch_size=batch_size)
query_input = keras.Input(shape=(max_sequence_length), dtype='int32')
x = keras.layers.Embedding(len(dm.token2id), embedding_dim)(query_input)
forward_layer = keras.layers.LSTM(hidden_dim, return_sequences=True)
backward_layer = keras.layers.LSTM(hidden_dim, activation='relu', return_sequences=True, go_backwards=True)
x = keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer)(x)
# x = tf.reshape(x, [-1, max_sequence_length, hidden_dim * 2])
# x = tf.reshape(x, [-1, hidden_dim * 2])
crf = CRF()
x = crf(x)
model = keras.models.Model(inputs=query_input, outputs=x)


for (input_tensor, label) in train_dataset:
    print(input_tensor)
    print(label)
    with tf.GradientTape() as tape:
        output = model(input_tensor)
        loss, trans = tfa.text.crf_log_likelihood(output)
# model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#               metrics=['acc'])
#
# model.summary()
# keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
#
# model.fit(x=np.asarray(querys_train), y=np.asarray(labels_train), epochs=1000, batch_size=512,
#           validation_data=(np.asarray(querys_val), np.asarray(labels_val)))