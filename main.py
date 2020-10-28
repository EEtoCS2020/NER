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
querys_train, labels_train, q_train, querys_val, labels_val, q_val = dm.getTrainingSet(0.9)
lengths_train = [len(query) for query in querys_train]
lengths_val = [len(query) for query in querys_val]

print(querys_train[:5])
print(labels_train[:5])

querys_train = keras.preprocessing.sequence.pad_sequences(querys_train, maxlen=max_sequence_length, padding='post',value=0)
querys_val = keras.preprocessing.sequence.pad_sequences(querys_val, maxlen=max_sequence_length, padding='post',value=0)
# labels_train = keras.preprocessing.sequence.pad_sequences(labels_train , maxlen=max_sequence_length, padding='post',value=0)
# labels_val = keras.preprocessing.sequence.pad_sequences(labels_val, maxlen=max_sequence_length, padding='post',value=0)
num_classes = len(dm.label2id)

train_dataset = tf.data.Dataset.from_tensor_slices((querys_train, lengths_train, labels_train, q_train)).batch(batch_size=batch_size)
query_input = keras.Input(shape=(max_sequence_length), dtype='int32')
x = keras.layers.Embedding(len(dm.token2id), embedding_dim)(query_input)
forward_layer = keras.layers.LSTM(hidden_dim, return_sequences=True)
backward_layer = keras.layers.LSTM(hidden_dim, return_sequences=True, go_backwards=True)
x = keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer)(x)
# x = tf.reshape(x, [-1, max_sequence_length, hidden_dim * 2])
# x = tf.reshape(x, [-1, hidden_dim * 2])
crf = CRF(num_classes, hidden_dim*2)
x = crf(x)
model = keras.models.Model(inputs=query_input, outputs=x)
optimizer = keras.optimizers.Adam(learning_rate=0.001)

epochs = 100
for epoch in range(epochs):
    print(f"epoch {epoch + 1}")
    epoch_loss_avg = tf.keras.metrics.Mean()
    for (input_tensor, lens, label, q) in train_dataset:
        with tf.GradientTape() as tape:
            output = model(input_tensor)
            loss, crf.trans_param = tfa.text.crf_log_likelihood(output, label, lens, crf.trans_param)
            loss = -loss
            tv = model.trainable_variables
            tv.append(crf.trans_param)
            tags, scores = tfa.text.crf_decode(output, crf.trans_param, lens)
            if epoch == epochs-1:
                for i in range(len(q)):
                    print(q[i])
                    print(tags[i])
                    print('----------')
        grad = tape.gradient(loss, tv)
        epoch_loss_avg.update_state(loss)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
    print(f"loss: {epoch_loss_avg.result()}")