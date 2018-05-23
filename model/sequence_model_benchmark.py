# encoding=utf-8
import tensorflow as tf
import numpy as np

n_hidden = 60
feature_embedding = 9
feature_layer = 32
sequence_lens = 150
url_char_scope = 59 + 1  # padding is zero(+1)

class_num = 2
keep_prob = 0.9
lr = 0.01

THRESHOLD = 0.2


class SeqModel(object):
    def __init__(self):
        self.x = tf.placeholder(tf.int32, [None, sequence_lens])
        self.e = tf.placeholder(tf.float32, [None, feature_embedding])
        self.y = tf.placeholder(tf.int32, [None])
        self.batch_lens = tf.placeholder(tf.int32, [None])

        input_x = tf.one_hot(self.x, depth=url_char_scope)
        # Current data input shape: (batch_size, n_steps, n_input)
        # Forward direction cell
        lstm_fw_cell = tf.contrib.rnn.LSTMCell(n_hidden)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=keep_prob)

        # network = rnn_cell.MultiRNNCell([lstm_fw_cell, lstm_bw_cell] * 3)
        # x shape is [batch_size, max_time, input_size]
        outputs, output_state = tf.nn.dynamic_rnn(lstm_fw_cell, input_x,
                                                  sequence_length=self.batch_lens,
                                                  dtype=tf.float32)
        # if RNNCell,GRUCell
        # outputs=output_sate
        # if LSTMCell
        output_rnn = output_state.h

        with tf.name_scope('feature_embedding_layer'):
            output_fn = tf.layers.dense(self.e, feature_layer, activation=tf.nn.relu)

        output_concat = tf.concat([output_rnn, output_fn], axis=1)
        with tf.name_scope("sigmoid_layer"):
            weights = tf.Variable(
                tf.truncated_normal([n_hidden + feature_layer, class_num]) * np.sqrt(
                    2.0 / (2 * (n_hidden + feature_layer))),
                dtype=tf.float32)
            bias = tf.Variable(tf.zeros([1, class_num]), dtype=tf.float32)
            logits = tf.matmul(output_concat, weights) + bias
            self.activation_logits = tf.nn.sigmoid(logits)[:, 1]

        with tf.name_scope("evaluation"):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.y, depth=2), logits=logits)
            self.cost = tf.reduce_mean(loss)
            self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.cost)
            # self.prediction = tf.arg_max(logits, dimension=1)
            self.prediction = tf.cast(tf.greater(logits, THRESHOLD), tf.int32)[:, 1]
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.prediction, tf.int32), self.y), tf.float32))
            # self.auc, self.auc_opt = tf.contrib.metrics.streaming_auc(self.activation_logits, self.y)
