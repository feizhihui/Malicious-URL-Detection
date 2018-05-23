# encoding=utf-8
import os

import numpy as np
import tensorflow as tf
from sklearn import metrics

from data_input_benchmark import DataMaster
from sequence_model_benchmark import SeqModel

os.environ["CUDA_VISIABLE_DEVICES"] = "1"

batch_size = 1024

THRESHOLD = 0.2

master = DataMaster(train_mode=False)
model = SeqModel()

logits = []
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, '../Cache/sequence_model')
    for step, index in enumerate(range(0, master.datasize, batch_size)):
        batch_xs, batch_lens = master.mapping(master.train_x[index:index + batch_size])
        batch_es = master.train_e[index:index + batch_size]
        logit = sess.run(model.activation_logits,
                         feed_dict={model.x: batch_xs, model.e: batch_es,
                                    model.batch_lens: batch_lens})

        logits.extend(logit)
    preds = (np.array(logits, dtype=np.float32) > THRESHOLD).astype(np.int32)
    print("Precision %.6f" % metrics.precision_score(master.train_y, preds))
    print("Recall %.6f" % metrics.recall_score(master.train_y, preds))
    print("F1-score %.6f" % metrics.f1_score(master.train_y, preds))
    print("AUC %.6f" % metrics.roc_auc_score(master.train_y, logits))
