# encoding=utf-8

# encoding=utf-8
import data_input
import tensorflow as tf
import numpy as np
import sequence_model
from sklearn import metrics
import warnings

warnings.filterwarnings("ignore")

filename = "../Data/29_1.txt"
train_eval_split_line = 0.9
batch_size = 9600
show_step = 200
filemark = ""
master = data_input.DataMaster(train_mode=False, test_file=filemark)

model = sequence_model.SeqModel()

dataset = []
with open(filename, "r") as file:
    for url in file.readlines():
        url = url.strip()
        dataset.append(url)

saver = tf.train.Saver()

y_pred = []
with tf.Session() as sess:
    saver.restore(sess, '../Cache/sequence_model')
    sess.run(tf.local_variables_initializer())
    for step, index in enumerate(range(0, len(dataset), batch_size)):
        batch_xs, batch_lens = master.mapping(dataset[index:index + batch_size])
        predictions = sess.run(model.prediction,
                               feed_dict={model.x: batch_xs,
                                          model.batch_lens: batch_lens})
        y_pred.extend(predictions.tolist())

print(y_pred)

with open(filename, "r") as file, open(filename[:-4] + "_pred.txt", "w") as fileWriter:
    for i, url in enumerate(file.readlines()):
        url = url.strip()
        mark = "malicious" if y_pred[i] else "benign"
        fileWriter.write(url + "\t" + mark + "\n")
    print("%s write done!" % (filename[:-4] + "pred.txt"))
