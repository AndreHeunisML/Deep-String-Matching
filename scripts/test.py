# Run data through a pretrained network

from pylab import *
import tensorflow as tf
from time import time
import pandas as pd
import datetime

from datalib.data_utils import load_data
from model.CharCnn import CharCnn


if __name__ == "__main__":

    #saved_model_path = "checkpoint/model.ckpt"
    saved_model_path = "checkpoint_big/model_2.ckpt"
    pretraining = True

    if pretraining:
        anchor_path = '../datalib/data/pretraining/standard_anchor.csv'
        match_path = '../datalib/data/pretraining/standard_match.csv'
        train_proportion = 0.998
    else:
        anchor_path = '../datalib/data/termtext/standard_anchor.csv'
        match_path = '../datalib/data/termtext/standard_match.csv'
        train_proportion = 0.0

    sequence_max_length = 32
    full_data = load_data(anchor_path, match_path, train_proportion)

    # Test Data
    x_test_anchor = full_data["test_anchor"]
    x_test_match = full_data["test_match"]
    test_anchor_indices = np.copy(x_test_anchor.anchor_index.values)
    num_test_anchors = len(test_anchor_indices)
    print("num_test_anchors: ", num_test_anchors)

    anchor_cols = ['anchor_{}'.format(ai) for ai in range(sequence_max_length)]
    match_cols = ['match_{}'.format(mi) for mi in range(sequence_max_length)]

    # ConvNet
    acc_list = [0]
    cnn = CharCnn(
        sequence_max_length=sequence_max_length,
        char_dict_size=29,
        embedding_size=56,
        output_embedding_size=256)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, saved_model_path)

    anchor = x_test_anchor[anchor_cols].values
    match = x_test_match[match_cols].values

    words_lookup = np.concatenate((anchor, match), axis=0)
    words = np.append(x_test_anchor.anchor.values, x_test_match.match.values)
    labels = np.append(x_test_anchor['anchor_index'].values, x_test_match['anchor_index'].values)

    constructed_batch_size = len(labels)
    print(constructed_batch_size)
    print(words)
    print(labels)

    feed_dict = {cnn.input_words: words_lookup,
                 cnn.match_label: labels,
                 cnn.is_training: False,
                 cnn.batch_size: constructed_batch_size}

    accuracy, min_indices, dist, diag = sess.run([cnn.accuracy, cnn.min_indices, cnn.distances, cnn.distdiag], feed_dict)

    print(dist)
    print(diag)

    print("Acc {:g}".format(accuracy))

    print('-' * 40)
    for iii, vvv in enumerate(min_indices):
        print('----')
        print('anchor: {}\t\t\tmatch: {}'.format(words[iii], words[vvv]))
        ai = labels[iii]
        print(words[np.where(labels == ai)])