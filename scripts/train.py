

from pylab import *
import tensorflow as tf
from time import time
import pandas as pd
import datetime

from datalib.data_utils import load_data
from model.CharCnn import CharCnn


def run_test(print_words=False):
    print("----- TEST RESULTS -----")

    np.random.shuffle(test_anchor_indices)
    full_accuracy = []
    full_loss = []

    # split anchor indices into batches
    for t in range(num_test_batches):
        # select all anchors and matches that correspond to batch anchor indices
        test_batch_anchor_indices = test_anchor_indices[t * batch_size: min([(t + 1) * batch_size, num_test_anchors])]

        x_test_anchor_batch = x_test_anchor[x_test_anchor.anchor_index.isin(test_batch_anchor_indices)]
        x_test_match_batch = x_test_match[x_test_match.anchor_index.isin(test_batch_anchor_indices)]

        # triplet
        anchor_test = x_test_anchor_batch[anchor_cols].values
        match_test = x_test_match_batch[match_cols].values

        test_words_lookup = np.concatenate((anchor_test, match_test), axis=0)
        words = np.append(x_test_anchor_batch.anchor.values, x_test_match_batch.match.values)
        labels = np.append(x_test_anchor_batch['anchor_index'].values, x_test_match_batch['anchor_index'].values)

        test_constructed_batch_size = len(labels)
        if test_constructed_batch_size > max_batch_size:
            test_words_lookup = test_words_lookup[:max_batch_size]
            labels = labels[:max_batch_size]
            test_constructed_batch_size = max_batch_size

        feed_dict = {cnn.input_words: test_words_lookup,
                     cnn.match_label: labels,
                     cnn.is_training: False,
                     cnn.batch_size: test_constructed_batch_size}

        loss, test_batch_accuracy, min_indices = sess.run([cnn.batch_loss, cnn.accuracy, cnn.min_indices], feed_dict)

        full_accuracy.append(test_batch_accuracy)
        full_loss.append(loss)

    time_str = datetime.datetime.now().isoformat()
    test_accuracy = mean(full_accuracy)

    print("{}: Loss {:g}, Acc {:g}".format(time_str, mean(full_loss), test_accuracy))

    if print_words:
        print('-' * 40)
        for iii, vvv in enumerate(min_indices):
            print('{}\t\t\t{}'.format(words[iii], words[vvv]))

    return test_accuracy


if __name__ == "__main__":

    tf.reset_default_graph()
    load_pretrained = False
    save_model = False
    saved_model_path = "model_2.ckpt"
    pretraining = True

    if pretraining:
        anchor_path = 'datalib/data/pretraining/standard_anchor.csv'
        match_path = 'datalib/data/pretraining/standard_match.csv'
        n_epochs = 12
        train_proportion = 0.998
        train_update_batches = 1
        test_update_batches = 4
    else:
        anchor_path = 'datalib/data/termtext/standard_anchor.csv'
        match_path = 'datalib/data/termtext//standard_match.csv'
        n_epochs = 12
        train_proportion = 0.9
        train_update_batches = 5
        test_update_batches = 10

    sequence_max_length = 32
    learning_rate = 0.001
    batch_size = 128
    max_batch_size = 512
    n_epochs = 12
    plot_batch_loss = []
    plot_train_acc = []
    plot_test_acc = []

    # Training Data
    full_data = load_data(anchor_path, match_path, train_proportion)

    train_anchor_indices = np.copy(full_data["train_anchor"].anchor_index.values)
    print("Train anchor index mismatch: ",
          set(full_data["train_match"].anchor_index.values) - set(full_data["train_anchor"].anchor_index.values))
    num_anchors = len(train_anchor_indices)
    num_batches = int(num_anchors / batch_size) + 1
    print("num_anchors: ", num_anchors)
    print("num_batches per epoch: ", num_batches)

    # Test Data
    x_test_anchor = full_data["test_anchor"]
    x_test_match = full_data["test_match"]
    test_anchor_indices = np.copy(x_test_anchor.anchor_index.values)
    num_test_anchors = len(test_anchor_indices)
    num_test_batches = int(num_test_anchors / batch_size) + 1
    print("num_test_anchors: ", num_test_anchors)
    print("num_test_batches per test run: ", num_test_batches)

    # ConvNet
    acc_list = [0]
    cnn = CharCnn(
        sequence_max_length=sequence_max_length,
        char_dict_size=29,
        embedding_size=56,
        output_embedding_size=256)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads_and_vars = optimizer.compute_gradients(cnn.batch_loss)
    train_op = optimizer.apply_gradients(grads_and_vars)

    anchor_cols = ['anchor_{}'.format(ai) for ai in range(sequence_max_length)]
    match_cols = ['match_{}'.format(mi) for mi in range(sequence_max_length)]

    # Initialize Graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if save_model or load_pretrained:
        saver = tf.train.Saver()

    if load_pretrained:
        saver.restore(sess, saved_model_path)

    # Training loop. For each batch...
    print("Training...")
    for e in range(n_epochs):
        print('-' * 20)
        print('Epoch: ', e)
        print('-' * 20)
        start_time = time()

        np.random.shuffle(train_anchor_indices)

        # split anchor indices into batches
        for b in range(num_batches):

            # select all anchors and matches that correspond to batch anchor indices
            batch_anchor_indices = train_anchor_indices[b * batch_size: min([(b + 1) * batch_size, num_anchors])]

            # anchor data
            x_train_anchor = full_data["train_anchor"][
                full_data["train_anchor"].anchor_index.isin(batch_anchor_indices)]

            # match data
            x_train_match = full_data["train_match"][
                full_data["train_match"].anchor_index.isin(batch_anchor_indices)]

            # triplet
            anchor_train = x_train_anchor[anchor_cols].values
            match_train = x_train_match[match_cols].values

            words_lookup = np.concatenate((anchor_train, match_train), axis=0)
            words = np.append(x_train_anchor.anchor.values, x_train_match.match.values)
            labels = np.append(x_train_anchor['anchor_index'].values, x_train_match['anchor_index'].values)

            constructed_batch_size = len(labels)
            if constructed_batch_size > max_batch_size:
                words_lookup = words_lookup[:max_batch_size]
                labels = labels[:max_batch_size]
                constructed_batch_size = max_batch_size

            feed_dict = {cnn.input_words: words_lookup,
                         cnn.match_label: labels,
                         cnn.is_training: True,
                         cnn.batch_size: constructed_batch_size}

            _, loss, accuracy, min_indices = sess.run([train_op, cnn.batch_loss, cnn.accuracy, cnn.min_indices], feed_dict)

            if b % train_update_batches == 0:

                time_str = datetime.datetime.now().isoformat()
                print("{}: Batch {}, Loss {:g}, Acc {:g}".format(time_str, b, loss, accuracy))
                plot_train_acc.append(accuracy)
                plot_batch_loss.append(loss)

            if (b + 1) % test_update_batches == 0:
                test_acc = run_test()
                plot_test_acc.append(test_acc)

    # Save the model to file
    if save_model:
        save_path = saver.save(sess, saved_model_path)

    # Save acc and loss to file
    train_metrics = pd.DataFrame(data=np.array([plot_train_acc, plot_batch_loss]).T, columns=['acc', 'loss'])
    train_metrics.to_csv("train_metrics.csv", mode='w+', index=False)
    test_metrics = pd.DataFrame(data=np.array([plot_test_acc]).T, columns=['acc'])
    test_metrics.to_csv("test_metrics.csv", mode='w+', index=False)

    figure()
    subplot(311)
    plot(plot_batch_loss)
    subplot(312)
    plot(plot_train_acc)
    subplot(313)
    plot(plot_test_acc)
    show()



