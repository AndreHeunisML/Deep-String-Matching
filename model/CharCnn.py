
import tensorflow as tf
from tensorflow.layers import conv1d, dense, dropout
import numpy as np


class CharCnn:

    def __init__(self, sequence_max_length, char_dict_size, embedding_size, output_embedding_size):
        """

        :param sequence_max_length:
        :param char_dict_size:
        :param embedding_size:
        :param output_embedding_size:
        """
        # input tensors
        self.batch_size = tf.placeholder(tf.int32, (), name="batch_size")
        self.input_words = tf.placeholder(tf.int32, [None, sequence_max_length], name="input_anchor")
        self.match_label = tf.placeholder(tf.float32, [None], name="match_label")

        self.char_dict_size = char_dict_size
        self.embedding_size = embedding_size
        self.output_embedding_size = output_embedding_size
        self.is_training = tf.placeholder(tf.bool)

        with tf.variable_scope("siamese") as scope:
            self.anchor_embed = tf.nn.l2_normalize(self.network(self.input_words, self.is_training), 1)

        self.batch_loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(
            self.match_label,
            self.anchor_embed,
            margin=1.0)

        self.accuracy = self.calc_accuracy(self.anchor_embed, self.match_label)

    def network(self, x, is_training):
        # Embedding Lookup 16
        with tf.device('/cpu:0'), tf.name_scope("embedding"):

            # use_he_uniform:
            self.embedding_W = tf.get_variable(
                name='lookup_W',
                shape=[self.char_dict_size, self.embedding_size],     # TODO bigger embedding?
                initializer=tf.keras.initializers.he_uniform())

            self.embedded_characters = tf.nn.embedding_lookup(params=self.embedding_W, ids=x, name='embed_lu')
            print("-" * 20)
            print("Embedded Lookup:", self.embedded_characters.get_shape())
            print("-" * 20)

        # Temp(First) Conv Layer
        with tf.variable_scope("temp_conv"):
            print("-" * 20)
            print("Convolutional Block 1")
            print("-" * 20)

            he_std = np.sqrt(2 / (1 * 64))

            conv1 = conv1d(
                inputs=self.embedded_characters,
                filters=128,
                padding='valid',
                kernel_size=3,
                strides=2,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(stddev=he_std))
            #conv1 = dropout(conv1, rate=0.5, training=is_training)
            conv1 = tf.layers.batch_normalization(inputs=conv1, momentum=0.997, epsilon=1e-5,
                                                   center=True, scale=True, training=is_training)

            print("Conv 1", conv1.get_shape())

            print("-" * 20)
            print("Convolutional Block 2")
            print("-" * 20)

            conv2 = conv1d(
                inputs=conv1,
                filters=128,
                padding='valid',
                kernel_size=3,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(stddev=he_std))
            conv2 = tf.layers.batch_normalization(inputs=conv2, momentum=0.997, epsilon=1e-5,
                                                   center=True, scale=True, training=is_training)

            print("Conv 2", conv2.get_shape())

            print("-" * 20)
            print("Convolutional Block 3")
            print("-" * 20)

            conv3 = conv1d(
                inputs=conv2,
                filters=128,
                padding='valid',
                kernel_size=3,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(stddev=he_std))
            conv3 = tf.layers.batch_normalization(inputs=conv3, momentum=0.997, epsilon=1e-5,
                                                  center=True, scale=True, training=is_training)

            print("Conv 3", conv3.get_shape())

        flatten1 = tf.contrib.layers.flatten(conv3)

        print("-" * 20)
        print("Dense1")
        print("-" * 20)

        print("flatten", flatten1.get_shape())

        dense1 = dense(flatten1, 512, activation=tf.nn.relu, name='dense1')
        print("dense1", dense1.get_shape())

        dense2 = dense(dense1, self.output_embedding_size, activation=tf.nn.log_softmax, name='dense2')   #
        print("dense2", dense2.get_shape())

        return dense2

    def loss_contrastive(self, margin=15.0):

        # Define loss function
        with tf.variable_scope("loss_function") as scope:
            labels = self.match_label
            # Euclidean distance squared
            #eucd2 = tf.pow(tf.subtract(self.anchor_embed, self.match_embed), 2, name='eucd2')

            gamma = 10
            normalize_anchor = tf.nn.l2_normalize(self.anchor_embed, 1)
            normalize_pos_match = tf.nn.l2_normalize(self.match_embed, 1)
            eucd2 = tf.exp(
                gamma * tf.losses.cosine_distance(
                    normalize_anchor,
                    normalize_pos_match,
                    axis=1,
                    reduction=tf.losses.Reduction.NONE))

            # Loss function
            self.diss = tf.squeeze(eucd2)
            self.dissred = tf.multiply(labels, tf.squeeze(eucd2))

            self.loss_neg = tf.reduce_sum(tf.squeeze(eucd2), name='constrastive_loss_2')

            self.loss_pos = tf.divide(tf.squeeze(eucd2), self.loss_neg)

            loss = -1 * tf.reduce_sum(tf.multiply(labels, tf.log(self.loss_pos)))

        return loss

    def calc_accuracy(self, word_embed, anchor_ind):
        '''


        :param word_embed:
        :param anchor_ind:
        :return:
        '''
        diag = tf.ones(self.batch_size)
        dot_product = tf.matmul(word_embed, tf.transpose(word_embed))

        square_norm = tf.diag_part(dot_product)
        distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        self.distances = tf.maximum(distances, 0.0)

        self.distdiag = tf.linalg.set_diag(self.distances, diag, name=None)
        self.min_indices = tf.argmin(self.distdiag, axis=1, output_type=tf.int64)

        self.matches = tf.equal(anchor_ind, tf.gather(anchor_ind, self.min_indices))
        match_count = tf.reduce_sum(tf.cast(self.matches, tf.float32))

        return match_count / tf.cast(self.batch_size, tf.float32)
