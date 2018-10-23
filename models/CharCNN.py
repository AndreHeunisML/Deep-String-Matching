
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from pylab import *

from datalib.data_utils import StringPrinter


class CharCNN(nn.Module):
    def __init__(self, embeddings):

        super(CharCNN, self).__init__()
        self.embeddings = embeddings
        self.vocab_size = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]
        self.dropout = 0.5

        self.word_embedding = nn.Embedding.from_pretrained(tc.FloatTensor(embeddings))

        self.conv1 = nn.Sequential(
            nn.Conv1d(self.embedding_dim, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=self.dropout)
        )

        self.fc2 = nn.Linear(512, 256)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        """


        :param x:           Mini batch input. Tensor of shape (batch_size, padded_seq_len, 1). Sorted from longest to
                            shortest actual length as a requirement for pack_padded_sequence
        :param x_lengths:   The unpadded lengths for each sequence in x
        :return:
        """
        batch_size, _ = x.size()
        x = self.word_embedding(x).view(batch_size, self.embedding_dim, -1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.log_softmax(x)

        return x

    def forward_match(self, x_minibatch):
        """

        :param x_minibatch:
        :return:
        """

        anchor_embedding = self.forward(x_minibatch['anchor'])
        match_embedding = self.forward(x_minibatch['match'])

        return anchor_embedding, match_embedding

    def generate_mini_batch(self, batch_i, batch_size, epoch_permutation, x_train):
        """


        :param all_data:    Pandas DF of the data to batch
        :param batch_size:  Size of each batch
        :param epoch_i:     Index of the epoch batches are being created for
        :return:            Torch tensor of shape (seq_len, batch, input_size)
        """
        batch_upper_lim = min([batch_i + batch_size, len(epoch_permutation)])
        indices = epoch_permutation[batch_i:batch_upper_lim]

        # Select the anchor and positive match for the batch
        anchor = x_train['anchor'][indices]
        match = x_train['match'][indices]

        return {"anchor": anchor,
                "match": match}

    def rank_similar(self, x_minibatch):
        anchor_embedding = self.forward(x_minibatch['anchor'])
        match_embedding = self.forward(x_minibatch['match'])

        evaluator = StringPrinter()
        correct = 0.0
        for i, a in enumerate(x_minibatch['anchor']):
            similarity = F.cosine_similarity(anchor_embedding[i].view(1, -1), match_embedding)

            similarity_sorted, idx = similarity.sort(dim=0, descending=True)

            if tc.all(tc.eq(x_minibatch['match'][idx][0], x_minibatch['match'][i])):
                correct += 1.0
            else:
                print("---------------------------")
                print(evaluator.get_string_from_embedding(a))

                for i in range(3):
                    print("{}\t {}".format(evaluator.get_string_from_embedding(x_minibatch['match'][idx][i]), similarity_sorted[i]))

        print(correct / x_minibatch['anchor'].size()[0])

    def get_train_accuracy(self, x_minibatch):
        anchor_embedding = self.forward(x_minibatch['anchor'])
        match_embedding = self.forward(x_minibatch['match'])

        correct = 0.0
        for i, a in enumerate(x_minibatch['anchor']):
            similarity = F.cosine_similarity(anchor_embedding[i].view(1, -1), match_embedding)

            similarity_sorted, idx = similarity.sort(dim=0, descending=True)

            if tc.all(tc.eq(x_minibatch['match'][idx][0], x_minibatch['match'][i])):
                correct += 1.0

        return correct / x_minibatch['anchor'].size()[0]

    def get_test_scores(self, test_inputs, print_strings):
        print("Returning full test set scores")

        # Compare each anchor against every match in the test set
        anchor_embedding = self.forward(test_inputs['anchor_test'])
        match_embedding = self.forward(test_inputs['match_test'])

        evaluator = StringPrinter()
        correct = 0.0
        for i, a in enumerate(test_inputs['anchor_test']):
            if print_strings:
                print("---------------------------")
                print(evaluator.get_string_from_embedding(a))

            similarity = F.cosine_similarity(anchor_embedding[i].view(1, -1), match_embedding)

            similarity_sorted, idx = similarity.sort(dim=0, descending=True)

            if tc.all(tc.eq(test_inputs['match_test'][idx][0], test_inputs['match_test'][i])):
                if print_strings: print("***")
                correct += 1.0

            if print_strings:
                for i in range(3):
                    print("{}\t {}".format(evaluator.get_string_from_embedding(test_inputs['match_test'][idx][i]), similarity_sorted[i]))

        print("test accuracy: {}".format(correct / test_inputs['anchor_test'].size()[0]))