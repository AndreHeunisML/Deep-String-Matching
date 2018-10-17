import torch as tc
import pandas as pd
from pylab import *

from SimilarityLoss import ConflateLoss
from TermtextEncoder import TermtextEncoder


def load_data(train, test):
    """

    :return:    PyTorch tensors
    """
    print("Loading data...")

    train = pd.read_csv(train)
    test = pd.read_csv(test)
    print("{} samples".format(train.shape))
    print("{} test samples".format(test.shape))

    anchor_cols = [col for col in train.columns if 'anchor' in col and 'length' not in col]
    match_cols = [col for col in train.columns if 'match' in col and 'length' not in col]

    anchor = tc.from_numpy(train[anchor_cols].values).long()
    positive = tc.from_numpy(train[match_cols].values).long()

    anchor_test = tc.from_numpy(test[anchor_cols].values).long()
    positive_test = tc.from_numpy(test[match_cols].values).long()

    return {"anchor": anchor,
            "match": positive,
            "anchor_test": anchor_test,
            "match_test": positive_test}


if __name__ == "__main__":

    train = True
    load_pretrained = True
    save_model = False
    # train_path = 'dataprep/data/pretraining/standard_train.csv'
    # test_path = 'dataprep/data/pretraining/standard_test.csv'
    train_path = 'dataprep/data/termtext/standard_train.csv'
    test_path = 'dataprep/data/termtext/standard_test.csv'

    # Character embeddings
    char_embed = pd.read_csv('dataprep/data/char_embeddings/char_embed_28D.csv').values
    vocab_size = len(char_embed)
    embedding_dim = char_embed.shape[1]

    x_train = load_data(train_path, test_path)
    train_shape = x_train["anchor"].size()
    num_samples = train_shape[0]

    # define a network
    net = TermtextEncoder(
        vocab_size=vocab_size,
        embeddings=char_embed,
        embedding_dim=embedding_dim)

    if load_pretrained:
        print("loading pretrained")
        net.load_state_dict(tc.load("models/char_cnn.pt"))

    if train:
        net.train()

        objective = ConflateLoss()
        optimiser = tc.optim.Adam(net.parameters(), lr=0.001)

        loss_history = []
        train_acc_full = []
        max_batch_size = 100
        n_epochs = 6
        update_i = 1000 / max_batch_size

        for epoch in range(n_epochs):
            print("-------- Epoch {} --------".format(epoch))

            # shuffle the data at the start of each epoch
            epoch_permutation = tc.randperm(num_samples)

            for iter_i, i in enumerate(range(0, num_samples, max_batch_size)):

                batch_time = time.time()
                optimiser.zero_grad()

                x_minibatch = net.generate_mini_batch(i, max_batch_size, epoch_permutation, x_train)

                anchor_embedding, match_embedding = net.forward_match(x_minibatch)

                batch_loss = objective(anchor_embedding, match_embedding)
                batch_loss.backward()
                optimiser.step()

                if iter_i % update_i == 0:
                    # Set to eval mode to get acc
                    net.eval()

                    train_acc = net.get_train_accuracy(x_minibatch)
                    print("Epoch number {}\t Batch number {} / {}\t Current loss {:3.2f}\t Train acc {}\t Time: {:3.3f}".format(epoch, i, num_samples, batch_loss.item(), train_acc, time.time() - batch_time))
                    train_acc_full.append(train_acc)
                    loss_history.append(batch_loss.item())

                    # Set back to train mode
                    net.train()

        tc.save(net.state_dict(), "models/char_cnn.pt")
        #net = tc.load("models/char_cnn.pt")

        figure()
        subplot(211)
        plot(loss_history)
        subplot(212)
        plot(train_acc_full)
        grid()
        show()

    net.eval()
    net.get_test_scores(test_inputs=x_train, print_strings=True)


