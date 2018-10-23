import torch as tc
import pandas as pd
from pylab import *

from datalib.data_utils import load_data
from models.SimilarityLoss import ConflateLoss
from models.CharCNN import CharCNN

from torch.optim.lr_scheduler import StepLR


if __name__ == "__main__":

    load_pretrained = False
    save_model = True
    saved_model_path = "models/char_cnn.pt"
    # train_path = 'datalib/data/pretraining/standard_train.csv'
    train_path = 'datalib/data/termtext/standard_train.csv'
    learning_rate = 0.01
    max_batch_size = 100
    n_epochs = 30
    update_i = 5000 / max_batch_size

    # Character embeddings
    char_embed = pd.read_csv('datalib/data/char_embeddings/char_embed_28D.csv').values

    # Training Data
    x_train = load_data(train_path=train_path)
    num_samples = x_train["anchor"].size()[0]

    model = CharCNN(embeddings=char_embed)

    if load_pretrained:
        print("loading pretrained")
        model.load_state_dict(tc.load(saved_model_path))

    model.train()

    objective = ConflateLoss()
    optimiser = tc.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimiser, step_size=1, gamma=0.9)

    loss_history = []
    train_acc_full = []

    for epoch in range(n_epochs):
        print("-------- Epoch {} --------".format(epoch))

        # shuffle the data at the start of each epoch
        epoch_permutation = tc.randperm(num_samples)

        for iter_i, i in enumerate(range(0, num_samples, max_batch_size)):

            batch_time = time.time()
            optimiser.zero_grad()

            x_minibatch = model.generate_mini_batch(i, max_batch_size, epoch_permutation, x_train)

            anchor_embedding, match_embedding = model.forward_match(x_minibatch)

            batch_loss = objective(anchor_embedding, match_embedding)
            batch_loss.backward()
            optimiser.step()

            if iter_i % update_i == 0:
                model.eval()

                train_acc = model.get_train_accuracy(x_minibatch)
                print("Epoch number {}\t Batch number {} / {}\t Current loss {:3.2f}\t Train acc {}\t Time: {:3.3f}".format(epoch, i, num_samples, batch_loss.item(), train_acc, time.time() - batch_time))
                train_acc_full.append(train_acc)
                loss_history.append(batch_loss.item())

                model.train()

    if save_model:
        tc.save(model.state_dict(), "models/char_cnn_termtext.pt")

    figure()
    subplot(211)
    plot(loss_history)
    subplot(212)
    plot(train_acc_full)
    grid()
    show()




