
import torch as tc
import pandas as pd

from datalib.data_utils import load_data
from models.CharCNN import CharCNN


if __name__ == "__main__":

    # test_path = '../datalib/data/pretraining/standard_test.csv'
    test_path = '../datalib/data/termtext/standard_test.csv'
    saved_model_path = "../models/char_cnn_termtext.pt"

    # Character embeddings
    char_embed = pd.read_csv('../datalib/data/char_embeddings/char_embed_28D.csv').values

    x_test = load_data(test_path=test_path)
    num_samples = x_test["anchor_test"].size()[0]

    model = CharCNN(embeddings=char_embed)

    model.load_state_dict(tc.load(saved_model_path))

    model.eval()

    model.get_test_scores(x_test, True)