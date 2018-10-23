

import json
import pandas as pd
import torch as tc


class StringPrinter:

    def __init__(self):
        char_to_index_fpath = '../datalib/data/char_embeddings/char_to_index_28.json'
        with open(char_to_index_fpath) as _:
            self.char_to_index = json.load(_)

        self.index_to_char = dict((y, x) for x, y in self.char_to_index.items())

    def get_string_from_embedding(self, em):
        return ''.join([self.index_to_char[i] for i in em.numpy() if i != 0])


def load_data(train_path=None, test_path=None):
    print("Loading data...")

    anchor_train = None
    positive_train = None
    anchor_test = None
    positive_test = None

    if train_path is not None:
        train = pd.read_csv(train_path)
        print("{} train samples".format(train.shape))
        anchor_cols = [col for col in train.columns if 'anchor' in col and 'length' not in col]
        match_cols = [col for col in train.columns if 'match' in col and 'length' not in col]
        anchor_train = tc.from_numpy(train[anchor_cols].values).long()
        positive_train = tc.from_numpy(train[match_cols].values).long()

    if test_path is not None:
        test = pd.read_csv(test_path)
        print("{} test samples".format(test.shape))
        anchor_cols = [col for col in test.columns if 'anchor' in col and 'length' not in col]
        match_cols = [col for col in test.columns if 'match' in col and 'length' not in col]
        anchor_test = tc.from_numpy(test[anchor_cols].values).long()
        positive_test = tc.from_numpy(test[match_cols].values).long()

    return {"anchor": anchor_train,
            "match": positive_train,
            "anchor_test": anchor_test,
            "match_test": positive_test}