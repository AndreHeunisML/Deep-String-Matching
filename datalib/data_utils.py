
import pandas as pd
import numpy as np


def load_data(anchor_path, match_path, train_proportion):
    print("Loading data...")

    anchor = pd.read_csv(anchor_path)
    match = pd.read_csv(match_path)

    print('full anchor length: ', len(anchor))
    print('full match length: ', len(match))

    total_anchor_count = len(anchor)
    anchor_index = np.copy(anchor['anchor_index'].values)
    np.random.shuffle(anchor_index)

    index_lim = int(train_proportion * total_anchor_count)
    train_index = anchor_index[:index_lim]
    test_index = anchor_index[index_lim:]

    train_anchor = anchor[anchor.anchor_index.isin(train_index)]
    train_match = match[match['anchor_index'].isin(train_index)]

    test_anchor = anchor[anchor['anchor_index'].isin(test_index)]
    test_match = match[match['anchor_index'].isin(test_index)]

    print("Train and test counts")
    print(len(train_anchor))
    print(len(test_anchor))
    print(len(train_match))
    print(len(test_match))


    return {"train_anchor": train_anchor,
            "test_anchor": test_anchor,
            "train_match": train_match,
            "test_match": test_match}