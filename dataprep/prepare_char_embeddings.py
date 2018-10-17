

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import json
import re
import seaborn as sns
from pylab import *


# embed training data
def embed_data(fpath, char_to_index_fpath, min_len, max_len):
    """

    1. Find the max length after cleaning
    3. Pad to max len

    :param fpath: path to csv with columns 'anchor' and 'pos_match'
    :return:
    """
    # Read data
    raw_train_data = pd.read_csv(fpath)

    # Read char map
    with open(char_to_index_fpath) as _:
        char_to_index = json.load(_)

    print('Clean data and find data parameters')
    anchor_char_full = []
    match_char_full = []
    anchor_lengths = []
    match_lengths = []

    for (i, row) in raw_train_data.iterrows():
        anchor_char = row['anchor']
        match_char = row['pos_match']

        try:
            anchor_char = anchor_char.lower()
            match_char = match_char.lower()

            match_char = re.sub('\s{3,}', '___', match_char)
            anchor_char = re.sub('\s{3,}', '___', anchor_char)

            match_char = re.sub('\d+', '0', match_char)
            anchor_char = re.sub('\d+', '0', anchor_char)

            match_char = re.sub('\s', '_', match_char)
            anchor_char = re.sub('\s', '_', anchor_char)

            anchor_char = re.sub('[^a-z0-9_]', '', anchor_char)
            match_char = re.sub('[^a-z_]', '', match_char)

            anchor_char = anchor_char[:max_len]
            match_char = match_char[:max_len]
        except:
            continue

        if len(anchor_char) < min_len or len(match_char) < min_len:
            continue

        anchor_char_full.append(anchor_char)
        match_char_full.append(match_char)
        anchor_lengths.append(len(anchor_char))
        match_lengths.append(len(match_char))

    # figure()
    # subplot(211)
    # sns.distplot(anchor_lengths)
    # subplot(212)
    # sns.distplot(match_lengths)
    # show()

    num_samples = len(anchor_char_full)
    anchor_char_final = np.zeros((num_samples, max_len))
    match_char_final = np.zeros((num_samples, max_len))

    for wi, _ in enumerate(anchor_char_full):
        for ci, c in enumerate(anchor_char_full[wi]):
            anchor_char_final[wi, ci] = char_to_index[c]

        for ci, c in enumerate(match_char_full[wi]):
            match_char_final[wi, ci] = char_to_index[c]

    anchor_df = pd.DataFrame(np.concatenate([anchor_char_final], axis=1),
                             columns=['anchor_{}'.format(i) for i in range(max_len)])

    pos_df = pd.DataFrame(np.concatenate([match_char_final], axis=1),
                             columns=['match_{}'.format(i) for i in range(max_len)])

    return pd.concat([anchor_df, pos_df], axis=1)


if __name__ == "__main__":
    use_pretrained = False

    if use_pretrained:
        # https://github.com/minimaxir/char-embeddings/blob/master/text_generator_keras.py
        char_to_index_fpath = 'data/char_embeddings/char_to_index.json'
        embed_fpath = 'data/char_embeddings/char_embed_50D.csv'
        embeddings_path = "data/char_embeddings/glove.840B.300d-char.txt"
        chars = []
        embedding_vectors = np.zeros((94, 300))     # add extra row for 0 pad after pca
        char_to_index = {}

        with open(embeddings_path, 'r') as f:
            for i, line in enumerate(f):
                line_split = line.strip().split(" ")

                char = line_split[0]
                chars.append(char)

                embedding_vectors[i, :] = np.array(line_split[1:], dtype=float)
                char_to_index[char] = i + 1

        print(char_to_index)
        print(embedding_vectors.shape)

        # Use PCA from sklearn to reduce
        embedding_dim = 300
        # pca = PCA(n_components=embedding_dim)
        # embedding_vectors = pca.fit_transform(embedding_vectors)
        print(embedding_vectors.shape)
        embedding_vectors = np.concatenate([np.zeros((1, embedding_dim)), embedding_vectors], axis=0)
        print(embedding_vectors.shape)
    else:
        char_to_index_fpath = 'data/char_embeddings/char_to_index_28.json'
        embed_fpath = 'data/char_embeddings/char_embed_28D.csv'

        char_to_index = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8, "i": 9, "j": 10, "k": 11,
                         "l": 12, "m": 13, "n": 14, "o": 15, "p": 16, "q": 17, "r": 18, "s": 19, "t": 20, "u": 21,
                         "v": 22, "w": 23, "x": 24, "y": 25, "z": 26, "_": 27, "0": 28}

        # input = tc.from_numpy(input).long()
        embedding_dim = 28
        embedding_vectors = np.eye(embedding_dim)
        embedding_vectors = np.concatenate([np.zeros((1, embedding_dim)), embedding_vectors], axis=0)

    # Write embeddings to file
    with open(char_to_index_fpath, 'w+') as _:
        json.dump(char_to_index, _)

    pd.DataFrame(embedding_vectors).to_csv(embed_fpath, mode='w+', index=False)

