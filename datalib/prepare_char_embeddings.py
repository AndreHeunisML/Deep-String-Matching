

import pandas as pd
import json
import re
from pylab import *


def clean_strings(match_df, min_len, max_len):

    print("Starting size: ", len(match_df))

    match_df = match_df.reset_index(drop=True)

    clean_match_list = []
    anchor_match_list = []

    for (i, row) in match_df.iterrows():

        if i % 5000 == 0:
            print(i)

        anchor_char = row['anchor']
        match_char = row['match']

        if isinstance(anchor_char, float) or isinstance(match_char, float):
            print(anchor_char)
            print(match_char)
            anchor_char = None
            match_char = None
        else:

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

            if max_len < len(anchor_char) or \
                    len(anchor_char) < min_len or \
                    max_len < len(match_char) or \
                    len(match_char) < min_len or \
                    anchor_char == match_char or \
                    'nan' in anchor_char or 'null' in anchor_char or 'nan' in match_char or 'null' in match_char or \
                    len(set(anchor_char)) < 3 or len(set(match_char)) < 3:
                anchor_char = None
                match_char = None

        anchor_match_list.append(anchor_char)
        clean_match_list.append(match_char)

    match_df["anchor"] = anchor_match_list
    match_df["match"] = clean_match_list

    # filter out all Nones
    print("Cleaned size with Nones: ", len(match_df))
    match_df = match_df[~match_df.anchor.isnull()]
    print("Cleaned size: ", len(match_df))

    return match_df


def embed_data(data_dir, doc_column, char_to_index_fpath, max_len):

    # Read data
    raw_data = pd.read_csv(data_dir)
    print("Number of unique doc_column: ", len(raw_data[doc_column].unique()))

    # Read char map
    with open(char_to_index_fpath) as _:
        char_to_index = json.load(_)

    num_samples = len(raw_data)
    char_embed = np.zeros((num_samples, max_len))
    docs = raw_data[doc_column].values
    docsindex = raw_data['anchor_index'].values

    nancount = 0

    for wi, w in enumerate(docs):
        if not isinstance(w, str):
            print("nan")
            print(docsindex[wi])
            nancount += 1
            continue

        for ci, c in enumerate(w):
            char_embed[wi, ci] = char_to_index[c]

    # Quality check
    print("nan count: ", nancount)

    embed_df = pd.DataFrame(np.concatenate([char_embed], axis=1),
                             columns=[(doc_column + '_{}').format(i) for i in range(max_len)])

    return pd.concat([raw_data, embed_df], axis=1)


if __name__ == "__main__":

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

