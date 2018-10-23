#

import pandas as pd

from text_match.dataprep.abb_data import get_all_abb
from text_match.dataprep.name_data import get_all_names
from text_match.dataprep.spelling_data import get_all_spell
from text_match.dataprep.stemlem_data import get_all_stemlem
from text_match.dataprep.prepare_char_embeddings import embed_data
from text_match.dataprep.termtext import get_all_termtext


def split_train_dev(data, train_ratio):
    """
    85 / 15

    :param data:
    :return:
    """
    total_count = len(data)
    train = int(train_ratio * total_count)

    return data.iloc[:train], data.iloc[train:]


if __name__ == "__main__":

    source = 'pretrain'
    #source = 'termtext'
    emb_dir = 'data/char_embeddings/char_to_index_28.json'

    if source == 'pretrain':
        all_data = pd.concat([get_all_abb(), get_all_names(), get_all_spell(), get_all_stemlem()])
        raw_dir = 'data/pretraining/raw.csv'
        out_train = 'data/pretraining/standard_train.csv'
        out_test = 'data/pretraining/standard_test.csv'

    elif source == 'termtext':
        all_data = get_all_termtext()
        raw_dir = 'data/termtext/raw.csv'
        out_train = 'data/termtext/standard_train.csv'
        out_test = 'data/termtext/standard_test.csv'


    # shuffle rows
    all_data = all_data.sample(frac=1)
    print(len(all_data))

    all_data.to_csv(raw_dir, mode='w+', index=False)

    aggregated = embed_data(raw_dir, emb_dir, min_len=2, max_len=36)
    print(len(aggregated))

    aggregated_train, aggregated_test = split_train_dev(aggregated, 0.9)
    print(len(aggregated_train))
    print(len(aggregated_test))

    aggregated_train.to_csv(out_train, mode='w+', index=False)
    aggregated_test.to_csv(out_test, mode='w+', index=False)









