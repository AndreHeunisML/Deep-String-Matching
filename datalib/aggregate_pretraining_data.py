#

import pandas as pd
import numpy as np
import time

from datalib.abb_data import get_all_abb
from datalib.name_data import get_all_names
from datalib.spelling_data import get_all_spell
from datalib.stemlem_data import get_all_stemlem
from datalib.prepare_char_embeddings import embed_data, clean_strings
from datalib.termtext import get_all_termtext
from datalib.news_data import get_all_news


if __name__ == "__main__":

    start = time.time()

    #source = 'pretrain'
    source = 'pretrain'
    emb_dir = 'data/char_embeddings/char_to_index_28.json'
    min_len = 3
    max_len = 32

    if source == 'pretrain':
        all_data = pd.concat([get_all_abb(), get_all_spell(), get_all_names(), get_all_stemlem(), get_all_news()])
        raw_anchors_dir = 'data/pretraining/raw_anchors.csv'
        raw_matches_dir = 'data/pretraining/raw_matches.csv'
        out_anchor = 'data/pretraining/standard_anchor.csv'
        out_match = 'data/pretraining/standard_match.csv'

    elif source == 'termtext':
        all_data = get_all_termtext()
        raw_dir = 'data/termtext/raw.csv'
        raw_anchors_dir = 'data/termtext/raw_anchors.csv'
        raw_matches_dir = 'data/termtext/raw_matches.csv'
        out_anchor = 'data/termtext/standard_anchor.csv'
        out_match = 'data/termtext/standard_match.csv'

    print("Cleaning input strings")
    all_data = clean_strings(all_data, min_len, max_len)

    # Split into DFs of unique anchors and matches
    all_data["anchor_index"] = range(len(all_data))
    anchor_distinct = all_data[["anchor", "anchor_index"]].groupby("anchor").first().reset_index()
    anchor_distinct.reset_index(drop=True)

    # add reduced anchor indices to the matches
    pos_matches = all_data[["anchor", "match"]]\
        .reset_index(drop=True).merge(anchor_distinct, on='anchor')[['match', 'anchor_index']]
    pos_matches = pos_matches.drop_duplicates()

    print("Number of unique anchors: ", len(all_data["anchor"].unique()))
    print("Number of unique matches: ", len(all_data["match"].unique()))

    anchor_distinct.to_csv(raw_anchors_dir, mode='w+', index=False)
    pos_matches.to_csv(raw_matches_dir, mode='w+', index=False)

    # raw_anchor_data = pd.read_csv(raw_anchors_dir)
    # print(raw_anchor_data[raw_anchor_data['anchor_index'] == 44199])
    # raw_match_data = pd.read_csv(raw_matches_dir)
    # print(raw_match_data[raw_match_data['anchor_index'] == 44199])

    print('Embed docs')
    anchor_embed = embed_data(raw_anchors_dir, "anchor", emb_dir, max_len=max_len)
    match_embed = embed_data(raw_matches_dir, "match", emb_dir, max_len=max_len)

    anchor_embed.to_csv(out_anchor, mode='w+', index=False)
    match_embed.to_csv(out_match, mode='w+', index=False)

    print((time.time() - start) / 60.0)









