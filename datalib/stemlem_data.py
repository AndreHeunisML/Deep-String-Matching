# Format positive examples for stem and lemma data

import json
import pandas as pd


def get_json(fpath):

    with open(fpath) as _:
        jdat = json.load(_)

    anchor = []
    pos = []

    # Use abbreviations as anchor
    for key, val in jdat.items():
        anchor.extend(val)
        pos.extend([key] * len(val))

    res = pd.DataFrame(data={'anchor': anchor, 'pos_match': pos})
    res = res[res.anchor != res.pos_match]

    return res


def get_all_stemlem():
    d1 = get_json("data/stemlem/lem.json")
    d2 = get_json("data/stemlem/stem.json")

    return pd.concat([d1, d2])


if __name__ == "__main__":
    result = get_all_stemlem()

    print(result.head(100))
    print(len(result))