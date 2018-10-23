# Format positive examples for name data

import pandas as pd
from random import random


def import_data(filepath):
    x1 = []
    x2 = []

    # positive samples from file
    for line in open(filepath):
        l = line.strip().split("\t")
        if len(l) < 2:
            continue

        if l[0].lower() == l[1].lower():
            continue

        if random() > 0.5:
            x1.append(l[0].lower())
            x2.append(l[1].lower())
        else:
            x1.append(l[1].lower())
            x2.append(l[0].lower())

    return pd.DataFrame(data={'anchor': x1, 'pos_match': x2})


def get_all_names():
    data1 = import_data(
        "data/names/person_match.train")

    data2 = import_data(
        "data/names/person_match.train2")

    return pd.concat([data1, data2])


if __name__ == "__main__":
    result = get_all_names()
    print(len(result))
    print(result)

