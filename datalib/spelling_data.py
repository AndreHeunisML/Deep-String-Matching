# Format positive examples for spelling data

import pandas as pd


def get_txt_spell(filepath):
    anchors = []
    positives = []

    # positive samples from file
    for line in open(filepath):

        fl = line.strip().split(":")
        anchor = fl[0].lower()
        all_positives = fl[1].strip().split(" ")
        all_positives = [p.lower() for p in all_positives]

        anchors.extend([anchor] * len(all_positives))
        positives.extend(all_positives)

    res = pd.DataFrame(data={'anchor': anchors, 'match': positives})
    res = res[res.anchor != res.match]

    return res


def get_csv_spell(filepath):

    d = pd.read_csv(filepath)
    d.columns = ["anchor", "match"]

    return d


def get_all_spell():
    d1 = get_txt_spell("data/spelling/aspell.txt")
    d2 = get_txt_spell("data/spelling/spell-testset1.txt")
    d3 = get_txt_spell("data/spelling/spell-testset2.txt")
    d4 = get_txt_spell("data/spelling/wikipedia.txt")

    d5 = get_csv_spell("data/spelling/spelling_variants_valid.csv")

    return pd.concat([d1, d2, d3, d4, d5])


if __name__ == "__main__":
    result = get_all_spell()

    print(result.head(20))
    print(len(result))

