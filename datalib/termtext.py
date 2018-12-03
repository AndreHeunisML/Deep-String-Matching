

import pandas as pd
from pylab import *


def get_all_termtext():
    fpath = 'data/termtext/raw.csv'

    tt_dat = pd.read_csv(fpath, sep=',', header=None)
    tt_dat.columns = ['match', 'anchor']

    print(tt_dat.head())

    shuffle_indices = np.random.permutation(np.arange(len(tt_dat)))
    x1 = np.array(tt_dat['anchor'].values)[shuffle_indices]
    x2 = np.array(tt_dat['match'].values)[shuffle_indices]

    return pd.DataFrame(data={'anchor': x1, 'match': x2})


if __name__ == "__main__":
    get_all_termtext()
