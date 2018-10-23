

import pandas as pd
import numpy as np
import seaborn as sns
from pylab import *


def get_all_termtext():
    fpath = 'data/termtext/staging.csv'

    tt_dat = pd.read_csv(fpath, sep=',', header=None)
    tt_dat.columns = ['pos_match', 'anchor']

    shuffle_indices = np.random.permutation(np.arange(len(tt_dat)))
    x1 = np.array(tt_dat['anchor'].values)[shuffle_indices]
    x2 = np.array(tt_dat['pos_match'].values)[shuffle_indices]

    return pd.DataFrame(data={'anchor': x1, 'pos_match': x2})


if __name__ == "__main__":
    get_all_termtext()
