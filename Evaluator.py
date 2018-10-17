

import json
import torch.nn.functional as F
import torch as tc

class Evaluator:

    def __init__(self):
        char_to_index_fpath = 'dataprep/data/char_embeddings/char_to_index_28.json'
        with open(char_to_index_fpath) as _:
            self.char_to_index = json.load(_)

        self.index_to_char = dict((y, x) for x, y in self.char_to_index.items())


    def get_string(self, em):

        return ''.join([self.index_to_char[i] for i in em.numpy() if i != 0])

    def print_similar(self, batch, embeddings):
        """
        Find top 3 most similar words for each anchor

        :param batch:
        :param embeddings:
        :return:
        """
        print(batch.keys())
        print(embeddings.keys())

        print(type(batch['anchor']))

        for i, v in enumerate(batch['anchor']):
            print("-------------------------------------")
            print(self.get_string(v))

            d1 = embeddings['anchor_embedding'][i].view(1, -1)

            all_dist = tc.zeros(len(batch['posmatch']))
            for j, w in enumerate(batch['posmatch']):
                d2 = embeddings['posmatch_embedding'][j].view(1, -1)
                dd = F.pairwise_distance(d1, d2)
                all_dist[j] = dd

            all_dist_sorted, idx = all_dist.sort(dim=0, descending=False)
            print(all_dist_sorted[:3])
            #print(all_dist_sorted)
            posmatch_sorted = batch['posmatch'][idx]

            for k in range(3):
                print(self.get_string(posmatch_sorted[k]))