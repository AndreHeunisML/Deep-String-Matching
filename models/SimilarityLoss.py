#

import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConflateLoss(nn.Module):
    def __init__(self):
        super(ConflateLoss, self).__init__()

    def forward(self, anchor_embedding, match_embedding):
        """


        :param anchor:
        :param pos:
        :param neg:
        :return:
        """
        n = 50
        gamma = 10
        prob_correct_candidate_full = tc.zeros(anchor_embedding.size()[0])
        batch_sample_count = match_embedding.size()[0]

        for i, v in enumerate(anchor_embedding):
            similarities = tc.zeros(n, match_embedding.size()[1])
            similarities[0] = match_embedding[i]

            # sample n negative samples
            for j in range(1, n):
                while True:
                    index = np.random.randint(0, batch_sample_count-1)
                    similarities[j] = match_embedding[index]

                    # Ensure pos and neg match are different
                    # Ensure the anchor for the pos and neg are different
                    if not tc.all(tc.eq(match_embedding[index], match_embedding[i])) and \
                            not tc.all(tc.eq(anchor_embedding[i], anchor_embedding[index])):
                        # print the tr
                        break

            similarities = F.cosine_similarity(anchor_embedding[i].view(1, -1), similarities)
            prob_correct_candidate = tc.exp(gamma * similarities[0]) / (tc.exp(gamma * similarities)).sum()

            prob_correct_candidate_full[i] = prob_correct_candidate

        loss = 0
        p, _ = prob_correct_candidate_full.sort(dim=0, descending=False)
        for i in prob_correct_candidate_full:
            loss = loss + tc.log10(i)

        loss = -1 * loss

        return loss