from copy import deepcopy

import numpy as np

class DecisionStump:
    def __init__(self):
        self.value = None
        self.index = None
        self.upper_value = None

    def fit(self, x, y, obs_w):
        split = np.zeros((x.shape[-1], 2))
        y = y[:, 0]
        for i, x_ in enumerate(x.T):
            threshold, entropy = self.compute_threshold(x_, y, obs_w)
            split[i][0] = threshold
            split[i][1] = entropy
        split_index = split[:, 1].argmin()
        self.index = split_index
        self.value = split[split_index][0]
        self.upper_value = y[x[:, self.index] > self.value].mean() > 0.5
        return

    def predict(self, x):
        pred = x[:, self.index] > self.value
        return pred ^ (not self.upper_value)

    @staticmethod
    def compute_threshold(x, y, obs_w):
        # note: the feature and threshold that return the max entropy 
        # hold the max precise. Even when you are fitting a decision stump
        unique = np.unique(x)
        entropy = []
        for i in unique:
            p = (x > i).mean()
            e1 = DecisionStump.entropy(y[x>i], obs_w[x>i]) * p
            e2 = DecisionStump.entropy(y[x<=i], obs_w[x<=i]) * (1-p)
            entropy.append(e1+e2)
        index = np.argmin(entropy)
        return unique[index], entropy[index]

    @staticmethod
    def entropy(C, obs_w):
        assert C.shape == obs_w.shape
        obs_w = obs_w / obs_w.sum()
        classes = np.unique(C)
        prob = np.array([(obs_w[C == i]).sum() for i in classes])
        log_prob = np.log(prob)
        return (-prob * log_prob).sum()

