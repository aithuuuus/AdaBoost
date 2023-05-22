import numpy as np

class LogisticRegression:
    def __init__(
        self, lr, 
    ):
        self.lr = lr
        self.w = None

    @staticmethod
    def sigmoid(x):
        return 1 / (1+np.exp(-x))

    def fit(self, x, y, obs_w, n_iters=100):
        w = np.zeros(x.shape[1]+1)
        x = np.concatenate((x, np.ones((x.shape[0], 1))), -1)
        y = y[:, 0]
        old_err = 1.1
        for i in range(n_iters):
            predict = self.sigmoid((x * w).sum(-1)) > 0.5
            err = ((predict != y) * obs_w).sum()
            if err > old_err:
                w += delta * obs_w.shape[0]
                break
            delta = self.lr * (((predict - y))[:, None] * x * obs_w[:, None]).mean(0)
            w -= delta * obs_w.shape[0]
            old_err = err
        self.w = w
        return err

    def predict(self, x):
        x = np.concatenate((x, np.ones((x.shape[0], 1))), -1)
        return self.sigmoid((x * self.w).sum(-1)) > 0.5
