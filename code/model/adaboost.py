import numpy as np

class SAdaBoost:
    '''adaboost.M1'''
    def __init__(
        self, 
        base_model, 
        ensemble_size=50, 
    ):
        self.base_model = base_model
        self.ensemble_size = ensemble_size
        self.model = []
        self.model_weight = np.zeros(ensemble_size)

    def reset(self):
        self.model = []
        self.model_weight = np.zeros(self.ensemble_size)

    def fit(self, x, y, ensemble_size=None):
        if ensemble_size != None:
            self.ensemble_size = ensemble_size
            self.model_weight = np.zeros(ensemble_size)
        self.obs_w = np.ones(x.shape[0]) / x.shape[0]
        for i in range(self.ensemble_size):
            model = self.base_model()
            prec = model.fit(x, y, self.obs_w)
            err_predict = model.predict(x) != y[:, 0]
            err = (err_predict * self.obs_w).sum() / self.obs_w.sum()
            if err > 0.5:
                self.model_weight = self.model_weight[:i]
                break
            # print(err)
            self.model_weight[i] = np.log(((1-err) / err))
            self.obs_w[err_predict] *= \
                np.exp(self.model_weight[i])
            # self.obs_w[np.logical_not(err_predict)] *= \
            #     np.exp(-self.model_weight[i])
            self.obs_w = self.obs_w / self.obs_w.sum()
            self.model.append(model)

    def predict(self, x):
        results = np.array([model.predict(x) for model in self.model])
        # changing the prediction form
        try:
            results_ = ((results * self.model_weight[:, None]) / self.model_weight.sum()).sum(0)
        except ValueError:
            import ipdb; ipdb.set_trace()
        return results_ > 0.5, results
