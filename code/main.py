import os
from copy import deepcopy

import numpy as np
import pandas as pd

from model import *
from utils import cross_val, save_result

class Adaboost:
    NUM_VAL = 10
    ENSEMBLE_SIZE = [1, 5, 10, 100]
    KWARGS = {
        'DecisionStump': {}, 
        'LogisticRegression': {'lr': 0.00003}
    }
    MODELMAP = ['LogisticRegression', 'DecisionStump']
    def __init__(self, base):
        '''
        :param base: 基分类器编号 0 代表对数几率回归 1 代表决策树桩
        在此函数中对模型相关参数进行初始化，如后续还需要使用参数请赋初始值
        '''
        self.base = self.MODELMAP[base]
        # save the best model
        self.best_model = None
        self.best_score = -np.inf

    def fit(self, x_file, y_file):
        '''
        在此函数中训练模型
        :param x_file:训练数据(data.csv)
        :param y_file:训练数据标记(targets.csv)
        '''
        X = pd.read_csv(x_file, header=None).values
        self.mean = X.mean(0)
        std = X.std(0)
        std[std < 1e-7] = 1e-7
        X = (X - X.mean(0)) / std
        Y = pd.read_csv(y_file, header=None)
        os.makedirs('experiments', exist_ok=True)
        self.std = std

        self.base_model = lambda: self.build_base_model(
            self.base, **self.KWARGS[self.base])

        for i in self.ENSEMBLE_SIZE:
            save = lambda fold, data, index: save_result(
                'experiments', i, fold, data, index)
            # origin model written before
            model = SAdaBoost(self.base_model, i)
            print(f'Ensemble size: {i}')
            model, score, data = cross_val(X, Y, model, self.NUM_VAL, save)
            if score > self.best_score:
                self.best_model = deepcopy(model)
                self.best_score = score
                self.test_data = data
        print('[*] Finished training, best score: {:.2f}'.format(self.best_score))

    def predict(self, x_file):
        '''
        :param x_file:测试集文件夹(后缀为csv)
        :return: 训练模型对测试集的预测标记
        '''
        if not self.best_model:
            import ipdb; ipdb.set_trace()
        X = pd.read_csv(x_file, header=None).values
        X = (X - self.mean) / self.std
        return self.best_model.predict(X)[0]

    @staticmethod
    def build_base_model(model_name, **kwargs):
        return eval(model_name)(**kwargs)
