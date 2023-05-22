import os
import argparse

import numpy as np
import pandas as pd

from model import *
from utils import cross_val, save_result

NUM_VAL = 10
ENSEMBLE_SIZE = [1, 5, 10, 100]
KWARGS = {
    'DecisionStump': {}, 
    'LogisticRegression': {'lr': 0.00003}
}


def build_base_model(model_name, **kwargs):
    return eval(model_name)(**kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Adaboost model!!!",)
    parser.add_argument('--base-model', default='DecisionStump')
    parser.add_argument('-o', '--output', default='experiments')
    parser.add_argument('-x', '--input-x', default='data.csv')
    parser.add_argument('-y', '--input-y', default='targets.csv')
    args = parser.parse_args()

    assert args.base_model in ['DecisionStump', 'LogisticRegression'], \
        'for base model, you should choose decisionStump or logisticRegression'
    X = pd.read_csv(args.input_x, header=None)
    std = X.std()
    std[std < 1e-7] = 1e-7
    X = (X - X.mean()) / std
    Y = pd.read_csv(args.input_y, header=None)
    os.makedirs(args.output, exist_ok=True)
    base_model = lambda: build_base_model(
        args.base_model, **KWARGS[args.base_model])

    for i in ENSEMBLE_SIZE:
        save = lambda fold, data, index: save_result(
            args.output, i, fold, data, index)
        model = AdaBoost(base_model, i)
        print(f'Ensemble size: {i}')
        cross_val(X, Y, model, NUM_VAL, save)
