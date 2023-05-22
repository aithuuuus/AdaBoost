import os
from copy import deepcopy

import numpy as np
import pandas as pd

def save_result(output_dir, base, fold, data, index):
    path = os.path.join(output_dir, f'base{base}_fold{fold}.csv')
    index = np.array(index)
    # for the code in evaluate.py (provided by prof. Zhang), I increase the index by one :)
    data = np.stack((index+1, data.astype(int)), -1)
    np.savetxt(path, data, delimiter=",", fmt='%i')

def cross_val(X, Y, model, num, save):
    best_model = None
    best_score = -np.inf
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X = X.values
    if isinstance(Y, pd.DataFrame) or isinstance(Y, pd.Series):
        Y = Y.values
    # !!! shuffling
    random_index = np.arange(X.shape[0])
    np.random.shuffle(random_index) # ! return None, inplace shuffle
    X, Y = X[random_index], Y[random_index]

    gap = int(X.shape[0] / num)
    index = list(range(0, X.shape[0], gap))
    if len(index) < num+1:
        index.append(X.shape[0])
    if len(index) < num+1:
        import ipdb; ipdb.set_trace()

    count = 1
    results = []
    for i, j in zip(index[:-1], index[1:]):
        model.reset()
        val_X = X[i: j]
        train_X = np.concatenate((X[:i], X[j:]), 0)
        val_Y = Y[i: j]
        train_Y = np.concatenate((Y[:i], Y[j:]), 0)
        model.fit(train_X, train_Y)
        pred, _ = model.predict(val_X)
        precise = (pred == val_Y[:, 0]).sum() / pred.shape[0]
        results.append(precise)
        print('\t[*] round {}, precision rate: {:.2f}'.format(count, precise))
        save(count, pred, random_index[i: j])
        if precise > best_score:
            best_model = deepcopy(model)
            best_score = precise
        count += 1
    print('\t[*] mean: {:.2f}'.format(np.mean(results)))
    return best_model, best_score, None
