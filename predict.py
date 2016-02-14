#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
import xgboost as xgb

import util
import util_logging
from dataset import Dataset


logger = util_logging.get_logger(__name__)

def cross_validation(X_train, y_train, params, X_test=None, verbose_eval=False):
    NUM_BOOST_ROUND = 1000
    best_iterations = []
    train_scores = []
    valid_scores = []
    y_preds = []

    kf = KFold(y_train.shape[0], n_folds=5, shuffle=True, random_state=12345)

    for train_index, valid_index in kf:
        _X_train, _X_valid = X_train.ix[train_index], X_train.ix[valid_index]
        _y_train, _y_valid = y_train[train_index], y_train[valid_index]

        dtrain = xgb.DMatrix(_X_train, _y_train)
        dvalid = xgb.DMatrix(_X_valid, _y_valid)

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        bst = xgb.train(params, dtrain, NUM_BOOST_ROUND, evals=watchlist,
                        early_stopping_rounds=200, verbose_eval=verbose_eval)

        # best iterations and valid score
        best_iterations.append(bst.best_iteration + 1)
        valid_scores.append(bst.best_score)

        if X_test is not None:
            dtest = xgb.DMatrix(X_test)
            y_pred = bst.predict(dtest, ntree_limit=bst.best_iteration)
            y_preds.append(y_pred)

    y_pred = util.sigmoid(np.mean(util.logit(np.array(y_preds)), axis=0))

    result = {"best-iterations": best_iterations,
              "best-iteration": np.mean(best_iterations),
              "valid-score": np.mean(valid_scores),
              "valid-scores": valid_scores,
              "y_pred": y_pred,
              "y_preds": y_preds}
    return result

def show_results(result, params):
    logger.info('eta: {eta} max_depth: {max_depth}'.format(**params))
    logger.info('subsample: {subsample} colsample_bytree: {colsample_bytree}'.format(**params))
    logger.info('valid-score: {valid-score} valid-scores: {valid-scores}'.format(**result))
    logger.info('best-iteration: {best-iteration} best-iterations: {best-iterations}'.format(**result))

def create_submission(y_pred, dataset, filename=None):
    ids = []
    countries = []
    for i, idx in enumerate(dataset.test['id']):
        ids += [idx] * 5
        countries += dataset.le_target.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

    result = pd.DataFrame(np.column_stack([ids, countries]), columns=['id', 'country'])
    if filename is None:
        result.to_csv('./submissions/submission.csv', index=False)
    else:
        result.to_csv('./submissions/{}'.format(filename), index=False)
    return result

def predict(dataset):
    logger.info('Start Model {}...'.format(dataset.name))
    X_train = dataset.train[dataset.features]
    y_train = dataset.train[dataset.target]
    X_test = dataset.test[dataset.features]

    params = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 12,

        # Control complexity of model
        'eta': 0.05,
        'max_depth': 5,
        'min_child_weight': 1,

        # Improve noise robustness
        'subsample': 0.60,
        'colsample_bytree': 0.60,

        'seed': 12345,
        'silent': 1,
    }

    result = cross_validation(X_train, y_train, params, X_test=X_test)

    logger.info('Result')
    show_results(result, params)

    logger.info('Create Submission...')
    create_submission(result['y_pred'], dataset,
                      filename='{}.csv'.format(dataset.name))

    logger.info('Finished')


if __name__ == '__main__':
    dataset = Dataset(name='b88369b').load_pkl()
    predict(dataset)
