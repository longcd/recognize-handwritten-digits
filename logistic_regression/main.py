#coding: utf-8
#Author: longcd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression

PREDICT_NUMBERS = 10

if __name__ == "__main__":
    print("Loading training data")
    pd_train = pd.read_csv("../data/training", header=None)

    print("Loading testing data")
    pd_test = pd.read_csv("../data/testing", header=None)

    print("Loading validating data")
    pd_validate = pd.read_csv("../data/validation", header=None)

    train_X = pd_train.drop([0], axis=1).values / 255.0
    predictors = list()
    for number in range(0, PREDICT_NUMBERS):
        train_y = np.array(list(map(int, pd_train[0].values == number)))
        predictors.append(LogisticRegression())
        print("------------- Training model for number: %d -------------\n" % number)
        predictors[number].fit(train_X, train_y, 0.5, batch_num=50000, verbose=True)

    print("Evaluate model on validating data")
    validate_X = pd_validate.drop([0], axis=1).values / 255.
    tot_right_num = 0
    for tx, ty in zip(validate_X, pd_validate[0].values):
        pred_y = list()
        for number in range(0, PREDICT_NUMBERS):
            pred_y.append(predictors[number].predict(np.reshape(tx, (1, -1))))

        if np.argmax(pred_y) == ty:
            tot_right_num += 1

    print("\nOn validating data: %s/%s\n" % (tot_right_num, validate_X.shape[0]))