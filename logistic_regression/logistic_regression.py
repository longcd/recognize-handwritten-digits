#coding: utf-8
#Author: longcd

'''
A simple Logistic Regression model.
'''
import sys
import numpy as np

class LogisticRegression(object):

    def __init__(self):
        pass


    def fit(self, X, y, alpha, batch_num=20, lmbda=0, regularization=None, verbose=False):
        N, F = X.shape
        self.__features_num = F
        self.coef_ = np.zeros(F)
        self.intercept_ = 0

        last_min_error = float("inf")
        last_step = 0
        costs = list()

        for step in range(0, 100):
            for s in range(0, N, batch_num):
                pred = 1. / (1 + np.exp(-X[s : s + batch_num].dot(self.coef_) - self.intercept_))
                    
                coef_grad = X[s : s + batch_num].T.dot(pred - y[s : s + batch_num]) / batch_num
                intercept_grad = sum(pred - y[s : s + batch_num]) / batch_num

                if regularization == 'l1':
                    self.coef_ = self.coef_ - alpha * coef_grad - alpha * lmbda * np.sign(self.coef_)
                elif regularization == 'l2':
                    self.coef_  = (1 - alpha * lmbda) * self.coef_ - alpha * coef_grad
                else:
                    self.coef_ = self.coef_ - alpha * coef_grad
                self.intercept_ = self.intercept_ - alpha * intercept_grad

            pred = 1. / (1 + np.exp(-X.dot(self.coef_) - self.intercept_))
            error = -sum(y * np.log(pred) + (1 - y) * np.log(1 - pred)) / N

            costs.append(error)
            if last_min_error - error > 1e-6:
                last_min_error = error
                last_step = step
            elif step - last_step >= 10:
                break

            if verbose is True and step % 10 == 0:
                print("step %s: %.6f" % (step, error))

        if verbose is True:
            pred = 1. / (1 + np.exp(-X.dot(self.coef_) - self.intercept_))
            error = -sum(y * np.log(pred) + (1 - y) * np.log(1 - pred)) / N
            print("Final training error: %.6f\n" % (error))

        return costs


    def predict(self, X):
        if X.shape[1] != self.__features_num:
            print("The data to be evaluated can't match training data's features\n")
            return None
        return 1. / (1 + np.exp(-X.dot(self.coef_) - self.intercept_))