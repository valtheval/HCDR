from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, Imputer
from sklearn_proxy import CategoricalImputer
import pandas as pd
import numpy as np

class Preprocessor():

    def __init__(self):
        self.jobs_done = []

    def imputation(self, X_train, X_test, missing_values="NaN", strategy="mean", axis=0, verbose=0, copy=True):
        X_train, X_test = self._float_imputation(X_train, X_test, missing_values, strategy, axis, verbose, copy)
        X_train, X_test = self._categorical_imputation(X_train, X_test)
        return X_train, X_test

    def _float_imputation(self, X_train, X_test, missing_values="NaN", strategy="mean", axis=0, verbose=0, copy=True):
        numerical_features = list(X_train.select_dtypes(include = ["float", "int"]).columns)

        imputer = Imputer(missing_values, strategy, axis, verbose, copy)
        X_train[numerical_features] = imputer.fit_transform(X_train[numerical_features])
        X_test[numerical_features] = imputer.transform(X_test[numerical_features])
        self.jobs_done.append({"float_imputation":imputer})
        return X_train, X_test

    def _categorical_imputation(self, X_train, X_test):
        """
        Fillna with most frequent values for object columns
        :param X_train:
        :param X_test:
        :return:
        """
        cat_columns = [c for c in X_train.columns
                       if ((X_train[c].dtypes == 'object') or (X_train[c].dtypes.name == 'category'))]
        imputer = CategoricalImputer()
        X_train[cat_columns] = imputer.fit_transform(X_train[cat_columns])
        X_test[cat_columns] = imputer.transform(X_test[cat_columns])
        self.jobs_done.append({"categorical_imputation":imputer})
        return X_train, X_test

    def encoding(self, X_train, X_test, **kwargs):
        X_train, X_test = pd.get_dummies(X_train, **kwargs), pd.get_dummies(X_test, **kwargs)
        col_in_train_not_in_test = [c for c in X_train.columns if (c not in X_test.columns)]
        n_test = len(X_test)
        for missing_col in col_in_train_not_in_test:
            X_test[missing_col] = pd.Series(np.zeros(n_test))
        X_test = X_test[X_train.columns]
        return X_train, X_test

    def standard_scaling(self, X_train, X_test, copy=True, with_mean=True, with_std=True):
        scaler = StandardScaler(copy, with_mean, with_std)
        numerical_features = list(X_train.select_dtypes(include = ["float", "int"]).columns)
        X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
        X_test[numerical_features] = scaler.transform(X_test[numerical_features])
        self.jobs_done.append({"standard_scaling":scaler})
        return X_train, X_test










