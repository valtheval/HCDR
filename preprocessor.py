from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, Imputer
from sklearn_proxy import CategoricalImputer
import pandas as pd

class Preprocessor():

    def __init__(self):
        self.jobs_done = []

    def imputation(self, X_train, X_test, missing_values="NaN", strategy="mean", axis=0, verbose=0, copy=True):
        X_train, X_test = self._float_imputation(X_train, X_test, missing_values, strategy, axis, verbose, copy)
        X_train, X_test = self._categorical_imputation(X_train, X_test)
        return X_train, X_test

    def _float_imputation(self, X_train, X_test, missing_values="NaN", strategy="mean", axis=0, verbose=0, copy=True):
        numerical_columns = [c for c in X_train.columns
                             if ((X_train[c].dtypes == 'float') or (X_train[c].dtypes == 'int'))]
        imputer = Imputer(missing_values, strategy, axis, verbose, copy)
        X_train[numerical_columns] = imputer.fit_transform(X_train[numerical_columns])
        X_test[numerical_columns] = imputer.transform(X_test[numerical_columns])
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
        X_test = X_test[[c for c in X_test.columns if c in X_train.columns]]
        # TODO reflechir voir si il faut garder l'intersection des 2 ou juste les colonnes de X_train au plus
        return X_train, X_test






