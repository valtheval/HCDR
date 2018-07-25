import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin


class CategoricalImputer():

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.fill = {c:X[c].value_counts().index[0]
                     for c in X if (X[c].dtype == np.dtype('O') or X[c].dtype.name == "category")}
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)



class ImputerProxy(TransformerMixin):

    def __init__(self, float_strategy="mean"):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.
        """
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if (X[c].dtype == np.dtype('O') or X[c].dtype.name == "category")
                               else X[c].mean() for c in X],
                              index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

class CategoricalEncoderProxy():

    def __init__(self):
        self.encoders = {}

    def _fit_column(self, column, col):
        column_unique_values = list(set(column))
        dict_values = {}
        for i, value in enumerate(column_unique_values):
            dict_values[value] = i
        self.encoders[col] = dict_values

    def _transform_column(self, column, col):
        new_column = []
        used_values = list(self.encoders[col].values())
        max_val = max(used_values)
        for e in column.values:
            try:
                new_column.append(self.encoders[col][e])
            except KeyError:
                # New modality
                new_column.append(max_val + 1)
        return new_column

    def fit(self, X):
        self.encoders = {}
        for col in X:
            column = X[col]
            self._fit_column(column, col)

    def transform(self, X):
        list_new_column = []
        for col in X:
            column = X[col]
            X[col] = self._transform_column(column, col)
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)