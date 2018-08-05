# coding: utf-8
import pandas as pd
import scipy

class Joiner():

    def __init__(self):
        pass


    def join(self, right, left, on, how='inner', **kwargs):
        return pd.merge(right, left, on=on, how=how, **kwargs)

    def map(self, df, agg_key, numerical_features_agg='mean', categorical_features_agg='most_frequent'):
        categorical_features = list(df.select_dtypes(include = ["object", 'category']).columns)
        numerical_features = list(df.select_dtypes(include = ["float", "int"]).columns)

        if not isinstance(agg_key, list):
            agg_key = [agg_key]

        for key in agg_key:
            if key not in categorical_features:
                categorical_features = [key] + categorical_features
            if key not in numerical_features:
                numerical_features = [key] + numerical_features

        if len(numerical_features)>len(agg_key):
            df_num = df[numerical_features].copy()
            print("groupby num features")
            df_num_grp = df_num.groupby(agg_key).agg(numerical_features_agg).reset_index()
        else:
            df_num_grp = None
        if len(categorical_features)>len(agg_key):
            df_cat = df[categorical_features].copy()
            print("groupby cat features")
            if categorical_features_agg == 'most_frequent':
                df_cat_grp = self._most_frequent(df_cat, agg_key)
        else:
            df_cat_grp = None
        #df_cat_grp = df_cat.groupby(agg_key).agg(self._most_frequent).reset_index()
        if df_num_grp is not None:
            if df_cat_grp is not None:
                print("merging")
                df_merge = pd.merge(df_num_grp, df_cat_grp, how='outer', on=agg_key)
            else:
                return df_num_grp
        else:
            return df_cat_grp
        return df_merge

    def _mode(self, df, key_cols, value_col, count_col):
        """
        Pandas does not provide a `mode` aggregation function
        for its `GroupBy` objects. This function is meant to fill
        that gap, though the semantics are not exactly the same.

        The input is a DataFrame with the columns `key_cols`
        that you would like to group on, and the column
        `value_col` for which you would like to obtain the mode.

        The output is a DataFrame with a record per group that has at least one mode
        (null values are not counted). The `key_cols` are included as columns, `value_col`
        contains a mode (ties are broken arbitrarily and deterministically) for each
        group, and `count_col` indicates how many times each mode appeared in its group.
        """
        return df.groupby(key_cols + [value_col]).size() \
                 .to_frame(count_col).reset_index() \
                 .sort_values(count_col, ascending=False) \
                 .drop_duplicates(subset=key_cols)\
                 .drop('count', axis=1)

    def _most_frequent(self, df_cat, agg_key):
        categorical_features = df_cat.columns
        agg_cat = []
        for col_cat in categorical_features:
            if col_cat not in agg_key:
                tmp_agg_cat = self._mode(df_cat, agg_key, col_cat, 'count')
                agg_cat.append(tmp_agg_cat)
        df_cat_grp = agg_cat[0]
        for df_cat_tmp in agg_cat[1::]:
            df_cat_grp = pd.merge(df_cat_grp, df_cat_tmp, how='outer', on=agg_key)
        return df_cat_grp