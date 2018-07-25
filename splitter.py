from sklearn.model_selection import train_test_split


class Splitter():

    def __init__(self):
        pass

    def train_test_split(self, df, target, test_size, **kwargs):
        y = df[target]
        tmp = df.drop(target, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(tmp, y, test_size=test_size, **kwargs)
        return X_train, X_test, y_train, y_test
