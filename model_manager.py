


class ModelManager():


    def __init__(self):
        pass

    def benchmark(self, X_train, y_train, X_test=None, y_test=None, list_model=[], list_metrics=[], results={}):
        for model_dictionnary in list_model:
            model_name = model_dictionnary['name']
            model = model_dictionnary["model"]
            #Training
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            train_scores = {}
            test_scores = {}
            for metric in list_metrics:
                score_train = metric(y_train, y_train_pred)
                train_scores[metric.__name__] = score_train
                if X_test is not None:
                    y_test_pred = model.predict(X_test)
                    score_test = metric(y_test, y_test_pred)
                    test_scores[metric.__name__] = score_test
            results[model_name] = {'Train': train_scores,
                                   'Test': test_scores,
                                   'model': model}
        return results