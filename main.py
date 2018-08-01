# coding: utf-8
from cleaner import Cleaner
from loader import Loader
from time import time
from preprocessor import Preprocessor
from splitter import Splitter
from model_manager import ModelManager
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

if __name__ == '__main__':
    start_time = time()

    conf_path = "/Users/valentinmasdeu/Documents/8 - Projets/HCDR/configuration/configuration.json"

    print("Loading")
    loader = Loader(conf_path=conf_path)
    df = loader.load_csv("application_train")

    print("Cleaning")
    cleaner = Cleaner()
    df = cleaner.format_float(df)
    df = cleaner.format_string(df)

    print("Split")
    splitter = Splitter()
    X_train, X_test, y_train, y_test = splitter.train_test_split(df, 'TARGET', test_size=0.20)

    print("Preprocessing")
    preprocessor = Preprocessor()
    X_train, X_test = preprocessor.imputation(X_train, X_test)
    X_train, X_test = preprocessor.encoding(X_train, X_test)

    print("Learning")
    model_manager = ModelManager()
    list_model = [{'model':LogisticRegression(),
                   "name":'LR'},
                  {'model':RandomForestClassifier(),
                   "name":"RF"},
                  {"model":GradientBoostingClassifier(),
                   "name":"GBC"},
                  {"model":GradientBoostingClassifier(n_estimators=1000),
                   "name":"gbcCV1000"},
                  {"model":DecisionTreeClassifier(),
                   "name":"DT"}]
    list_metric = [accuracy_score, f1_score, recall_score, precision_score]
    results = model_manager.benchmark(X_train, y_train, X_test, y_test, list_model=list_model[0:1],
                                      list_metrics=list_metric)
    print("Results :")
    print(results)


    end_time = time()
    min, sec = int((end_time - start_time)/60), ((end_time - start_time) % 60)
    print("Completed in %d min %d second"%(min, sec))




