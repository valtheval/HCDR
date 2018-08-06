# coding: utf-8
from cleaner import Cleaner
from loader import Loader
from flattener import Flattener
from time import time
from preprocessor import Preprocessor
from splitter import Splitter
from model_manager import ModelManager
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import numpy as np

if __name__ == '__main__':
    start_time = time()

    conf_path = "/Users/valentinmasdeu/Documents/8 - Projets/HCDR/configuration/configuration.json"

    print("Loading")
    loader = Loader(conf_path=conf_path)
    df = loader.load_csv("application_train", **{'nrows':100})
    prev_app = loader.load_csv("previous_applications", **{'nrows':100})
    bureau = loader.load_csv("bureau", **{'nrows':100})
    bureau_balance = loader.load_csv("bureau_balance", **{'nrows':100})
    pos = loader.load_csv("POS_CASH_balance", **{'nrows':100})
    instal_pmt = loader.load_csv("installments_payments", **{'nrows':100})
    cc_balance = loader.load_csv("credit_card_balance", **{'nrows':100})
    print(df.shape)

    print("Cleaning 1")
    cleaner = Cleaner()
    df = cleaner.format_float(df)
    df = cleaner.format_string(df)
    print(df.shape)

    print("Flattening and joining")
    flattener = Flattener()
    print("\tbureau_balance")
    map_bureau_balance = flattener.map(bureau_balance, "SK_ID_BUREAU")
    join_bureau_bureau_balance = flattener.join(bureau, map_bureau_balance, on="SK_ID_BUREAU", how='left', drop_existing_column='right')
    print("\tbureau")
    map_bureau = flattener.map(join_bureau_bureau_balance, "SK_ID_CURR")
    print("\tpos")
    map_pos = flattener.map(pos, "SK_ID_PREV")
    print("\tinstallments_payments")
    map_instal_pmt = flattener.map(instal_pmt, "SK_ID_PREV")
    print("\tcredit_card_balance")
    map_cc_balance = flattener.map(cc_balance, "SK_ID_PREV")
    print("\tprevious_applications")
    prev_app = flattener.join(prev_app, map_pos, on="SK_ID_PREV", how="left", drop_existing_column='right')
    prev_app = flattener.join(prev_app, map_instal_pmt, on="SK_ID_PREV", how="left", drop_existing_column='right')
    prev_app = flattener.join(prev_app, map_cc_balance, on="SK_ID_PREV", how="left", drop_existing_column='right')
    map_prev_app = flattener.map(prev_app, "SK_ID_CURR")
    print("\tmain table")
    df = flattener.join(df, map_prev_app, on="SK_ID_CURR", how="left")
    df = flattener.join(df, map_bureau, on="SK_ID_CURR", how="left")
    print(df.shape)

    print("Cleaning 2")
    df = cleaner.remove_empty_columns(df, 0.9)
    df = cleaner.format_float(df)
    df = cleaner.format_string(df)
    print(df.shape)

    print("Split")
    splitter = Splitter()
    X_train, X_test, y_train, y_test = splitter.train_test_split(df, 'TARGET', test_size=0.20, **{'random_state':63})
    print(X_train.shape, X_test.shape)

    print("Preprocessing")
    preprocessor = Preprocessor()
    X_train, X_test = preprocessor.imputation(X_train, X_test)
    X_train, X_test = preprocessor.standard_scaling(X_train, X_test)
    X_train, X_test = preprocessor.encoding(X_train, X_test)
    print(X_train.shape, X_test.shape)
    

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




