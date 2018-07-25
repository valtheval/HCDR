# coding: utf-8
from loader import Loader
from cleaner import Cleaner

if __name__ == '__main__':

    path_conf = "configuration/configuration.json"
    ld = Loader(path_conf)

    app_train = ld.load_csv("application_train")

    print(app_train.shape)
    print(app_train.columns)


