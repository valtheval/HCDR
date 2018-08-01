# coding: utf-8
import pandas as pd
import json


class Loader():

    def __init__(self, conf_path):
        self.conf = self._load_configuration(conf_path)


    def load_csv(self, file, **kwargs):
        try:
            return pd.read_csv(file, **kwargs)
        except (IOError, OSError) as e:
            return pd.read_csv(self.conf['files'][file], **kwargs)

    def _load_configuration(self, path):
        with open(path, encoding='utf-8') as f:
            conf = json.load(f)
        return conf
