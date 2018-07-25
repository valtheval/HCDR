# coding: utf-8
import re
from data_explorer import DataExplorer

class Cleaner():

    def __init__(self):
        pass

    ####################################################################################################################
    # Columns removing
    ####################################################################################################################
    def remove_empty_columns(self, df, threshold):
        data_explorer = DataExplorer()
        empty_col = data_explorer.get_empty_columns(df, threshold)
        return df.drop(empty_col, axis=1)

    def remove_columns(self, df, columns):
        return df.drop(columns, axis=1)

    #TODO enlever colonne quasi constante (1 modalite >x%)

    ####################################################################################################################
    # String formatting
    ####################################################################################################################
    def format_string(self, df, columns=None, spec_char={}, nan_replacement=None):
        if columns is None:
            string_col = [col for col in df.columns if ((df[col].dtype == 'object') or
                                                        (df[col].dtype.name == 'category'))]
        else:
            string_col = columns
        df[string_col] = df[string_col].applymap(lambda x: self._format_string(x, spec_char, nan_replacement))
        return df

    def _format_string(self, x, spec_char={}, nan_replacement=None):
        if isinstance(x, str):
            x = x.lower()
            x = x.strip()
            x = ' '.join(x.split())
            for char in spec_char:
                x = x.replace(char, spec_char[char])
            #x = self._strip_non_alphanum(x)
            return x
        elif x is None:
            return nan_replacement
        else:
            return x

    def _strip_non_alphanum(self, text):
        return ' '.join(re.compile(r'\W+', re.UNICODE).split(text))

    ####################################################################################################################
    # Float formatting
    ####################################################################################################################
    def format_float(self, df, scale=None, columns=None):
        if columns is None:
            num_col = [col for col in df.columns if ((df[col].dtype == 'float') or (df[col].dtype == 'int'))]
        else:
            num_col = columns
        df[num_col] = df[num_col].applymap(lambda x: self._format_float(x, scale))
        return df

    def _format_float(self, x, scale=None):
        if isinstance(x, float) or isinstance(x, int):
            x = float(x)
            if scale is None:
                return x
            else:
                return round(x, scale)
        elif isinstance(x, str):
            x = ''.join(x.split())
            try:
                x_new = x.replace(',', '.') #12,3
                return float(x_new)
            except ValueError:
                try:
                    x_new = x.replace(',', '')# thousands separator 1,230.45
                    return float(x_new)
                except ValueError:
                    return x
        else:
            return x

    ####################################################################################################################
    # Date formatting
    ####################################################################################################################
    #TODO
    def format_date(self, df, columns, date_format):
        pass





