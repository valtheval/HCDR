#coding : utf-8
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, levene, ttest_ind


class DataExplorer():

    def __init__(self):
        pass


    def compute_basic_stat_one_var(self, serie, nb_cat_max):
        stat_desc = {}
        name = serie.name
        nb_elem = len(serie)
        nb_empty = serie.isnull().sum()
        p_empty = float(nb_empty)/nb_elem
        nb_cat = len([u for u in serie.unique() if not pd.isnull(u)])
        if nb_cat <= nb_cat_max :
            nb_par_cat = list(zip(list(serie.value_counts().index), list(serie.value_counts().values)))
        else:
            nb_par_cat = "too many categories"
        stat_desc[name] = {"type" : serie.dtype,
                           "nb_empty" : nb_empty,
                           "p_empty" : p_empty,
                           "nb_cat" : nb_cat,
                           "nb_per_cat" : nb_par_cat}

        if (serie.dtype=='float') | (serie.dtype=='int'):
            stat_desc[name]['mean'] = serie.mean()
            stat_desc[name]['std'] = serie.std()
            stat_desc[name]['minimum'] = serie.min()
            stat_desc[name]['maximum'] = serie.max()
        return stat_desc

    def compute_basic_stat(self, df, nb_cat_max, compute_corr=False):
        stat_desc = {}
        for c in df.columns:
            stat_desc[c] = self.compute_basic_stat_one_var(df[c], nb_cat_max)[c]

        df_stat_desc = pd.DataFrame(stat_desc).transpose()
        df_stat_desc = df_stat_desc[["type", 'nb_empty', 'p_empty', "nb_cat", "nb_per_cat", 'mean', 'std','minimum',
                                     'maximum']]
        if compute_corr:
            df_corr = df.corr()
            df_corr.values[[np.arange(len(df_corr))]*2] = 0
            df_corr = df_corr.fillna(0)
            df_corr['var_max_corr'] = df_corr.apply(lambda x: np.argmax(x), axis=1)
            df_corr['max_coef_corr'] = df_corr.apply(lambda x: np.max(x[:-1]), axis=1)
            df_stat_desc = pd.concat([df_stat_desc, df_corr[['var_max_corr', 'max_coef_corr']]], axis=1)
        return df_stat_desc

    def variables_relation(self, var1, var2):
        """

        :param var1: pandas series. To be considered as categorical it must be either of dtype string (object) or
        category
        :param var2: pandas series. To be considered as categorical it must be either of dtype string (object) or
        category
        :return: (s, p, name) stat, p-value and name of  statistical test used to measure the link between the 2
        variables
        """
        if (var1.dtype == "object") or (var1.dtype == "category"):
            if (var2.dtype == "object") or (var2.dtype == "category"):
                return self._chi2_test(var1,var2)
            else:
                cat = [u for u in var1.unique() if not pd.isnull(u)]
                nb_cat = len(cat)
                #TODO
                if nb_cat==2:#ttest
                    var21 = var2[(var1==cat[0])]
                    var22 = var2[(var1==cat[1])]
                    if (len(var22)>=30) & (len(var22)>=30):
                        #Pas de test de normalité
                        var_egal = (self._levene_test(var21, var22)[1]<0.05)
                        if var_egal:
                            #ttest
                            return self._ttest_ind(var21, var22, True)
                        else:
                            #Welch test
                            return self._ttest_ind(var21, var22, False)
                    else:
                        normal = True # TODO faire test de normalité (kolmogorov)
                        if normal:
                            var_egal = (self._levene_test(var21, var22)[1]<0.05)
                            if var_egal:
                                #ttest
                                return self._ttest_ind(var21, var22, True)
                            else:
                                #Welch test
                                return self._ttest_ind(var21, var22, False)
                        else:
                            pass
                elif nb_cat>2:#ANOVA
                    if len(var2)>=30:
                        #Pas de test de normalité
                        var_egal = True #TODO faire test d'égalité des variances
                        if var_egal:
                            #TODO test de student
                        else:
                            # Test de welch
                pass
        else:
            if (var2.dtype == "object") or (var2.dtype == "category"):
                #TODO
                pass
            else:
                #TODO
                pass

    def _chi2_test(self, var1, var2):
        contingency_table = pd.crosstab(var1, var2).values
        s, p, _, _ = chi2_contingency(contingency_table)
        return s, p, "chi2_contingency"

    def _levene_test(self, var1, var2):
        w, p = levene(var1.values, var2.values)
        return w, p, "levene_test (Brown–Forsythe)"

    def _ttest_ind(self, var1, var2, equal_var=True):
        s, p = ttest_ind(var1.values, var2.values, equal_var)
        return s, p, "ttest_ind_var_eq_is_%s"%(str(equal_var))







