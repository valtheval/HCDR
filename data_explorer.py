# coding: utf-8
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, levene, ttest_ind, norm, ks_2samp, mannwhitneyu, f_oneway, pearsonr,\
    spearmanr, kruskal
from itertools import combinations

class DataExplorer():

    def __init__(self):
        pass

    def get_empty_columns(self, df, threshold):
        """
        Return the list of the columns of df that have more than threshold prportion of empty value
        :param df: pandas dataframe
        :param threshold: float between 0 and 1
        :return: string list, list of columns' name
        """
        empty_col = []
        nb_elem = len(df)
        for col in df.columns:
            nb_empty = df[col].isnull().sum()
            p_empty = float(nb_empty)/nb_elem
            if p_empty > threshold:
                empty_col.append(col)
        return empty_col

    def compute_basic_stat_one_var(self, serie, nb_cat_max=20):
        """
        Compute basic statistic for 1 pandas serie
        :param serie: pandas serie
        :param nb_cat_max: max number of different values to display. If the serie has less than this number we display
        all the category, otherwise none are displayed
        :return: Dictionnary of the form with type, nb and percentage of empty value, number of categories and number of
         element per category, mean, std, min, max for float var
        """
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

    def compute_basic_stat(self, df, nb_cat_max=20, compute_corr=False):
        """
        Compute basic statistic for all the variables of df
        :param df: pandas dataframe
        :param nb_cat_max: int, max number of different values to display. If the serie has less than this number we display
        all the category, otherwise none
        :param compute_corr: boolean, to compute the most correlated variable with the current variable
        :return:
        """
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

    def relation_independant_var(self, var1, var2, conditions_thresholds=0.05):
        """
        Assess relation between 2 variables of a dataframe
        :param var1: pandas series. To be considered as categorical it must be either of dtype string (object) or
        category
        :param var2: pandas series. To be considered as categorical it must be either of dtype string (object) or
        category
        :param conditions_thresholds: float, threshold for p-value used when checking conditions of test application
        (normality condition, variance equality...)
        :return: (s, p, name) stat, p-value and name of  statistical test used to measure the link between the 2
        variables
        """
        ct = conditions_thresholds
        if (var1.dtype == "object") or (var1.dtype.name == "category"):
            if (var2.dtype == "object") or (var2.dtype.name == "category"):
                # TODO si effectif < 5 il faut faire un test exact de fisher
                return self._chi2_test(var1,var2)
            else:
                cat = [u for u in var1.unique() if not pd.isnull(u)]
                nb_cat = len(cat)
                if nb_cat==2:#ttest (ou welch test)
                    var21 = var2[(var1==cat[0])]
                    var22 = var2[(var1==cat[1])]
                    if (len(var22)>=30) & (len(var22)>=30):
                        #No need of normality test
                        var_egal = (self._levene_test(var21, var22)[1]<ct)
                        if var_egal:
                            #ttest
                            return self._ttest_ind(var21, var22, True)
                        else:
                            #Welch test
                            return self._ttest_ind(var21, var22, False)
                    else:
                        normal = (self._ks_normal(var21)[1]<ct) & (self._ks_normal(var22)[1]<ct)
                        if normal:
                            var_egal = (self._levene_test(var21, var22)[1]<ct)
                            if var_egal:
                                #ttest
                                return self._ttest_ind(var21, var22, True)
                            else:
                                #Welch test
                                return self._ttest_ind(var21, var22, False)
                        else:
                            return self._wilcoxon(var21, var22)
                elif nb_cat>2:#ANOVA
                    samples = [var2[(var1==c)] for c in cat]
                    if len(var2)>=30:
                        #No need of normality test
                        var_egal = bool(np.prod([self._levene_test(c[0], c[1])[1]<ct for c in combinations(samples, 2)]))
                        if var_egal:
                            return self._anova(*samples)
                        else:
                            s, p, _ = self._anova(*samples)
                            return s, p, "anova_no_var_equal"
                    else:
                        normal = bool(np.prod([self._ks_normal(s)[1]<ct for s in samples]))
                        if normal:
                            var_egal = bool(np.prod([self._levene_test(c[0], c[1])[1]<ct for c in combinations(samples, 2)]))
                            if var_egal:
                                return self._anova(*samples)
                            else:
                                s, p, _ = self._anova(*samples)
                                return s, p, "anova_no_var_equal"
                        else:
                            return self._kruskal(*samples)
                else:
                    print("no test")
                    return np.nan, np.nan, "no_test"
        else:
            if (var2.dtype == "object") or (var2.dtype.name == "category"):
                return self.relation_independant_var(var2, var1, ct)
            else:
                normal = (self._ks_normal(var1)[1]<ct) & (self._ks_normal(var2)[1]<ct)
                if normal:
                    return self._pearson(var1, var2)
                else:
                    return self._spearman(var1, var2)

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

    def _ks_2samp(self, var1, var2):
        s, p = ks_2samp(var1.values, var2.values)
        return s, p, "kolmogorov_smirnof_2samples"

    def _ks_normal(self, var):
        norm_dist = norm.rvs(size=len(var), loc=np.mean(var.values), scale=np.var(var.values))
        return self._ks_2samp(var, pd.Series(norm_dist))

    def _wilcoxon(self, var1, var2):
        s, p = mannwhitneyu(var1.values, var2.values, alternative='two-sided')
        return s, p, "wilcoxon_rank_test_2sided"

    def _anova(self, *args):
        s, p = f_oneway(*args)
        return s, p, "anova"

    def _kruskal(self, *args):
        s, p = kruskal(*args)
        return s, p, "kruskal"

    def _pearson(self, var1, var2):
        s, p = pearsonr(var1.values, var2.values)
        return s, p, "pearson"

    def _spearman(self, var1, var2):
        s, p = spearmanr(var1.values, var2.values)
        return s, p, "spearman"







