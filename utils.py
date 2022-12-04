# Ramzor - student_drouput-advanced-EDA.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calc_metrics(the_df, the_index=['baseline'], pred_col='Naive.Xgboost2_predict_response',
                 ground_truth='IND_NIRSHAM_SEM', id_col='id', append_df=pd.DataFrame()):
    """
    Calc Qualisense format model metrics on specific prediction column and ground truth, return as DataFrame, optionally append to existing metrics df
    :param the_df: A df with a predictions column (values 1,0)
    :param the_index: name of the expriement in list, return as the index of the row
    :param pred_col: name of the prediction response column (values 1,0)
    :param ground_truth: The model's target feature
    :param id_col: Can be None. if exists, adds the nunique id col
    :param append_df: Can be None. if exists, adds the current calcs to prior calcs df
    :return: A dataframe with the results
    """
    my_dict = {}
    if not isinstance(the_index, list):
        the_index = [the_index]
    TP = sum((the_df[ground_truth] == 1) & (the_df[pred_col] == 1))
    TN = sum((the_df[ground_truth] == 0) & (the_df[pred_col] == 0))
    FP = sum((the_df[ground_truth] == 0) & (the_df[pred_col] == 1))
    FN = sum((the_df[ground_truth] == 1) & (the_df[pred_col] == 0))
    # print(TP, FP, FN, TN)
    my_dict['distribution'] = the_df[ground_truth].mean()
    my_dict['observations'] = TP + TN + FP + FN
    if id_col:
        my_dict['unique_ids'] = the_df[id_col].nunique()
    my_dict['recall'] = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    my_dict['precision'] = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    my_dict['f1'] = (2 * my_dict['recall'] * my_dict['precision']) / (my_dict['recall'] + my_dict['precision']) if (
                                                                                                                           my_dict[
                                                                                                                               'recall'] +
                                                                                                                           my_dict[
                                                                                                                               'precision']) > 0 else 0
    my_dict['lift'] = (my_dict['precision'] / my_dict['distribution']) if my_dict['distribution'] > 0 else np.nan
    my_dict['diff1'] = TP - FP
    my_dict['diff0'] = TN - FN
    my_dict['TP'] = TP
    my_dict['FP'] = FP
    my_dict['TN'] = TN
    my_dict['FN'] = FN
    my_dict['predicted_1'] = (TP + FP) / (TP + FP + TN + FN)
    my_dict['predicted_0'] = (TN + FN) / (TP + FP + TN + FN)
    res = pd.DataFrame(my_dict, index=the_index)
    if len(append_df) > 0:
        return pd.concat([append_df, res], axis=0)
    return res


def get_metrics_on_all_thresholds(the_df, prob_col):
    """
    Calc metrics from calc_metrics() on 100 thresholds, return in a dataframe
    :param the_df: A df with a probability column
    :param prob_col: the probability column name
    :return: A dataframe with 100 rows, each row with metrics from calc_metrics()
    """
    thresholds = np.arange(0, 1, 0.01)
    the_metrics = pd.DataFrame()
    for thresh in thresholds:
        the_df['temp'] = (the_df[prob_col] >= thresh) * 1
        the_metrics = calc_metrics(the_df, the_index=[thresh], pred_col='temp', ground_truth='target',
                                   append_df=the_metrics)
    return the_metrics


def move_cols_to_first(the_df, first_cols):
    """
    Rearrange df columns so that first_cols will be first
    :param the_df: A dataframe
    :param first_cols: a list of columns to be first cols in the df
    :return: the dataframe rearranged
    """
    the_df = pd.concat([the_df[first_cols], the_df.loc[:, ~the_df.columns.isin(first_cols)]], axis=1)
    return the_df


def get_all_models_max_f1_thresh(the_df):
    """
    Calc metrics on all models results, where the threshold was maximized for f1-score for each model
    :param the_df: A dataframe with one or more prediction columns that their name has 'prob' in them
    :return: A dataframe with a row per each model, with metrics on maximized threshold on f1-score
    """
    prob_cols = [col for col in the_df.columns if 'prob' in col]
    all_models_max_f1_thresh = {}
    len_cols = len(prob_cols)
    for i, col in enumerate(prob_cols):
        print(i + 1, "/", len_cols, " :", col)
        the_metrics_thresh = get_metrics_on_all_thresholds(the_df, col)
        max_f1_metrics = the_metrics_thresh[the_metrics_thresh.index == the_metrics_thresh.f1.idxmax()].iloc[0]
        all_models_max_f1_thresh[col] = max_f1_metrics.to_dict()
        all_models_max_f1_thresh[col]['threshold'] = \
            the_metrics_thresh[the_metrics_thresh.index == the_metrics_thresh.f1.idxmax()].index[0]
    res = pd.DataFrame(all_models_max_f1_thresh).T.sort_values('f1', ascending=False).drop(
        columns=['distribution', 'observations', 'unique_ids'], errors='ignore')
    res = move_cols_to_first(res, ['f1', 'threshold', 'precision', 'recall', 'lift'])
    return res


def get_metrics_on_all_col_values(the_df, the_col, pred_col='Naive.Logistic_predict_response', ground_truth='target',
                                  to_print=True):
    """
    Calc metrics with calc_metrics() for a specific feature, on each specific value
    :param the_df: A df with a predictions column (values 1,0)
    :param the_col: the feature name to calc metrics on all each one of its values
    :param pred_col: the feature name of reposne col from the model  (values 1,0)
    :param ground_truth: the target name
    :param to_print: bool to report which feature value are we calclating on now
    :return: the df with indexes as the feature values, and columns as the metrics
    """
    all_values = np.sort(the_df[the_col].unique())
    metrics = pd.DataFrame()
    for val in all_values:
        print(val) if to_print else ""
        metrics = calc_metrics(the_df[the_df[the_col] == val], the_index=[val], pred_col=pred_col,
                               ground_truth=ground_truth, append_df=metrics)
    return metrics


def plot_metric_to_feature(the_metrics, col='nekudot_zchut_sem_b4_sem', max_val=75, min_val=10, distribution=True,
                           unique_ids=False, observations=False, precision=True, recall=False, lift=False,
                           the_xticks=None, the_yticks=None, all=False, f1=False, title='ביצועי מודל הרמזור לכל נ"ז',
                           remove_non_integer_vals=True, distribution_heb_title='אחוז המסיימים',
                           recall_heb_title='אחוז המסיימים שנתפסו',
                           unique_ids_heb_title='מספר הסטודנטים', n_observations_heb_title='מספר התצפיות',
                           lift_heb_title='המשלים להופכי של אחוז ההרמה )אחוז המסיימים/דיוק(',
                           f1_heb_title='מממוצע הרמוני של הדיוק ואחוז המסיימים שנתפסו'):
    """
    Plot a metrics df with row per each columns value 
    :param the_metrics: a metrics df with row per each columns value 
    :param col: the name of the col where the metrics df with row per each value was calculated on
    :param max_val: The feature maximum value to plot (end of x axis)
    :param min_val: The feature minimum value to plot (start of x axis)
    :param distribution: bool for showing target distribution per each of the feature's values
    :param unique_ids: bool for showing target unique_ids per each of the feature's values
    :param observations: bool for showing target n observations per each of the feature's values
    :param precision: bool for showing target precision per each of the feature's values
    :param recall: bool for showing target recall per each of the feature's values
    :param lift: bool for showing target lift per each of the feature's values
    :param the_xticks: optional: specify exact xticks (feature's values to show)
    :param the_yticks: optional: specify exact yticks (metrics values range)
    :param all: all 4 bools set to True: distribution, precision, recall, lift
    :param f1: bool for showing target f1 per each of the feature's values
    :param title: the plot's title
    :param remove_non_integer_vals: ignore non-integer feature's values. default is True.
    :param distribution_heb_title: distribution legend description in hebrew
    :param recall_heb_title: recall legend description in hebrew
    :param unique_ids_heb_title: unique_ids legend description in hebrew
    :param n_observations_heb_title: N observations legend description in hebrew
    :param lift_heb_title: lift legend description in hebrew
    :param f1_heb_title: f1-score legend description in hebrew
    :return: the plot of the metrics df with row per each columns value. Can be fed into a display() command
    """
    max_val = max_val if max_val else the_metrics.index.max()
    min_val = min_val if min_val else the_metrics.index.min()
    if all:
        distribution = True
        precision = True
        recall = True
        lift = True
    the_xticks = the_xticks if the_xticks else np.concatenate(
        [np.arange(the_metrics.index.min(), 26, 1), np.arange(30, 55, 5), np.arange(60, the_metrics.index.max(), 10)])
    the_xticks = the_xticks if not max_val else the_xticks[the_xticks <= max_val]
    the_xticks = the_xticks if not min_val else the_xticks[the_xticks >= min_val]
    the_yticks = the_yticks if the_yticks else np.arange(0, 1.1, 0.1)
    to_plot = the_metrics[(the_metrics.index < max_val) & (the_metrics.index > min_val)]
    to_plot = to_plot[to_plot.index.isin(np.arange(0, to_plot.index.max(), 1))] if remove_non_integer_vals else to_plot
    # to_plot = to_plot * 100
    to_plot = to_plot[to_plot.index.isin(to_plot.precision.dropna().index)]
    plt.rcParams.update({'font.size': 14})  # must set in top
    to_plot.precision.plot(figsize=(20, 8), label='דיוק'[::-1] + " (precision)") if precision else ""
    to_plot.recall.plot(figsize=(20, 8), label=recall_heb_title[::-1] + " (recall)") if recall else ""
    to_plot.distribution.plot(figsize=(20, 8),
                              label=distribution_heb_title[::-1] + " (distribution)") if distribution else ""
    to_plot.unique_ids.plot(figsize=(20, 8), label=unique_ids_heb_title[::-1] + " (unique ids)") if unique_ids else ""
    to_plot.observations.plot(figsize=(20, 8),
                              label=n_observations_heb_title[::-1] + " (observations)") if observations else ""
    to_plot['lift'] = 1 - 1 / to_plot['lift']
    to_plot.lift.plot(figsize=(20, 8), label=lift_heb_title[::-1] + " (1-1/lift)") if lift else ""
    to_plot.f1.plot(figsize=(20, 8), label=f1_heb_title[::-1] + " (f1)") if f1 else ""
    plt.legend()
    plt.xticks(the_xticks)
    plt.yticks(the_yticks)
    plt.xlabel(col)
    title = title[::-1]
    to_plot['weighted_observations'] = to_plot['observations'] / to_plot['observations'].sum()
    to_plot['weighted_distribution'] = to_plot['distribution'] * to_plot['weighted_observations']
    to_plot['weighted_recall'] = to_plot['recall'] * to_plot['weighted_observations']
    to_plot['weighted_precision'] = to_plot['precision'] * to_plot['weighted_observations']
    to_plot['weighted_f1'] = (2 * to_plot['weighted_recall'] * to_plot['weighted_precision']) / (
            to_plot['weighted_recall'] + to_plot['weighted_precision'])
    to_plot['weighted_lift'] = to_plot['weighted_precision'] / to_plot['weighted_distribution'] / 100
    weighted_cols = [col for col in to_plot.columns if 'weighted' in col]
    to_plot_statistics = (to_plot[weighted_cols].sum() * 100).round(0).astype(int).astype(str) + "%"
    to_plot_statistics = to_plot_statistics.drop(index=['weighted_observations'], errors='ignore')
    title_statistics = 'עבור כלל הנתונים בתרשים:'[::-1]
    # if
    title = title
    plt.title(title)
    plt.xlabel(col + "\n\n" + title_statistics + "\n" + to_plot_statistics.to_string().replace('weighted_', ""))
    plt.ylabel('percent')
    plt.show()
    return to_plot


def compare_feature_distributions_two_dfs(the_df1, df1_name, the_df2, df2_name, groupby_feature='id',
                                          feature='nekudot_zchut_sem_b4_sem', agg_func='max', add_cumsum=True):
    """
    Show a side-by-side comparison two df2 aggregated function value_counts() of value of a feature grouped by other feature
    :param the_df1: The 1st df to show aggregated values on
    :param df1_name: The 1st df name (will be column name)
    :param the_df2: The 2nd df to show aggregated values on
    :param df2_name: The 2nd df name (will be column name)
    :param groupby_feature: the feautre to groupby on. Default is id.
    :param feature: the feature to calc aggregated values on. Default is nekudot_zchut_sem_b4_sem
    :param agg_func: The calculation to run. Default is max. Do notice value_counts() is calculated on agg_func's results.
    :param add_cumsum: optional: bool to add cumsum to the calculated cols. Might be easier to compare, but might not work for all agg_funcs.
    :return: The dataframe with one col per each df, where each row is the feature's value.
             if add_cumsum=True, another column for each df with the orig col cumsum.
    """
    agg_val_df1 = the_df1.groupby(groupby_feature)[feature].agg(agg_func).value_counts(normalize=True,
                                                                                       dropna=False).to_frame().rename(
        columns={feature: df1_name})
    agg_val_df2 = the_df2.groupby(groupby_feature)[feature].agg(agg_func).value_counts(normalize=True,
                                                                                       dropna=False).to_frame().rename(
        columns={feature: df2_name})
    res = pd.concat([agg_val_df1, agg_val_df2], axis=1)
    if add_cumsum:
        res = pd.concat([res, res[df1_name].cumsum().to_frame().add_suffix('_cumsum'),
                         res[df2_name].cumsum().to_frame().add_suffix('_cumsum')], axis=1)
    return res


def get_diff_in_cols(the_df1, the_df2):
    """
    get list of cols that are in 1st df and not in 2nd df
    :param the_df1: the 1st df
    :param the_df2: the 2nd df
    :return: the list of cols that are in 1st df and not in 2nd df
    """
    res = [col for col in the_df1.columns if col not in the_df2.columns]
    return res


def get_metrics_df_on_feature_on_prob(the_df, threshold=None, target_col='target',
                                      groupby_col='nekudot_zchut_sem_b4_sem',
                                      proba_col='Naive.Xgboost2_predict_probability',
                                      response_col='Naive.Xgboost2_predict_response'):
    """
    Calc a df with a metrics row for each feature value, on a specified threshold.
    optional: first find f1-optimized threshold.
    It's very similair to get_metrics_on_all_col_values but with probability and threshold optimization.
    :param the_df: a dataframe with prediction cols, to calc the metrics on
    :param threshold: optional: a specific threshold to calc metrics on. Default is False = will find f1 optimized threhold on proba_col
    :param target_col: the name of the target feature
    :param groupby_col: the name of the feature to calc metrics on each of it's values
    :param proba_col: the name of the probability col to use for calculation of the metrics
    :param response_col: the name of the reposne col. a new col will be created with the name + the threshold
    :return:
    """
    if not threshold:
        print(f"No threshold was inserted, calculating threshold on {proba_col} that maximizes f1:")
        models_max_f1_df = get_all_models_max_f1_thresh(the_df)
        display(models_max_f1_df)
        threshold = models_max_f1_df[models_max_f1_df.index == proba_col].threshold.values[0]
        print(f"Maximized Threshold: {threshold}")
    pred_col = response_col + "_" + str(threshold)
    the_df[pred_col] = (the_df[proba_col] >= threshold) * 1
    the_df_metrics_nz = get_metrics_on_all_col_values(the_df, the_col=groupby_col, pred_col=pred_col,
                                                      ground_truth=target_col)
    return the_df_metrics_nz


def plot_metrics_of_feature_two_dfs(the_df_metrics_nz, the_df2_metrics_nz,
                                    the_df_title='ביצועי מודל הרמזור לכל נ"ז, מודל שאומן על k611',
                                    the_df2_title='ביצועי מודל הרמזור לכל נ"ז, מודל שאומן על k614', min_val=-1,
                                    max_val=121, lift=False, precision=True, f1=True, recall=True, distribution=False,
                                    return_table=False):
    """
    Run plot_metric_to_feature() for two dfs at once, and control whether to output the data tables

    :param the_df_metrics_nz: the 1st metrics df with row per each columns value
    :param the_df2_metrics_nz: the 2nd metrics df with row per each columns value
    :param the_df_title: the 1st plot's title
    :param the_df2_title: the 2nd plot's title
    :param min_val: The feature minimum value to plot (start of x axis)
    :param max_val: The feature maximum value to plot (end of x axis)
    :param lift: bool for showing target lift per each of the feature's values
    :param precision: bool for showing target precision per each of the feature's values
    :param f1: bool for showing target f1 per each of the feature's values
    :param recall: bool for showing target recall per each of the feature's values
    :param distribution: bool for showing target distribution per each of the feature's values
    :param return_table: bool for also outputting the data tables under each plot. Default is False.
    :return: Two plots, one above the other, with possibly the data tables beneath each plot.
             Each plot is of the metrics df with row per each columns value.
    """
    if return_table:
        display(
            plot_metric_to_feature(the_df_metrics_nz, max_val=max_val, min_val=min_val, lift=lift, precision=precision,
                                   f1=f1, recall=recall, distribution=distribution,
                                   title=the_df_title))
        display(
            plot_metric_to_feature(the_df2_metrics_nz, max_val=max_val, min_val=min_val, lift=lift, precision=precision,
                                   f1=f1, recall=recall, distribution=distribution,
                                   title=the_df2_title))
    else:
        plot_metric_to_feature(the_df_metrics_nz, max_val=max_val, min_val=min_val, lift=lift, precision=precision,
                               f1=f1, recall=recall, distribution=distribution,
                               title=the_df_title)
        plot_metric_to_feature(the_df2_metrics_nz, max_val=max_val, min_val=min_val, lift=lift, precision=precision,
                               f1=f1, recall=recall, distribution=distribution,
                               title=the_df2_title)

        
def add_ramzor_cols(the_df, ramzor_edges=None, the_prob_col=None):
    """
    Groups the models probability predictions into 3 colors: red (low), yellow (medium) and green (high)
    :param the_df: a DataFrame with probability predictions column
    :param ramzor_edges: the edges defining the groups. By default, [0,0.333,0.667,1]
    :param the_prob_col: the probability columns to use. if not provided, take the first column containing the string 'prob'
    :return: the dataframe with two additional group columns - the group probability range and the group color
    """
    if not the_prob_col:
        the_prob_col = [col for col in the_df.columns if 'prob' in col][0]
    if not ramzor_edges:
        ramzor_edges = [0,0.333,0.667,1]
    the_df['ramzor_prob'] = pd.cut(the_df[the_prob_col],ramzor_edges)
    ramzor_mapper = dict(zip(the_df['ramzor_prob'].unique().tolist(),['Green','Yellow','Red']))
    the_df['ramzor'] = the_df['ramzor_prob'].map(ramzor_mapper)
    return the_df
