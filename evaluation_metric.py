import pandas as pd
import numpy as np
import lightgbm as lgb

def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

    def top_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)

def amex_list(pred, act):
    mirror_act = [1 if x == 0 else -1 for x in act]
    sorted_data = [0 if n == 1 else 1 for _,n in sorted(zip(pred,mirror_act), reverse=True)]
    weight = [20 - i * 19 for i in sorted_data]
    sum_weight = sum(weight)
    four_pct_cutoff = int(sum_weight * 0.04)

    weight_cumsum_v = 0
    cum_pos_found = 0
    weight_cumsum = []
    random = []
    cum_pos_found_list = []
    lorentz = []
    gini = []

    total_pos = sum(sorted_data)
    for indx, weight_v in enumerate(weight):
        
        weight_cumsum_v += weight_v
        weight_cumsum.append(weight_cumsum_v)

        random_v = weight_cumsum_v/sum_weight
        random.append(random_v)

        cum_pos_found += sorted_data[indx] * weight_v
        cum_pos_found_list.append(cum_pos_found)

        lorentz_v = cum_pos_found/total_pos
        lorentz.append(lorentz_v)

        gini_v = (lorentz_v - random_v) * weight_v
        gini.append(gini_v)

    total_neg = len(sorted_data) - total_pos
    gini_max = 10 * total_neg * (total_pos + 20 * total_neg - 19) / (total_pos + 20 * total_neg)

    indx_cutoff = sum(map(lambda x : x <= four_pct_cutoff, weight_cumsum))

    d = sum(sorted_data[:indx_cutoff]) / total_pos
    g_not_normalized = sum(gini)

    return 0.5 * (d + g_not_normalized/gini_max)

def lgb_amex_metric(y_true, y_pred):
    """The competition metric with lightgbm's calling convention"""
    return ('amex',
            amex_metric(pd.DataFrame({'target': y_true}), pd.Series(y_pred, name='prediction')),
            True)

def amex_metric_np(y_pred: np.ndarray, y_true: np.ndarray) -> float:

    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0]+ top_four)


def xgb_amex(y_pred, y_true):
    return 'amex', amex_metric_np(y_pred,y_true.get_label())


def amex_metric_v2(y_true, y_pred):
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:,0]==0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])
    gini = [0,0]
    for i in [1,0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:,0]==0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] *  weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)
    return 0.5 * (gini[1]/gini[0] + top_four)

# ====================================================
# LGBM amex metric
# ====================================================
def lgb_amex_metric_v2(y_pred, y_true):
    y_true = y_true.get_label()
    return 'amex_metric', amex_metric_v2(y_true, y_pred), True


