from evaluation_metric import *
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
base_seed = 0

parameters_lgbm = {
    'objective': 'binary',
    'boosting': 'gbdt',
    'learning_rate': 0.01,
    'min_child_samples': 500,
    'reg_lambda': 15,
    'feature_fraction':0.3,
    'bagging_fraction':0.3,
    'n_estimators':100000
    }

def base_model_lgbm(train_data, train_labels, parameters = parameters_lgbm):
    
    
    n_fold = 5
    n_seed = 1

    kf = StratifiedKFold(n_splits=n_fold)

    importances = []
    models = {}
    df_scores = []

    for fold, (idx_tr, idx_va) in enumerate(kf.split(train_data, train_labels)):
        
        X_tr = train_data.iloc[idx_tr]
        X_va = train_data.iloc[idx_va]
        y_tr = train_labels.iloc[idx_tr]
        y_va = train_labels.iloc[idx_va]
        
        lgb_train_data = lgb.Dataset(X_tr, label=y_tr)
        lgb_val_data = lgb.Dataset(X_va, label=y_va)
        
        for seed in range(n_seed):
            print('Fold: '+str(fold)+ ' - seed: '+str(seed))
            key = str(fold)+'-'+str(seed)

            parameter_variable = {
                'seed':seed,
            }

            parameters.update(parameter_variable)
    
            clf = lgb.train(parameters,
                                lgb_train_data,
                                valid_sets = [lgb_val_data],
                                verbose_eval = 500,
                                feval=amex_metric_mod_lgbm,
                                early_stopping_rounds=1500)

            score = amex_metric(y_va.reset_index(drop=True), pd.Series(clf.predict(X_va)).rename('prediction'))
            models[key] = clf
            df_scores.append((fold, seed, score))
            print(f'Fold: {fold} - seed: {seed} - score {score:.2%}')
            importances.append(clf.feature_importance(importance_type='gain'))
    
    df_results = pd.DataFrame(df_scores,columns=['fold','seed','score']).pivot(index='fold',columns='seed',values='score')
    df_results.loc['seed_mean']= df_results.mean(numeric_only=True, axis=0)
    df_results.loc[:,'fold_mean'] = df_results.mean(numeric_only=True, axis=1)
    
    score_cv = df_results.loc[:,'fold_mean']['seed_mean']
    return models, importances, df_results, score_cv


##############################################################################
##############################################################################
##############################################################################
##############################################################################

def xgb_train(x, y, xt, yt,seed):
    print("# of features:", x.shape[1])
    dtrain = xgb.DMatrix(data=x, label=y)
    dvalid = xgb.DMatrix(data=xt, label=yt)
    params = {
            'objective': 'binary:logistic', 
            'tree_method': 'hist', 
            'max_depth': 7,
            'subsample':0.88,
            'colsample_bytree': 0.5,
            'gamma':1.5,
            'min_child_weight':8,
            'lambda':70,
            'eta':0.03,
            'random_state': seed
    }
    watchlist = [(dvalid, 'eval')]
    bst = xgb.train(params, dtrain=dtrain,
                num_boost_round=9999,evals=watchlist,
                early_stopping_rounds=1000, feval=xgb_amex, maximize=True,
                verbose_eval=200)
    print('best ntree_limit:', bst.best_ntree_limit)
    print('best score:', bst.best_score)
    pred = bst.predict(dvalid, iteration_range=(0,bst.best_ntree_limit))

    return pred, bst