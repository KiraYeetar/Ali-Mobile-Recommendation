# -- coding: utf-8 --
from Tools import common as com
from Tools import special as sp
from sklearn import metrics
import numpy as np
import xgboost as xgb
import pandas as pd
import math

train_y_ui = None
test_y_ui = None
train_ui = None
test_ui = None

def run():
    global train_y_ui
    global test_y_ui
    global train_ui
    global test_ui

    train_x = pd.read_csv(com.get_project_path('Data/Csv/FeaData/_A/fea_all_label30_dur31_sl1.csv'))
    test_x = pd.read_csv(com.get_project_path('Data/Csv/FeaData/_A/fea_all_label31_dur31_sl1_p.csv'))

    train_y_ui = sp.get_csv_label(pd.read_csv(com.get_project_path('Data/Csv/ClnData/csv_data_all.csv')), 30)
    test_y_ui = sp.get_csv_label(pd.read_csv(com.get_project_path('Data/Csv/ClnData/csv_data_p.csv')), 31)

    print('特征数量: '+str(len(train_x.columns)-2))
    print('训练集数量: ' + str(len(train_x)))

    train_ui = train_x.loc[:, ['user_id', 'item_id']]
    test_ui = test_x.loc[:, ['user_id', 'item_id']]

    train_y = sp.get_ui_id(train_x).isin(sp.get_ui_id(train_y_ui)).replace({True: 1, False: 0})
    test_y = sp.get_ui_id(test_x).isin(sp.get_ui_id(test_y_ui)).replace({True: 1, False: 0})
    # ########### 模型 ############ #
    pre_label = xgb_pre(train_x.drop(['user_id', 'item_id'], axis=1), train_y, test_x.drop(['user_id', 'item_id'], axis=1), test_y=test_y, if_save_imp=False)

    tmp = list(pre_label.sort_values(ascending=False))[700]
    pre_label = pre_label.apply(lambda a: a>=tmp).replace({True: 1, False: 0})
    test_x['label'] = pre_label
    test_pre_ui = test_x[test_x['label']==1].loc[:, ['user_id', 'item_id']]
    sp.f1_score(test_pre_ui, test_y_ui.loc[:, ['user_id', 'item_id']], if_print=True)
    del test_x['label']


def xgb_pre(train_x, train_y, test_x, num_round=500, params=None, test_y=None, if_save_imp=True):
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x, label=test_y)
    if params is None:
        params = {
            'objective': 'binary:logistic',
            # 'objective': 'rank:pairwise',
            'eta': 0.01,
            'max_depth': 5,
            'colsample_bytree': 0.8,
            'subsample': 0.8,
            'min_child_weight': 16,
            'tree_method': 'exact',
            # 'gamma': 0.1,
            # 'scale_pos_weight': 10,
            # 'max_delta_step': 0.7,
            # 'eval_metric': 'auc',
        }
    watchlist=[(dtrain, 'train'), (dtest, 'test')]
    if test_y is None:
        bst = xgb.train(params, dtrain, num_boost_round=num_round)
    else:
        bst = xgb.train(params, dtrain, num_round, watchlist, feval=evalerror)

    if if_save_imp:
        imp_dict = bst.get_fscore(fmap='')
        imp = pd.DataFrame({'column': list(imp_dict.keys()), 'importance': list(imp_dict.values())})
        com.save_csv(imp.sort_values(by='importance'), com.get_project_path('Data/Temp/'), 'xgb-val_importance.csv')
    pre_label = pd.Series(bst.predict(dtest))
    return pre_label


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0-preds)
    return grad, hess


def evalerror(preds, dtrain):
    if len(dtrain.data)==len(test_ui):
        pre = test_ui
        true = test_y_ui
    else:
        pre = train_ui
        true = train_y_ui

    pre['label'] = preds
    pre = pre.sort_values(by='label', ascending=False).head(500)
    score = sp.f1_score(true, pre)
    return 'sp-f1', score


if __name__ == '__main__':
    run()
