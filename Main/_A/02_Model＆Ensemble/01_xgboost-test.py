# -- coding: utf-8 --
from Tools import common as com
from Tools import special as sp
from sklearn import metrics
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np


FEATURE_PATH = 'Data/Csv/FeaData/_A/'
RESULT_PATH = 'Data/Csv/ResData/_A/'

def run():
    data_all = pd.read_csv(com.get_project_path('Data/Csv/ClnData/csv_data_all.csv'))

    train_x = pd.read_csv(com.get_project_path(FEATURE_PATH+'fea_all_label31_dur31_sl3.csv'))
    train_x['ui_id'] = sp.get_ui_id(train_x)
    test_x = pd.read_csv(com.get_project_path(FEATURE_PATH+'fea_all_label32_dur31_sl3_p.csv'))
    test_x['ui_id'] = sp.get_ui_id(test_x)

    train_y = sp.get_csv_label(data_all, 31)
    train_y['ui_id'] = sp.get_ui_id(train_y)
    train_y = train_x['ui_id'].isin(train_y['ui_id']).replace({True: 1, False: 0})

    print('特征数量: '+str(len(train_x.columns)-3))
    print('训练集数量: ' + str(len(train_x)))
    # ########### 搞模型 ############ #
    pre_label = xgb_pre(train_x.drop(['user_id', 'item_id', 'ui_id'], axis=1), train_y,
                        test_x.drop(['user_id', 'item_id', 'ui_id'], axis=1))

    tmp = list(pre_label.sort_values(ascending=False))[500]
    pre_label = pre_label.apply(lambda a: a>=tmp).replace({True: 1, False: 0})
    test_x['label'] = pre_label
    csv_fea_label24_dur14_p = test_x[test_x['label']==1].loc[:, ['user_id', 'item_id']]
    save_name = '_A_02_xgb_202001032331.csv'
    com.save_csv(csv_fea_label24_dur14_p.loc[:, ['user_id', 'item_id']], com.get_project_path(RESULT_PATH), save_name)


def xgb_pre(train_x, train_y, test_x, num_round=500, params=None, test_y=None):
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x, label=test_y)
    if params is None:
        params = {
            'objective': 'binary:logistic',
            # 'objective': 'rank:pairwise',
            'silent': 0,
            'eta': 0.05,
            'max_depth': 5,
            'colsample_bytree': 0.8,
            'subsample': 0.8,
            'min_child_weight': 16,
            'tree_method': 'exact',
            'eval_metric': 'auc',
        }
    watchlist=[(dtrain, 'train'), (dtest, 'test')]
    if test_y is None:
        bst = xgb.train(params, dtrain, num_boost_round=num_round)
    else:
        bst = xgb.train(params, dtrain, num_boost_round=num_round, evals=watchlist)
    pre_label = pd.Series(bst.predict(dtest))
    return pre_label


if __name__ == '__main__':
    run()
