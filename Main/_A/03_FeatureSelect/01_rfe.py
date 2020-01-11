# -- coding: utf-8 --
from Tools import common as com
from Tools import special as sp
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from xgboost.sklearn import XGBClassifier


def run():
    train_x = pd.read_csv(com.get_project_path('Data/Csv/FeaData/_A/fea_all_label31_dur31_sl1.csv'))
    train_y_ui = sp.get_csv_label(pd.read_csv(com.get_project_path('Data/Csv/ClnData/csv_data_all.csv')), 31)


    print('特征数量: '+str(len(train_x.columns)-2))
    print('训练集数量: ' + str(len(train_x)))

    train_y = sp.get_ui_id(train_x).isin(sp.get_ui_id(train_y_ui)).replace({True: 1, False: 0})
    train_x = train_x.replace({np.inf: 1})


    rfe = RFE(estimator=XGBClassifier(n_estimators=10, learning_rate=0.05, max_depth=5, colsample_bytree=0.8, subsample=0.8, min_child_weight=16), n_features_to_select=30)
    rfe.fit(train_x.drop(['user_id', 'item_id'], axis=1), train_y)

    result = pd.DataFrame(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), train_x.drop(['user_id', 'item_id'], axis=1).columns)), columns=['score', 'feature'])
    result.to_csv(com.get_project_path('Data/Temp/feature_rfe_.csv'), index=None)
    print(result)



if __name__ == '__main__':
    run()
