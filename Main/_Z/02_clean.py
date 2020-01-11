# -- coding: utf-8 --
from Tools import common as com
from Tools import geohash32 as gh64
import pandas as pd
import numpy as np


def run():
    csv_data_all = pd.read_csv(com.get_project_path('Data/Csv/OriData/csv_data_all.csv'))
    csv_data_item = pd.read_csv(com.get_project_path('Data/Csv/OriData/tianchi_fresh_comp_train_item.csv'), header=0, names=['item_id', 'item_geo', 'item_cate'])
    # 测试代码时解注下面一条
    # csv_data_all = pd.read_csv(com.get_project_path('Data/Csv/OriData/csv_data_all_h1w.csv'))

    # 处理time
    csv_data_all['time'] = pd.to_datetime(csv_data_all['time'], format='%Y%m%d %H')
    csv_data_all['hour'] = csv_data_all['time'].dt.hour
    csv_data_all['time'] = csv_data_all['time'].dt.normalize()
    csv_data_all['week'] = csv_data_all['time'].apply(lambda a: a.weekday()+1)
    csv_data_all['day_rank'] = csv_data_all['time'].rank(method='dense').apply(lambda a: int(a))
    # del csv_data_all['time']

    # 处理经纬度
    csv_data_item['item_geo'] = csv_data_item['item_geo'].replace('input_data_is_error', '').fillna('').apply(lambda a: gh64.decode(a))
    csv_data_item['item_geo_lat'] = csv_data_item['item_geo'].apply(lambda a: get_lat_lon(a, 0, inplace=-90))
    csv_data_item['item_geo_lon'] = csv_data_item['item_geo'].apply(lambda a: get_lat_lon(a, 1, inplace=180))
    del csv_data_item['item_geo']
    csv_data_all['user_geo'] = csv_data_all['user_geo'].replace('input_data_is_error', '').fillna('').apply(lambda a: gh64.decode(a))
    csv_data_all['user_geo_lat'] = csv_data_all['user_geo'].apply(lambda a: get_lat_lon(a, 0, inplace=90))
    csv_data_all['user_geo_lon'] = csv_data_all['user_geo'].apply(lambda a: get_lat_lon(a, 1, inplace=-180))
    del csv_data_all['user_geo']

    # 保存
    com.save_csv(csv_data_all.sort_values(by=['user_id', 'day_rank', 'item_id', 'beh_type']), com.get_project_path('Data/Csv/ClnData/'), 'csv_data_all.csv')
    com.save_csv(csv_data_item.sort_values(by=['item_id', 'item_cate']), com.get_project_path('Data/Csv/ClnData/'), 'csv_data_item.csv')
    com.save_csv(csv_data_all[csv_data_all['item_id'].isin(csv_data_item['item_id'])].sort_values(by=['user_id', 'day_rank', 'item_id', 'beh_type']),
                 com.get_project_path('Data/Csv/ClnData/'), 'csv_data_p.csv')

    # 保存1w条做来测试代码
    csv_data_all.head(10000).sort_values(by=['user_id', 'day_rank', 'item_id']).to_csv(com.get_project_path('Data/Csv/ClnData/csv_data_all_h1w.csv'), index=None)


def get_lat_lon(tup, sub, inplace=0):
    try:
        return tuple(tup)[sub]
    except IndexError:
        return inplace


if __name__ == '__main__':
    run()
