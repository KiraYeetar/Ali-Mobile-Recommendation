# -- coding: utf-8 --
from Tools import common as com
from Tools import special as sp
import pandas as pd
import time
import gc

SET_LENGTH = 3
FEATURE_PATH = 'Data/Csv/FeaData/_A/'

def run():
    csv_data_all = pd.read_csv(com.get_project_path('Data/Csv/ClnData/csv_data_all.csv'))
    csv_data_item = pd.read_csv(com.get_project_path('Data/Csv/ClnData/csv_data_item.csv'))
    csv_data_p = pd.read_csv(com.get_project_path('Data/Csv/ClnData/csv_data_p.csv'))

    csv_data_all['ui_id'] = sp.get_ui_id(csv_data_all)
    csv_data_p['ui_id'] = sp.get_ui_id(csv_data_p)
    csv_data_all['uc_id'] = sp.get_uc_id(csv_data_all)
    csv_data_p['uc_id'] = sp.get_uc_id(csv_data_p)

    get_feature(data_all=csv_data_all, data_p=csv_data_p, data_item=csv_data_item, label_day_rank=31, p_only=False, duration=31, save=True)
    get_feature(data_all=csv_data_all, data_p=csv_data_p, data_item=csv_data_item, label_day_rank=32, p_only=True, duration=31, save=True)


'''
详细说一下参数: 
data_p 为只含子集商品的用户行为数据
data_all 为含全集商品的用户行为数据
label_day_rank 指定哪一天(day_rank为准)为标签提取特征; 即以label_day_rank之前的数据提取特征，label_day_rank为预测目标
duration 提取标签前几天的特征，默认为一周
p_only 指定是否只针对子集商品提取特征，默认为True 
data_item 为商品数据，可以用于提取商品经纬度特征，默认不传
save 是否存储特征，默认不存；不存返回特征DataFrame，存返回存储地址
——————————
思路：取用户前天的浏览商品为标签，用户X商品 为单独的一行进行特征提取
'''
def get_feature(data_all, data_p, label_day_rank, duration=7, p_only=True, data_item=None, save=False):
    # 第一部分: 用户的特征
    fea_user_path = get_user_feature(data_all=data_all, data_p=data_p, data_item=data_item, label_day_rank=label_day_rank, duration=duration, p_only=p_only, save=True)
    # fea_user_path = com.get_project_path(FEATURE_PATH) + get_save_name(label_day_rank, duration, p_only, index='user')

    # 第二部分: 商品的特征
    fea_item_path = get_item_feature(data_all=data_all, data_p=data_p, data_item=data_item, label_day_rank=label_day_rank, duration=duration, p_only=p_only, save=True)
    # fea_item_path = com.get_project_path(FEATURE_PATH) + get_save_name(label_day_rank, duration, p_only, index='item')

    # 第三部分: 用户X商品 的特征
    fea_ui_path = get_ui_feature(data_all=data_all, data_p=data_p, data_item=data_item, label_day_rank=label_day_rank, duration=duration, p_only=p_only, save=True)
    # fea_ui_path = com.get_project_path(FEATURE_PATH) + get_save_name(label_day_rank, duration, p_only, index='ui')

    # 组合特征
    data_fea = pd.read_csv(fea_ui_path).loc[:, ['user_id', 'item_id']]
    data_fea = pd.merge(data_fea, pd.read_csv(fea_user_path), on='user_id', how='left')
    data_fea = pd.merge(data_fea, pd.read_csv(fea_item_path), on='item_id', how='left')
    data_fea = pd.merge(data_fea, pd.read_csv(fea_ui_path), on=['user_id', 'item_id'], how='left')

    if save is True:
        save_name = get_save_name(label_day_rank, duration, p_only, index='all')
        com.save_csv(data_fea, com.get_project_path(FEATURE_PATH), save_name)
        return com.get_project_path(FEATURE_PATH) + save_name
    else:
        return data_fea


'''
提取用户特征
'''
def get_user_feature(data_all, data_p, label_day_rank, duration=7, p_only=True, data_item=None, save=False):
    data_all = data_all[(data_all['day_rank']>=label_day_rank-duration) & (data_all['day_rank']<=label_day_rank-1)]
    data_p = data_p[(data_p['day_rank']>=label_day_rank-duration) & (data_p['day_rank']<=label_day_rank-1)]

    if p_only is True:
        data_fea = data_p[(data_p['day_rank']>=label_day_rank-SET_LENGTH) & (data_p['beh_type']==1)].loc[:, ['user_id']].drop_duplicates()
    else:
        data_fea = data_all[(data_all['day_rank']>=label_day_rank-SET_LENGTH) & (data_all['beh_type']==1)].loc[:, ['user_id']].drop_duplicates()

    # 用户在 全集中的/子集中的 总/前1天内/前2天内/前3天内  浏览/收藏/购物车/购买 的计数
    for data_index in ['data_all', 'data_p']:
        for duration_time in [duration, 1, 2, 3]:
            for beh_type in [1, 2, 3, 4]:
                if data_index=='data_all': data = data_all
                else: data = data_p
                fea_name = 'beh_type_'+str(beh_type)+'_count&user&latest_'+str(duration_time)+'&'+str(data_index)
                feature = com.pivot_table_plus(data[(data['beh_type']==beh_type) & (data['day_rank']>=label_day_rank-duration_time)],
                                               index='user_id', values='beh_type', aggfunc='count', new_name=fea_name)
                data_fea = pd.merge(data_fea, feature, on='user_id', how='left')
                data_fea[fea_name] = data_fea[fea_name].fillna(0).astype(int)
                print('# -- '+fea_name+' complete -- #')

    # 用户在 全集中/子集中 浏览/收藏/购物车/购买 过几种商品
    for data_index in ['data_all', 'data_p']:
        for beh_type in [1, 2, 3, 4]:
            if data_index=='data_all': data = data_all
            else: data = data_p
            fea_name = 'item_count&'+'user&'+'beh_type_'+str(beh_type)+'&'+str(data_index)
            feature = com.pivot_table_plus(data[(data['beh_type']==beh_type)], index='user_id', values='item_id',
                                           aggfunc=com.count_with_drop_duplicates_for_series, new_name=fea_name)
            data_fea = pd.merge(data_fea, feature, on='user_id', how='left')
            data_fea[fea_name] = data_fea[fea_name].fillna(0).astype(int)
            print('# -- ' + fea_name + ' complete -- #')

    # 用户在 全集中的/子集中的 转化率
    data_fea['user_ctr&data_all'] = 1.*data_fea['beh_type_4_count&user&latest_'+str(duration)+'&data_all'] / data_fea['beh_type_1_count&user&latest_'+str(duration)+'&data_all']
    data_fea['user_ctr&data_all'] = data_fea['user_ctr&data_all'].fillna(0)
    data_fea['user_ctr&data_p'] = 1.*data_fea['beh_type_4_count&user&latest_'+str(duration)+'&data_p'] / data_fea['beh_type_1_count&user&latest_'+str(duration) +'&data_p']
    data_fea['user_ctr&data_p'] = data_fea['user_ctr&data_p'].fillna(0)

    # 用户最后一次 浏览/收藏/购物车/购买 距标签多少小时
    for beh_type in [1, 2, 3, 4]:
        fea_name = 'beh_type_'+str(beh_type)+'_latest_to_now_hour&user_id'
        feature = data_all.loc[:, ['user_id', 'day_rank', 'hour']].sort_values(by=['user_id', 'day_rank', 'hour'], ascending=[0, 0, 0]).drop_duplicates('user_id')
        feature[fea_name] = feature['day_rank'].apply(lambda a: label_day_rank-a)
        feature[fea_name] = (feature[fea_name]*24) + (24-feature['hour'])
        data_fea = pd.merge(data_fea, feature.loc[:, ['user_id', fea_name]], how='left', on='user_id')
        data_fea[fea_name] = data_fea[fea_name].fillna(24*duration).astype(int)
        print('# -- ' + fea_name + ' complete -- #')

    # 用户有几个经纬度
    fea_name ='geo_count&user_id'
    feature = data_all[data_all['user_geo_lat']!=90]
    feature = com.pivot_table_plus(feature, 'user_id', 'user_geo_lat', com.count_with_drop_duplicates_for_series, fea_name)
    data_fea = pd.merge(data_fea, feature, on='user_id', how='left')
    data_fea[fea_name] = data_fea[fea_name].fillna(0).astype(int)
    print('# -- ' + fea_name + ' complete -- #')

    if save is True:
        save_name = get_save_name(label_day_rank, duration, p_only, index='user')
        com.save_csv(data_fea, com.get_project_path(FEATURE_PATH), save_name)
        return com.get_project_path(FEATURE_PATH) + save_name
    else:
        return data_fea


'''
提取商品特征
'''
def get_item_feature(data_all, data_p, label_day_rank, duration=7, p_only=True, data_item=None, save=False):
    data_all = data_all[(data_all['day_rank']>=label_day_rank-duration) & (data_all['day_rank']<=label_day_rank-1)]
    data_p = data_p[(data_p['day_rank']>=label_day_rank-duration) & (data_p['day_rank']<=label_day_rank-1)]

    if p_only is True:
        data_fea = data_p[(data_p['day_rank']>=label_day_rank-SET_LENGTH) & (data_p['beh_type']==1)].loc[:, ['item_id', 'item_cate']].drop_duplicates()
    else:
        data_fea = data_all[(data_all['day_rank']>=label_day_rank-SET_LENGTH) & (data_all['beh_type']==1)].loc[:, ['item_id', 'item_cate']].drop_duplicates()

    # 商品在/商品种类在 总/前1天内/前2天内/前3天内浏览/收藏/购物车/购买 的计数
    for item_index in ['item_id', 'item_cate']:
        for duration_time in [duration, 1, 2, 3]:
            for beh_type in [1, 2, 3, 4]:
                fea_name = 'beh_type_'+str(beh_type)+'_count&'+str(item_index)+'&latest_'+str(duration_time)
                feature = com.pivot_table_plus(data_all[(data_all['beh_type']==beh_type) & (data_all['day_rank']>=label_day_rank-duration_time)],
                                               index=item_index, values='user_id', aggfunc='count', new_name=fea_name)
                data_fea = pd.merge(data_fea, feature, on=item_index, how='left')
                data_fea[fea_name] = data_fea[fea_name].fillna(0).astype(int)
                print('# -- ' + fea_name + ' complete -- #')

    # 商品的/商品种类的 购买/收藏 的计数在 全集/子集 中的 正/反 排序
    for data_index in ['data_all', 'data_p']:
        for item_index in ['item_id', 'item_cate']:
            for beh_type in [2, 4]:
                for ascending in [0, 1]:
                    if data_index == 'data_all': data = data_all
                    else: data = data_p
                    fea_name = 'count_rank'+str(ascending)+'&'+str(item_index)+'&beh_type_'+str(beh_type)+'&'+str(data_index)
                    feature = com.pivot_table_plus(data[(data['beh_type']==beh_type)], index=item_index, values='user_id', aggfunc='count', new_name='tmp')
                    data_fea = pd.merge(data_fea, feature.loc[:, [item_index, 'tmp']], on=item_index, how='left')
                    data_fea['tmp'] = data_fea['tmp'].fillna(0)
                    data_fea[fea_name] = data_fea['tmp'].rank(ascending=ascending, method='dense')
                    print('# -- ' + fea_name + ' complete -- #')
                    del data_fea['tmp']

    # 商品/商品类型 被多少人 浏览/收藏/购物车/购买 过
    for item_index in ['item_id', 'item_cate']:
        for beh_type in [1, 2, 3, 4]:
            fea_name = 'user_count&'+item_index+'&'+'beh_type_'+str(beh_type)
            feature = data_all[(data_all[item_index].isin(data_fea[item_index])) & (data_all['beh_type']==beh_type)]
            feature = com.pivot_table_plus(feature, index=item_index, values='user_id', aggfunc=com.count_with_drop_duplicates_for_series, new_name=fea_name)
            data_fea = pd.merge(data_fea, feature, on=item_index, how='left')
            data_fea[fea_name] = data_fea[fea_name].fillna(0).astype(int)
            print('# -- ' + fea_name + ' complete -- #')

    # 商品的/商品类型的 转化率(浏览X购买)
    data_fea['item_id_ctr'] = 1.*data_fea['beh_type_4_count&item_id&latest_'+str(duration)] / data_fea['beh_type_1_count&item_id&latest_'+str(duration)]
    data_fea['item_id_ctr'] = data_fea['item_id_ctr'].fillna(0)
    data_fea['item_cate_ctr'] = 1.*data_fea['beh_type_4_count&item_cate&latest_'+str(duration)] / data_fea['beh_type_1_count&item_cate&latest_'+str(duration)]
    data_fea['item_cate_ctr'] = data_fea['item_cate_ctr'].fillna(0)

    # 商品有几个经纬度
    for item_index in ['item_id', 'item_cate']:
        fea_name ='geo_count&'+item_index
        feature = data_item[data_item['item_geo_lat']!=-90]
        feature = com.pivot_table_plus(feature, item_index, 'item_geo_lat', com.count_with_drop_duplicates_for_series, fea_name)
        data_fea = pd.merge(data_fea, feature, on=item_index, how='left')
        data_fea[fea_name] = data_fea[fea_name].fillna(0).astype(int)

    del data_fea['item_cate']
    if save is True:
        save_name = get_save_name(label_day_rank, duration, p_only, index='item')
        com.save_csv(data_fea, com.get_project_path(FEATURE_PATH), save_name)
        return com.get_project_path(FEATURE_PATH) + save_name
    else:
        return data_fea


'''
提取用户X商品特征
'''
def get_ui_feature(data_all, data_p, label_day_rank, duration=7, p_only=True, data_item=None, save=False):
    data_all = data_all[(data_all['day_rank']>=label_day_rank-duration) & (data_all['day_rank']<=label_day_rank-1)]
    data_p = data_p[(data_p['day_rank']>=label_day_rank-duration) & (data_p['day_rank']<=label_day_rank-1)]

    if p_only is True:
        data_fea = data_p[(data_p['day_rank']>=label_day_rank-SET_LENGTH) & (data_p['beh_type']==1)].loc[:, ['user_id', 'item_id', 'item_cate', 'ui_id', 'uc_id']].drop_duplicates()
    else:
        data_fea = data_all[(data_all['day_rank']>=label_day_rank-SET_LENGTH) & (data_all['beh_type']==1)].loc[:, ['user_id', 'item_id', 'item_cate', 'ui_id', 'uc_id']].drop_duplicates()

    # 用户 前1天当天/前2天当天/前3天当天 购买了/浏览了 几次这个商品
    for ago_time in [1, 2, 3]:
        for beh_type in [1, 2, 3, 4]:
            fea_name = 'beh_type_'+str(beh_type)+'_count&ui_id&'+str(ago_time)+'_day_ago'
            feature = com.pivot_table_plus(data_all[(data_all['beh_type']==beh_type) & (data_all['day_rank']==label_day_rank-ago_time)],
                                           index='ui_id', values='user_id', aggfunc='count', new_name=fea_name)
            data_fea = pd.merge(data_fea, feature, on='ui_id', how='left')
            data_fea[fea_name] = data_fea[fea_name].fillna(0).astype(int)
            print('# -- ' + fea_name + ' complete -- #')

    # 用户是否 收藏/购买 过这个商品
    data_fea['beh_type_4_if&ui_id'] = (data_fea['ui_id'].isin(data_all[data_all['beh_type']==4]['ui_id'])).replace({True: 1, False: 0})
    data_fea['beh_type_2_if&ui_id'] = (data_fea['ui_id'].isin(data_all[data_all['beh_type']==2]['ui_id'])).replace({True: 1, False: 0})

    # 用户与这个商品最后一次交互 是购买1 还是收藏1.5 还是浏览2 还是加购物车4
    fea_name = 'beh_type_?_last&ui_id'
    feature = data_all.copy()
    feature['tmp'] = feature['day_rank']*100 + feature['hour']
    feature['rank'] = feature.groupby('ui_id')['tmp'].rank(ascending=0)
    feature = feature[feature['rank']==1]
    data_fea = pd.merge(data_fea, feature.loc[:, ['ui_id', 'beh_type']], on='ui_id', how='left')
    data_fea[fea_name] = data_fea['beh_type'].replace({1: 2, 2: 1.5, 3: 4, 4: 1}).fillna(0)
    del data_fea['beh_type']

    # 商品在 全集/子集 是否是用户最后的交互对象
    for data_index in ['data_all', 'data_p']:
        if data_index == 'data_all': data = data_all
        else: data = data_p
        fea_name = 'is_last&ui_id&'+data_index
        feature = data.loc[:, ['user_id', 'ui_id', 'day_rank', 'hour']].sort_values(by=['user_id', 'day_rank', 'hour'], ascending=[0, 0, 0]).drop_duplicates('user_id')
        data_fea[fea_name] = (data_fea['ui_id'].isin(feature['ui_id'])).replace({True: 1, False: 0})
        print('# -- ' + fea_name + ' complete -- #')

    # 商品是用户在 全集/子集 倒数第几个交互对象
    for data_index in ['data_all', 'data_p']:
        if data_index == 'data_all': data = data_all
        else: data = data_p
        fea_name = 'last_?&ui_id&'+data_index
        feature = data.loc[:, ['ui_id', 'day_rank', 'hour']]
        feature['tmp'] = feature['day_rank']*100 + feature['hour']
        feature['rank'] = feature.groupby('ui_id')['tmp'].rank(method='dense', ascending=1)
        feature = feature.sort_values(by=['rank', 'ui_id'], ascending=[True, True]).drop_duplicates(['ui_id']).loc[:, ['ui_id', 'rank']]
        data_fea = pd.merge(data_fea, feature, on='ui_id', how='left').fillna(max(feature['rank']+1))
        data_fea = data_fea.rename(columns={'rank': fea_name})
        print('# -- ' + fea_name + ' complete -- #')

    # 用户最后一次 浏览/收藏/购物车/购买 该商品/该商品类型 距标签多少小时
    for beh_type in [1, 2, 3, 4]:
        for id_index in ['ui_id', 'uc_id']:
            fea_name = 'beh_type_'+str(beh_type)+'_latest_to_now_hour&'+id_index
            feature = data_all[(data_all['beh_type']==beh_type)].loc[:, [id_index, 'day_rank', 'hour']].sort_values(by=['day_rank', 'hour'], ascending=[0, 0]).drop_duplicates(id_index)
            feature[fea_name] = feature['day_rank'].apply(lambda a: label_day_rank-label_day_rank)
            feature[fea_name] = (feature[fea_name]*24) + (24-feature['hour'])
            data_fea = pd.merge(data_fea, feature.loc[:, [id_index, fea_name]], how='left', on=id_index)
            data_fea[fea_name] = data_fea[fea_name].fillna(24*duration).astype(int)
            print('# -- ' + fea_name + ' complete -- #')

    del data_fea['uc_id']
    del data_fea['ui_id']
    del data_fea['item_cate']
    if save is True:
        save_name = get_save_name(label_day_rank, duration, p_only, index='ui')
        com.save_csv(data_fea, com.get_project_path(FEATURE_PATH), save_name)
        return com.get_project_path(FEATURE_PATH)+save_name
    else:
        return data_fea


def get_save_name(label_day_rank, duration, p_only, index, set_length=SET_LENGTH):
    save_name = 'fea_'+str(index)+'_label'+str(label_day_rank)+'_dur'+str(duration)+'_sl'+str(set_length)
    if p_only: save_name += '_p'
    save_name += '.csv'
    return save_name


def mean_beh_type_1_count_between_buy(series):
    x = ''.join(list(series)).split('4')
    return 1.*sum(pd.Series(x).apply(lambda a: len(a)))/len(x)


if __name__ == '__main__':
    run()



