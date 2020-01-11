# -- coding: utf-8 --
from Tools import common as com
from Tools import geohash32 as gh64
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gc


'''
    01   02   03   04   05   06   07 
         18   19   20   21   22   23   
    24   25   26   27   28   29   30
    01   02   03   04   05   06   07
    08   09   10   11   12   13   14
    15   16   17   18  
    
    01   02   03   04   05   06   07 
         01   02   03   04   05   06  
    07   08   09   10   11   12   13
    14   15   16   17   18   19   20
    21   22   23   24   25   26   27
    28   29   30   31  
    
'''

def run():
    csv_data_all = pd.read_csv(com.get_project_path('Data/Csv/OriData/csv_data_all.csv'))
    csv_data_item = pd.read_csv(com.get_project_path('Data/Csv/OriData/tianchi_fresh_comp_train_item.csv'), header=0,
                                names=['item_id', 'item_geo', 'item_cate'])
    csv_data_p = csv_data_all[csv_data_all['item_id'].isin(csv_data_item['item_id'])]


    # 测试代码时解注下面一条
    # csv_data_all = pd.read_csv(com.get_project_path('Data/Csv/OriData/csv_data_all_h1w.csv'))

    # 对time处理下, 根据需求自行注释可节省大量时间
    csv_data_all['time'] = pd.to_datetime(csv_data_all['time'], format='%Y%m%d %H')
    # csv_data_all['hour'] = csv_data_all['time'].dt.hour
    csv_data_all['time'] = csv_data_all['time'].dt.normalize()
    # csv_data_all['week'] = csv_data_all['time'].apply(lambda a: a.weekday()+1)
    # csv_data_all['day'] = csv_data_all['time'].dt.day
    csv_data_all['day_rank'] = csv_data_all['time'].rank(method='dense').apply(lambda a: int(a))

    # ###### 文字分析 ###### #
    '''
    最小日期 2014-11-18 00:00:00
    最大日期 2014-12-18 23:00:00
    '''
    # print(min(csv_data_all['time']))
    # print(max(csv_data_all['time']))

    '''
    总人数 20000
    总商品数 4758484
    '''
    # print(len(set(csv_data_all['user_id'])))
    # print(len(set(csv_data_all['item_id'])))

    '''
    同一商品只有一种类型
    '''
    # print(len(set(csv_data_item['item_id'])))
    # print(len(set(csv_data_item['item_id'].apply(lambda a: str(a)) + csv_data_item['item_cate'].apply(lambda a: str(a)))))

    '''
    同一商品会有多个经纬度
    '''
    # print(len(set(csv_data_item['item_id'])))
    # print(len(set(csv_data_item['item_id'].apply(lambda a: str(a)) + csv_data_item['item_geo'].apply(lambda a: str(a)))))

    '''
    同一用户会有多个经纬度
    '''
    # print(len(set(csv_data_all['user_id'])))
    # print(len(set(csv_data_all['user_id'].apply(lambda a: str(a)) + csv_data_all['user_geo'].apply(lambda a: str(a)))))

    '''
    找到异常用户
    '''
    # csv_user_bh_count = com.pivot_table_plus(csv_data_all, 'user_id', 'item_id', 'count', 'bh_count')
    # csv_user_day_count = com.pivot_table_plus(csv_data_all, 'user_id', 'day_rank',  com.count_with_drop_duplicates_for_series, 'day_count')
    # csv_user_bh_count = pd.merge(csv_user_bh_count, csv_user_day_count, on='user_id', how='left')
    # csv_user_bh_count['bh_count_mean'] = csv_user_bh_count['bh_count'] / csv_user_bh_count['day_count']
    # csv_user_bh_count = csv_user_bh_count.sort_values(by='bh_count_mean', ascending=False).head(15)
    # print(csv_user_bh_count)
    #
    # csv_user_4_count = com.pivot_table_plus(csv_data_all[csv_data_all['user_id'].isin(csv_user_bh_count['user_id']) & (csv_data_all['beh_type']==4)],
    #                                         'user_id', 'item_id', 'count', 'bh4_count')
    # print(csv_user_4_count)


    '''
    经纬度中的字符set{'d', 'n', '4', '3', 'l', 'e', 'j', 'p', 'h', 
    't', '_', 'c', 'm', '5', 'v', '7', 'o', 'k', 's', '9', '0', 
    'g', 'w', 'r', 'u', 'q', '1', 'f', '2', 'a', 'b', 'i', '6'}
    其中'_'来自'input_data_is_error'
    整理下，正常的set为:012345679abcdefghijklmnopqrstuvw，缺少 8 x y z
    '''
    # set_geo = set(list(csv_data_all['item_geo'].dropna())+list(csv_data_all['user_geo'].dropna()))
    # str_geo = str(set_geo).replace('\'', '').replace(',', '').replace(' ', '')[1:-1]
    # print(set(str_geo))

    '''
    销售量 全集/子集
    大于1 的商品有 31235/3010 件
    大于2 的商品有 11759/1090 件
    大于10 的商品有 504/62 件
    大于20 的商品有 107/10 件
    大于50 的商品有 24/1 件
    '''
    # csv_data_cate4 = csv_data_all[csv_data_all['beh_type']==4]
    # csv_data_cate4 = pd.pivot_table(csv_data_cate4, index='item_id', values='user_id', aggfunc='count').reset_index()
    # print(csv_data_cate4[csv_data_cate4['user_id']>1])
    # print(csv_data_cate4[csv_data_cate4['user_id']>2])
    # print(csv_data_cate4[csv_data_cate4['user_id']>10])
    # print(csv_data_cate4[csv_data_cate4['user_id']>20])
    # print(csv_data_cate4[csv_data_cate4['user_id']>50])

    # csv_data_cate4 = csv_data_p[csv_data_p['beh_type']==4]
    # csv_data_cate4 = pd.pivot_table(csv_data_cate4, index='item_id', values='user_id', aggfunc='count').reset_index()
    # print(csv_data_cate4[csv_data_cate4['user_id']>1])
    # print(csv_data_cate4[csv_data_cate4['user_id']>2])
    # print(csv_data_cate4[csv_data_cate4['user_id']>10])
    # print(csv_data_cate4[csv_data_cate4['user_id']>20])
    # print(csv_data_cate4[csv_data_cate4['user_id']>50])


    '''
    有人会同一天多次买多种商品，目测的
    '''

    '''
    总的行为有23291027条
    对于子集商品只有2084859条
    '''
    # print(len(csv_data_all))
    # print(len(csv_data_all[csv_data_all['item_id'].isin(csv_data_item['item_id'])]))

    '''
    全集商品4758484种，分类9557种
    子集商品422858种，分类1054种
    '''
    # print(len(set(csv_data_all['item_id'])))
    # print(len(set(csv_data_all['item_cate'])))
    #
    # print(len(set(csv_data_item['item_id'])))
    # print(len(set(csv_data_item['item_cate'])))


    # ##### 图分析 ###### #
    '''
    销售长达x天的商品数量(两张图)
    # '''
    # csv_item_count_by_sale_day_count = csv_data_all[csv_data_all['beh_type']==4].copy().loc[:, ['day_rank', 'item_id']]
    # csv_item_count_by_sale_day_count = pd.pivot_table(csv_item_count_by_sale_day_count, index='item_id', values='day_rank',
    #                                                   aggfunc=com.count_with_drop_duplicates_for_series).reset_index().rename(columns={'day_rank': 'sale_day_count'})
    # # print(csv_item_count_by_sale_day_count)
    # csv_item_count_by_sale_day_count = pd.pivot_table(csv_item_count_by_sale_day_count, index='sale_day_count', values='item_id',
    #                                                   aggfunc='count').rename(columns={'item_id': 'item_count'}).sort_values(by='item_count', ascending=False)
    # # print(csv_item_count_by_sale_day_count)
    # csv_item_count_by_sale_day_count.plot.bar()
    # plt.xlabel('sale days count')
    # plt.savefig(com.get_project_path('Data/Graph/item_count_by_sale_day_count.jpg'))
    # # plt.show()
    #
    # csv_item_count_by_sale_day_count.tail(21).plot.bar()
    # plt.xlabel('sale days count')
    # plt.savefig(com.get_project_path('Data/Graph/item_count_by_sale_day_count_t21.jpg'))
    # # plt.show()
    # gc.collect()

    '''
    记录长达x天的商品数量(两张图)
    '''
    # csv_item_count_by_log_day_count = csv_data_all.copy().loc[:, ['day_rank', 'item_id']]
    # csv_item_count_by_log_day_count = pd.pivot_table(csv_item_count_by_log_day_count, index='item_id', values='day_rank',
    #                                                  aggfunc=com.count_with_drop_duplicates_for_series).reset_index().rename(columns={'day_rank': 'log_day_count'})
    # # print(csv_item_count_by_log_day_count)
    # csv_item_count_by_log_day_count = pd.pivot_table(csv_item_count_by_log_day_count, index='log_day_count', values='item_id',
    #                                                  aggfunc='count').rename(columns={'item_id': 'item_count'}).sort_values(by='item_count', ascending=False)
    # # print(csv_item_count_by_log_day_count)
    # csv_item_count_by_log_day_count.plot.bar()
    # plt.xlabel('log days count')
    # plt.savefig(com.get_project_path('Data/Graph/item_count_by_log_day_count.jpg'))
    # # plt.show()
    #
    # csv_item_count_by_log_day_count.tail(21).plot.bar()
    # plt.xlabel('log days count')
    # plt.savefig(com.get_project_path('Data/Graph/item_count_by_log_day_count_t21.jpg'))
    # # plt.show()
    # gc.collect()

    '''
    购买长达x天的个人数量
    '''
    # csv_user_count_by_sale_day_count = csv_data_all[csv_data_all['beh_type']==4].copy().loc[:, ['day_rank', 'user_id']]
    # csv_user_count_by_sale_day_count = pd.pivot_table(csv_user_count_by_sale_day_count, index='user_id', values='day_rank',
    #                                                   aggfunc=com.count_with_drop_duplicates_for_series).reset_index().rename(columns={'day_rank': 'sale_day_count'})
    # # print(csv_user_count_by_sale_day_count)
    # csv_user_count_by_sale_day_count = pd.pivot_table(csv_user_count_by_sale_day_count, index='sale_day_count', values='user_id',
    #                                                   aggfunc='count').rename(columns={'user_id': 'user_count'}).sort_values(by='user_count', ascending=False)
    # # print(csv_user_count_by_sale_day_count)
    # csv_user_count_by_sale_day_count.plot.bar()
    # plt.xlabel('sale days count')
    # plt.savefig(com.get_project_path('Data/Graph/user_count_by_sale_day_count.jpg'))
    # # plt.show()
    # gc.collect()

    '''
    记录长达x天的个人数量
    '''
    # csv_user_count_by_log_day_count = csv_data_all.copy().loc[:, ['day_rank', 'user_id']]
    # csv_user_count_by_log_day_count = pd.pivot_table(csv_user_count_by_log_day_count, index='user_id', values='day_rank',
    #                                                  aggfunc=com.count_with_drop_duplicates_for_series).reset_index().rename(columns={'day_rank': 'log_day_count'})
    # # print(csv_user_count_by_log_day_count)
    # csv_user_count_by_log_day_count = pd.pivot_table(csv_user_count_by_log_day_count, index='log_day_count', values='user_id',
    #                                                  aggfunc='count').rename(columns={'user_id': 'user_count'}).sort_values(by='user_count', ascending=False)
    # # print(csv_user_count_by_log_day_count)
    # csv_user_count_by_log_day_count.plot.bar()
    # plt.xlabel('log days count')
    # plt.savefig(com.get_project_path('Data/Graph/user_count_by_log_day_count.jpg'))
    # # plt.show()
    # gc.collect()

    '''
    商品全集/子集 销售前100计数
    '''
    # csv_item_sale_by_user = csv_data_all[(csv_data_all['beh_type']==4) & csv_data_all['item_id'].isin(csv_data_item['item_id'])].copy().loc[:, ['user_id', 'item_id']]
    # csv_item_sale_by_user = pd.pivot_table(csv_item_sale_by_user, index='item_id', values='user_id', aggfunc='count').rename(columns={'user_id': 'item_sale'}).sort_values(by='item_sale', ascending=False).head(100)
    #
    # csv_item_sale_by_user.plot.bar()
    # plt.xlabel('items')
    # plt.xticks(np.arange(0, 101, 10), np.arange(0, 101, 10))
    # plt.savefig(com.get_project_path('Data/Graph/item_sale_by_item_P.jpg'))
    # plt.show()
    # gc.collect()

    # csv_item_sale_by_user = csv_data_all[csv_data_all['beh_type']==4].copy().loc[:, ['user_id', 'item_id']]
    # csv_item_sale_by_user = pd.pivot_table(csv_item_sale_by_user, index='item_id', values='user_id', aggfunc='count').rename(columns={'user_id': 'item_sale'}).sort_values(by='item_sale', ascending=False).head(100)
    #
    # csv_item_sale_by_user.plot.bar()
    # plt.xlabel('items')
    # plt.xticks(np.arange(0, 101, 10), np.arange(0, 101, 10))
    # plt.savefig(com.get_project_path('Data/Graph/item_sale_by_item.jpg'))
    # # plt.show()
    # gc.collect()

    '''
    商品记录计数
    '''
    # csv_item_log_by_user = csv_data_all.copy().loc[:, ['user_id', 'item_id']]
    # csv_item_log_by_user = pd.pivot_table(csv_item_log_by_user, index='item_id', values='user_id',
    #                                       aggfunc='count').rename(columns={'user_id': 'item_log'}).sort_values(by='item_log', ascending=False).head(100)
    #
    # csv_item_log_by_user.plot.bar()
    # plt.xlabel('items')
    # plt.xticks(np.arange(0, 101, 10), np.arange(0, 101, 10))
    # plt.savefig(com.get_project_path('Data/Graph/item_log_by_item.jpg'))
    # # plt.show()
    # gc.collect()

    '''
    个人购买计数
    '''
    # csv_item_sale_by_user = csv_data_all[csv_data_all['beh_type']==4].copy().loc[:, ['user_id', 'item_id']]
    # csv_item_sale_by_user = pd.pivot_table(csv_item_sale_by_user, index='user_id', values='item_id',
    #                                        aggfunc='count').rename(columns={'item_id': 'item_sale'}).sort_values(by='item_sale', ascending=False)
    #
    # csv_item_sale_by_user.plot.bar()
    # plt.xticks([])
    # plt.xlabel('users')
    # plt.xticks(np.arange(0, 20001, 1000), np.arange(0, 20001, 1000), rotation=60)
    # plt.savefig(com.get_project_path('Data/Graph/item_sale_by_user.jpg'))
    # # plt.show()
    # gc.collect()

    '''
    个人记录计数
    '''
    # csv_item_log_by_user = csv_data_all.copy().loc[:, ['user_id', 'item_id']]
    # csv_item_log_by_user = pd.pivot_table(csv_item_log_by_user, index='user_id', values='item_id',
    #                                       aggfunc='count').rename(columns={'item_id': 'item_log'}).sort_values(by='item_log', ascending=False)
    #
    # csv_item_log_by_user.plot.bar()
    # plt.xticks([])
    # plt.xlabel('users')
    # plt.xticks(np.arange(0, 20001, 1000), np.arange(0, 20001, 1000), rotation=60)
    # plt.savefig(com.get_project_path('Data/Graph/item_log_by_user.jpg'))
    # # plt.show()
    # gc.collect()

    '''
    商品种类记录当天的增加占比和减少减少
    eg: day1[a, b, c], day2[b, c, d, e]
    increase_rate_of_log_count = [d, e] / [b, c, d, e] = 1/2
    decrement_rate_of_log_count = [a] / [a, b, c] = 1/3
    '''
    # csv_data_all_copy = csv_data_all.copy().loc[:, ['day_rank', 'item_id']]
    # csv_item_count_by_day_rank = pd.pivot_table(csv_data_all_copy, index='day_rank', values='item_id',
    #                                             aggfunc=com.count_with_drop_duplicates_for_series).reset_index()
    # csv_item_count_by_day_rank['increase_rate_of_log_count'] = [np.nan] + [
    #     len(com.get_difference_for_series(csv_data_all_copy[csv_data_all_copy['day_rank'] == day_rank]['item_id'],
    #                                       csv_data_all_copy[csv_data_all_copy['day_rank'] == (day_rank - 1)][
    #                                           'item_id'])) /
    #     len(csv_data_all_copy[csv_data_all_copy['day_rank'] == day_rank]) for day_rank in range(2, 32)]
    # csv_item_count_by_day_rank['decrement_rate_of_log_count'] = [np.nan] + [
    #     len(com.get_difference_for_series(csv_data_all_copy[csv_data_all_copy['day_rank'] == day_rank - 1]['item_id'],
    #                                       csv_data_all_copy[csv_data_all_copy['day_rank'] == day_rank]['item_id'])) /
    #     len(csv_data_all_copy[csv_data_all_copy['day_rank'] == day_rank - 1]) for day_rank in range(2, 32)]
    #
    # del csv_item_count_by_day_rank['item_id']
    # csv_item_count_by_day_rank = csv_item_count_by_day_rank.set_index('day_rank')
    #
    # csv_item_count_by_day_rank.plot()
    # plt.xticks(range(1, 32, 2))
    # plt.xlabel('day rank')
    # plt.savefig(com.get_project_path('Data/Graph/item_inc＆dec_rate_by_day_rank.jpg'))
    # # plt.show()
    # gc.collect()

    '''
    去重人数当天的增加占比和减少减少
    '''
    # csv_data_all_copy = csv_data_all.copy().loc[:, ['day_rank', 'user_id']]
    # csv_item_count_by_day_rank = pd.pivot_table(csv_data_all_copy, index='day_rank', values='user_id', aggfunc=com.count_with_drop_duplicates_for_series).reset_index()
    # csv_item_count_by_day_rank['increase_rate_of_user_count'] = [np.nan] + [
    #     len(com.get_difference_for_series(csv_data_all_copy[csv_data_all_copy['day_rank']==day_rank]['user_id'], csv_data_all_copy[csv_data_all_copy['day_rank']==(day_rank-1)]['user_id'])) /
    #     len(csv_data_all_copy[csv_data_all_copy['day_rank'] == day_rank]) for day_rank in range(2, 32)]
    # csv_item_count_by_day_rank['decrement_rate_of_user_count'] = [np.nan] + [
    #     len(com.get_difference_for_series(csv_data_all_copy[csv_data_all_copy['day_rank'] == day_rank-1]['user_id'], csv_data_all_copy[csv_data_all_copy['day_rank'] == day_rank]['user_id'])) /
    #     len(csv_data_all_copy[csv_data_all_copy['day_rank'] == day_rank-1]) for day_rank in range(2, 32)]
    #
    # del csv_item_count_by_day_rank['user_id']
    # csv_item_count_by_day_rank = csv_item_count_by_day_rank.set_index('day_rank')
    #
    # csv_item_count_by_day_rank.plot()
    # plt.xticks(range(1, 32, 2))
    # plt.xlabel('day rank')
    # plt.savefig(com.get_project_path('Data/Graph/user_inc＆dec_rate_by_day_rank.jpg'))
    # # plt.show()
    # gc.collect()

    '''
    五个星期的记录对比图
    '''
    # csv_data_week1 = csv_data_all[csv_data_all['day_rank']<=6]
    # csv_data_week1.loc[:, ['day_rank']] = csv_data_week1['day_rank']+1
    # csv_data_week1 = pd.pivot_table(csv_data_week1, index='day_rank', values='item_id', aggfunc='count')
    #
    # csv_data_week2 = csv_data_all[(csv_data_all['day_rank']>6) & (csv_data_all['day_rank']<=13)]
    # csv_data_week2.loc[:, ['day_rank']] = csv_data_week2['day_rank']-6
    # csv_data_week2 = pd.pivot_table(csv_data_week2, index='day_rank', values='item_id', aggfunc='count')
    #
    # csv_data_week3 = csv_data_all[(csv_data_all['day_rank']>13) & (csv_data_all['day_rank']<=20)]
    # csv_data_week3.loc[:, ['day_rank']] = csv_data_week3['day_rank']-13
    # csv_data_week3 = pd.pivot_table(csv_data_week3, index='day_rank', values='item_id', aggfunc='count')
    #
    # csv_data_week4 = csv_data_all[(csv_data_all['day_rank']>20) & (csv_data_all['day_rank']<=27)]
    # csv_data_week4.loc[:, ['day_rank']] = csv_data_week4['day_rank']-20
    # csv_data_week4 = pd.pivot_table(csv_data_week4, index='day_rank', values='item_id', aggfunc='count')
    #
    # csv_data_week5 = csv_data_all[csv_data_all['day_rank']>27]
    # csv_data_week5.loc[:, ['day_rank']] = csv_data_week5['day_rank']-27
    # csv_data_week5 = pd.pivot_table(csv_data_week5, index='day_rank', values='item_id', aggfunc='count')
    #
    # csv_data_weeks = pd.concat([csv_data_week1, csv_data_week2, csv_data_week3, csv_data_week4, csv_data_week5], axis=1)
    # csv_data_weeks.columns=['week1', 'week2', 'week3', 'week4', 'week5']
    # csv_data_weeks = csv_data_weeks.fillna(np.mean(csv_data_weeks)//2)
    # csv_data_weeks.plot.bar()
    # plt.ylabel('sale count')
    # plt.xlabel('day of the week')
    # plt.savefig(com.get_project_path('Data/Graph/log_count_by_week.jpg'))
    # # plt.show()
    # gc.collect()

    '''
    五个星期的销售对比图
    '''
    # csv_data_all_copy = csv_data_all[csv_data_all['beh_type']==4].copy()
    # csv_data_week1 = csv_data_all_copy[csv_data_all_copy['day_rank']<=6]
    # csv_data_week1.loc[:, ['day_rank']] = csv_data_week1['day_rank']+1
    # csv_data_week1 = pd.pivot_table(csv_data_week1, index='day_rank', values='item_id', aggfunc='count')
    #
    # csv_data_week2 = csv_data_all_copy[(csv_data_all_copy['day_rank']>6) & (csv_data_all_copy['day_rank']<=13)]
    # csv_data_week2.loc[:, ['day_rank']] = csv_data_week2['day_rank']-6
    # csv_data_week2 = pd.pivot_table(csv_data_week2, index='day_rank', values='item_id', aggfunc='count')
    #
    # csv_data_week3 = csv_data_all_copy[(csv_data_all_copy['day_rank']>13) & (csv_data_all_copy['day_rank']<=20)]
    # csv_data_week3.loc[:, ['day_rank']] = csv_data_week3['day_rank']-13
    # csv_data_week3 = pd.pivot_table(csv_data_week3, index='day_rank', values='item_id', aggfunc='count')
    #
    # csv_data_week4 = csv_data_all_copy[(csv_data_all_copy['day_rank']>20) & (csv_data_all_copy['day_rank']<=27)]
    # csv_data_week4.loc[:, ['day_rank']] = csv_data_week4['day_rank']-20
    # csv_data_week4 = pd.pivot_table(csv_data_week4, index='day_rank', values='item_id', aggfunc='count')
    #
    # csv_data_week5 = csv_data_all_copy[csv_data_all_copy['day_rank']>27]
    # csv_data_week5.loc[:, ['day_rank']] = csv_data_week5['day_rank']-27
    # csv_data_week5 = pd.pivot_table(csv_data_week5, index='day_rank', values='item_id', aggfunc='count')
    #
    # csv_data_weeks = pd.concat([csv_data_week1, csv_data_week2, csv_data_week3, csv_data_week4, csv_data_week5], axis=1)
    # csv_data_weeks.columns=['week1', 'week2', 'week3', 'week4', 'week5']
    # csv_data_weeks = csv_data_weeks.fillna(np.mean(csv_data_weeks)//2)
    # csv_data_weeks.plot.bar()
    # plt.ylabel('log count')
    # plt.xlabel('day of the week')
    # plt.savefig(com.get_project_path('Data/Graph/sale_count_by_week.jpg'))
    # # plt.show()
    # gc.collect()

    '''
    按日期排序的销量图
    '''
    # csv_data_all_copy = csv_data_all[csv_data_all['beh_type']==4].copy()
    # csv_data_all_copy = pd.pivot_table(csv_data_all_copy, index='day_rank', values='item_id', aggfunc='count')
    # csv_data_all_copy.plot(color='g', kind='bar')
    # plt.xlabel('day rank')
    # plt.legend(['item sale'])
    # plt.savefig(com.get_project_path('Data/Graph/item_sale_by_day_rank.jpg'))
    # # plt.show()
    # gc.collect()

    '''
    按日期排序的记录量量图
    '''
    # csv_data_all_copy = csv_data_all.copy()
    # csv_data_all_copy = pd.pivot_table(csv_data_all_copy, index='day_rank', values='item_id', aggfunc='count')
    # csv_data_all_copy.plot(color='g', kind='bar')
    # plt.xlabel('day rank')
    # plt.legend(['log count'])
    # plt.savefig(com.get_project_path('Data/Graph/log_count_by_day_rank.jpg'))
    # # plt.show()
    # gc.collect()

    '''
    按日期排序的商品种类图
    '''
    # csv_data_all_copy = csv_data_all.copy()
    # csv_data_all_copy = pd.pivot_table(csv_data_all_copy, index='day_rank', values='item_id', aggfunc=com.count_with_drop_duplicates_for_series)
    # csv_data_all_copy.plot(color='b', kind='bar')
    # plt.xlabel('day rank')
    # plt.legend(['item count'])
    # plt.savefig(com.get_project_path('Data/Graph/item_count_by_day_rank.jpg'))
    # # plt.show()
    # gc.collect()

    '''
    按日期排序的去重人数图
    '''
    # csv_data_all_copy = csv_data_all.copy()
    # csv_data_all_copy = pd.pivot_table(csv_data_all_copy, index='day_rank', values='user_id', aggfunc=com.count_with_drop_duplicates_for_series)
    # csv_data_all_copy.plot(color='b', kind='bar')
    # plt.xlabel('day rank')
    # plt.legend(['user count'])
    # plt.savefig(com.get_project_path('Data/Graph/user_count_by_day_rank.jpg'))
    # # plt.show()
    # gc.collect()

    '''
    只在某一天出现的用户和商品计数 （没优化，跑的很慢）
    '''
    user_len = []
    item_len = []


    for i in range(1, 31):
        csv_user_day_count = com.pivot_table_plus(csv_data_all, 'user_id', 'day_rank', com.count_with_drop_duplicates_for_series, 'day_count')
        csv_user_day_count = csv_user_day_count[csv_user_day_count['day_count'] == 1]
        csv_user_only1212 = csv_data_all[csv_data_all['day_rank'] == i].drop_duplicates('user_id')
        csv_user_only1212 = csv_user_only1212[csv_user_only1212['user_id'].isin(csv_user_day_count['user_id'])]
        user_len += [len(csv_user_only1212)]
    user_len = pd.DataFrame({'day_rank': range(1, 31), 'user_count': user_len}).set_index('day_rank')
    plt.plot(user_len)
    plt.savefig(com.get_project_path('Data/Graph/user_only_one_day_count_by_day_rank.jpg'))
    plt.show()

    # for i in range(20, 31):
    #     csv_item_day_count = com.pivot_table_plus(csv_data_all, 'item_id', 'day_rank', com.count_with_drop_duplicates_for_series, 'day_count')
    #     csv_item_day_count = csv_item_day_count[csv_item_day_count['day_count'] == 1]
    #     csv_item_only1212 = csv_data_all[csv_data_all['day_rank'] == i].drop_duplicates('item_id')
    #     csv_item_only1212 = csv_item_only1212[csv_item_only1212['item_id'].isin(csv_item_day_count['item_id'])]
    #     item_len += [len(csv_item_only1212)]
    # item_len = pd.DataFrame({'day_rank': range(20, 31), 'item_count': item_len}).set_index('day_rank')
    # plt.plot(item_len)
    # plt.savefig(com.get_project_path('Data/Graph/item_only_one_day_count_by_day_rank.jpg'))
    # plt.show()


if __name__ == '__main__':
    run()
