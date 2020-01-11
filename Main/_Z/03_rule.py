# -- coding: utf-8 --
from Tools import common as com
from Tools import special as sp
import pandas as pd
import numpy as np

def run():
    csv_data_all = pd.read_csv(com.get_project_path('Data/Csv/ClnData/csv_data_all.csv'))
    csv_data_p = pd.read_csv(com.get_project_path('Data/Csv/ClnData/csv_data_p.csv'))

    # 31号全部的购物车记录
    # a = get_result_by_rule1(csv_data_p, beh_type=3, day_rank=31)
    # com.save_csv(a, com.get_project_path('Data/Csv/ResData/_Z/beh_type_3&latest_1_day_201912281826.csv'), 'beh_type_3&latest_1_day.csv')

    '''
    以 前一天所有在购物车 的商品交上去
    '''
    # for i in range(20, 30):
    #     print("第"+str(i+1)+"天为标签")
    #     a = get_result_by_rule1(csv_data_all, beh_type=3, day_rank=i)
    #     b = sp.get_csv_label(csv_data_all, i+1)
    #     print(sp.f1_score(b, a))
    #
    #     a = get_result_by_rule1(csv_data_p, beh_type=3, day_rank=i)
    #     b = sp.get_csv_label(csv_data_p, i+1)
    #     print(sp.f1_score(b, a))

    '''
    以 前一天所有浏览过 的商品交上去
    '''
    # for i in range(20, 30):
    #     print("第"+str(i+1)+"天为标签")
    #     a = get_result_by_rule1(csv_data_all, beh_type=1, day_rank=i)
    #     b = sp.get_csv_label(csv_data_all, i+1)
    #     print(len(a), len(b))
    #     print(sp.f1_score(b, a))
    #
    #     a = get_result_by_rule1(csv_data_p, beh_type=1, day_rank=i)
    #     b = sp.get_csv_label(csv_data_p, i+1)
    #     print(len(a), len(b))
    #     print(sp.f1_score(b, a))

    '''
    以 前一天所有在购物车且收藏过 的商品交上去
    '''
    # for i in range(20, 30):
    #     print("\n第"+str(i+1)+"天为标签")
    #     a = get_result_by_rule2(csv_data_all, day_rank=i)
    #     b = sp.get_csv_label(csv_data_all, i+1)
    #     print(len(a), len(b))
    #     print(sp.f1_score(b, a))
    #
    #     a = get_result_by_rule2(csv_data_p, day_rank=i)
    #     b = sp.get_csv_label(csv_data_p, i+1)
    #     print(len(a), len(b))
    #     print(sp.f1_score(b, a))


    '''
    纯马后炮测试， 交 标签日期中，曾出现在前一天的购物车里 的商品
    '''
    # for i in range(20, 30):
    #     print("\n第" + str(i + 1) + "天为标签")
    #     b = sp.get_csv_label(csv_data_all, i + 1)
    #     a = csv_data_all[(csv_data_all['day_rank']==i) & (csv_data_all['beh_type']==3)].loc[:, ['user_id', 'item_id']].drop_duplicates()
    #     a = a[(a['item_id'] + a['user_id']).isin(b['item_id'] + b['user_id'])]
    #     print(len(a), len(b))
    #     print(sp.f1_score(b, a))
    #
    #     b = sp.get_csv_label(csv_data_p, i + 1)
    #     a = csv_data_p[(csv_data_p['day_rank']==i) & (csv_data_p['beh_type']==3)].loc[:, ['user_id', 'item_id']].drop_duplicates()
    #     a = a[(a['item_id'] + a['user_id']).isin(b['item_id'] + b['user_id'])]
    #     print(len(a), len(b))
    #     print(sp.f1_score(b, a))

    '''
    纯马后炮测试， 交 标签日期中，曾在前一天浏览过 的商品
    '''
    # for i in range(20, 30):
    #     print("\n第" + str(i + 1) + "天为标签")
    #     b = sp.get_csv_label(csv_data_all, i + 1)
    #     a = csv_data_all[(csv_data_all['day_rank']==i) & (csv_data_all['beh_type']==1)].loc[:, ['user_id', 'item_id']].drop_duplicates()
    #     a = a[(a['item_id']+a['user_id']).isin(b['item_id']+b['user_id'])]
    #     print(len(a), len(b))
    #     print(sp.f1_score(b, a))
    #
    #     b = sp.get_csv_label(csv_data_p, i + 1)
    #     a = csv_data_p[(csv_data_p['day_rank']==i) & (csv_data_p['beh_type']==1)].loc[:, ['user_id', 'item_id']].drop_duplicates()
    #     a = a[(a['item_id']+a['user_id']).isin(b['item_id']+b['user_id'])]
    #     print(len(a), len(b))
    #     print(sp.f1_score(b, a))

    '''
    纯马后炮测试， 交 标签日期中，曾在前两天浏览过 的商品
    '''
    # for i in range(20, 30):
    #     print("\n第" + str(i + 1) + "天为标签")
    #     b = sp.get_csv_label(csv_data_all, i + 1)
    #     a = csv_data_all[(csv_data_all['day_rank']>=i-1) & (csv_data_all['day_rank']<=i) & (csv_data_all['beh_type']==1)].loc[:, ['user_id', 'item_id']].drop_duplicates()
    #     a = a[(a['item_id']+a['user_id']).isin(b['item_id']+b['user_id'])]
    #     print(len(a), len(b))
    #     sp.f1_score(b, a, if_print=True)
    #
    #     b = sp.get_csv_label(csv_data_p, i + 1)
    #     a = csv_data_p[(csv_data_p['day_rank']>=i-1) & (csv_data_p['day_rank']<=i) & (csv_data_p['beh_type']==1)].loc[:, ['user_id', 'item_id']].drop_duplicates()
    #     a = a[(a['item_id']+a['user_id']).isin(b['item_id']+b['user_id'])]
    #     print(len(a), len(b))
    #     sp.f1_score(b, a, if_print=True)

    '''
    纯马后炮测试， 交 标签日期中，曾在前七天浏览过 的商品
    '''
    for i in range(20, 30):
        print("\n第" + str(i + 1) + "天为标签")
        b = sp.get_csv_label(csv_data_all, i + 1)
        a = csv_data_all[(csv_data_all['day_rank']>=i-6) & (csv_data_all['day_rank']<=i) & (csv_data_all['beh_type']==1)].loc[:, ['user_id', 'item_id']].drop_duplicates()
        a = a[(a['item_id']+a['user_id']).isin(b['item_id']+b['user_id'])]
        print(len(a), len(b))
        sp.f1_score(b, a, if_print=True)

        b = sp.get_csv_label(csv_data_p, i + 1)
        a = csv_data_p[(csv_data_p['day_rank']>=i-6) & (csv_data_p['day_rank']<=i) & (csv_data_p['beh_type']==1)].loc[:, ['user_id', 'item_id']].drop_duplicates()
        a = a[(a['item_id']+a['user_id']).isin(b['item_id']+b['user_id'])]
        print(len(a), len(b))
        sp.f1_score(b, a, if_print=True)

    '''
    纯马后炮测试， 交 标签日期中，曾有过任何记录 的商品
    '''
    for i in range(20, 30):
        print("\n第" + str(i + 1) + "天为标签")
        b = sp.get_csv_label(csv_data_all, i + 1)
        a = csv_data_all[(csv_data_all['day_rank']<=i) & (csv_data_all['beh_type']==1)].loc[:, ['user_id', 'item_id']].drop_duplicates()
        a = a[(a['item_id']+a['user_id']).isin(b['item_id']+b['user_id'])]
        print(len(a), len(b))
        sp.f1_score(b, a, if_print=True)

        b = sp.get_csv_label(csv_data_p, i + 1)
        a = csv_data_p[(csv_data_p['day_rank']<=i) & (csv_data_p['beh_type']==1)].loc[:, ['user_id', 'item_id']].drop_duplicates()
        a = a[(a['item_id']+a['user_id']).isin(b['item_id']+b['user_id'])]
        print(len(a), len(b))
        sp.f1_score(b, a, if_print=True)

    '''
    纯马后炮测试， 交 标签日期中，曾经收藏过 的商品
    '''
    # for i in range(20, 30):
    #     print("\n第" + str(i + 1) + "天为标签")
    #     b = sp.get_csv_label(csv_data_all, i + 1)
    #     a = csv_data_all[(csv_data_all['day_rank']<=i) & (csv_data_all['beh_type']==2)].loc[:, ['user_id', 'item_id']].drop_duplicates()
    #     a = a[(a['item_id'] + a['user_id']).isin(b['item_id'] + b['user_id'])]
    #     print(len(a), len(b))
    #     print(sp.f1_score(b, a))
    #
    #     b = sp.get_csv_label(csv_data_p, i + 1)
    #     a = csv_data_p[(csv_data_p['day_rank']<=i) & (csv_data_p['beh_type']==2)].loc[:, ['user_id', 'item_id']].drop_duplicates()
    #     a = a[(a['item_id'] + a['user_id']).isin(b['item_id'] + b['user_id'])]
    #     print(len(a), len(b))
    #     print(sp.f1_score(b, a))



def get_result_by_rule1(data, beh_type, day_rank):
    """按照指定beh_type和指定day_rank获取一天的标签csv格式的结果文件"""
    data = data[(data['beh_type']==beh_type) & (data['day_rank']==day_rank)]
    return data.loc[:, ['user_id', 'item_id']].drop_duplicates()


def get_result_by_rule2(data, day_rank):
    """按照有过beh_type==2的行为，且beh_type==3和指定day_rank获取一天的标签csv格式的结果文件"""
    data = data[(data['beh_type']==3) & (data['day_rank']==day_rank) & (data['item_id'].isin(data[(data['day_rank']<=day_rank) & (data['beh_type']==2)]['item_id']))]
    return data.loc[:, ['user_id', 'item_id']].drop_duplicates()



if __name__ == '__main__':
    run()
