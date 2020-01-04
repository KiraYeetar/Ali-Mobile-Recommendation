import pandas as pd
import numpy as np

'''
Precision = True Positive / (True Positive + False Positive)
recall = True Positive / (True Positive + False Negative)
F1 = (2 * Precision * recall) / (Precision + recall)

要求结果表名为：tianchi_mobile_recommendation_predict.csv，
且以utf-8格式编码；包含user_id和item_id两列（均为string类型, 要求去除重复。
'''

def f1_score(y_true, y_pred, if_print=False):
    """阿里移动推荐算法赛题的f1评分方法, 格式要求DataFrame顺序包含user_id和item_id两列,有低概率出现误判情况"""
    try:
        y_true = pd.DataFrame(y_true).drop_duplicates()
        y_pred = pd.DataFrame(y_pred).drop_duplicates()
        y_true = y_true.iloc[:, 0].apply(lambda x: str(x)) + y_true.iloc[:, 1].apply(lambda x: str(x))
        y_pred = y_pred.iloc[:, 0].apply(lambda x: str(x)) + y_pred.iloc[:, 1].apply(lambda x: str(x))
        precision = len(y_pred[y_pred.isin(y_true)]) / len(y_pred) * 1.
        recall = len(y_pred[y_pred.isin(y_true)]) / len(y_true) * 1.
        f1 = 2*precision*recall / (recall+precision)
        if if_print is not False:
            print("f1: %f, precision: %f, recall: %f" % (f1, precision, recall))
        else:
            return f1
    except ZeroDivisionError:
        return 0


def get_csv_label(data, day_rank):
    """获取指定day_rank当天的标签csv"""
    # 筛选只出现在day_rank前的商品
    data = data[data['item_id'].isin(data[data['day_rank']<day_rank]['item_id'])]
    # 筛选指定日期
    data = data[data['day_rank']==day_rank]
    # 筛选beh_type=4的商品
    data = data[data['beh_type']==4]
    # 返回两列
    return data.loc[:, ['user_id', 'item_id']].drop_duplicates()


def get_ui_id(data):
    """用于将user_id于item_id皆视作str进行顺序拼接"""
    return data['user_id'].astype(str) + data['item_id'].astype(str)


def get_uc_id(data):
    """用于将user_id于item_cate皆视作str进行顺序拼接"""
    return data['user_id'].astype(str) + data['item_cate'].astype(str)
