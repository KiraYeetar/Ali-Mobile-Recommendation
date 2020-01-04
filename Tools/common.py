import pandas as pd
import numpy as np
import os

# 复制项目请修改工程路径
def get_project_path(file_name=''):
    return r'F:\pyCodes/Ali-Mobile-Recommendation/'+file_name


def pivot_table_plus(data, index, values, aggfunc, new_name):
    """pivot_table增强版，自带reset_index和新列改名"""
    return pd.pivot_table(data, index=index, values=values, aggfunc=aggfunc).reset_index().rename(columns={values: new_name})


def save_csv(data, path, file_name):
    """自带提示的to_csv，麻烦的是需要多传一个重复参数"""
    print("# -------    saving     ------- #")
    print("# "+file_name+"                  ")
    data.to_csv(path+file_name, index=None)
    print("# -------   complete    ------- #\n")


def count_nan_for_df(df, axis=0):
    """为df计数多少空值, axis控制行列"""
    return df.isnull().sum(axis=axis)


def count_with_drop_duplicates_for_series(series):
    """为series计算去重元素数量，常用在pd.pivot_table"""
    return len(set(series))


def get_difference_for_series(x, y):
    """求x-y即x与y的差集（x有y没有的元素）, 返回series"""
    return pd.Series(list(set(x).difference(set(y))))


def get_file_list(file_path=''):
    """返回文件夹下的文件名list"""
    file_list = os.listdir(file_path)
    return file_list

