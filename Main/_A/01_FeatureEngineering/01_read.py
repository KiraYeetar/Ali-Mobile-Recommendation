from Tools import common as com
import pandas as pd

def run():
   csv_data_item = pd.read_csv(com.get_project_path('Data/Csv/OriData/tianchi_fresh_comp_train_item.csv'), header=0, names=['item_id', 'item_geo', 'item_cate'])
   csv_data_user = pd.read_csv(com.get_project_path('Data/Csv/OriData/tianchi_fresh_comp_train_user.csv'), header=0, names=['user_id', 'item_id', 'beh_type', 'user_geo', 'item_cate', 'time'])

   # 多此一举
   csv_data_all = pd.merge(csv_data_user, csv_data_item.loc[:, ['item_id']].drop_duplicates(), how='left', on='item_id')
   csv_data_all.to_csv(com.get_project_path('Data/Csv/OriData/csv_data_all.csv'), index=None)

   # 保存1w条做来测试代码
   csv_data_all.head(10000).to_csv(com.get_project_path('Data/Csv/OriData/csv_data_all_h1w.csv'), index=None)


if __name__ == '__main__':
    run()
