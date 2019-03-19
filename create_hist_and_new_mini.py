import numpy as np
import pandas as pd


#Récupération des card_id du train
card_id_train = list(pd.read_csv("train.csv", usecols = ["card_id"], squeeze = True))

#Récupération des card_id du test
card_id_test = list(pd.read_csv("test.csv", usecols = ["card_id"], squeeze = True))

#Fonction pour découper les datasets en chunks (sinon trop gros)
def reduction_dataset_card_id_col(in_f, out_f, my_columns, other_columns, my_card_id, size=100000):
    reader = pd.read_csv(in_f, sep=',', chunksize=size)
    df_mini = pd.DataFrame(columns = my_columns)
    for chunk in reader:
        chunk = chunk.drop(other_columns, axis = 1) 
        chunk = chunk[(chunk["card_id"].isin(my_card_id))]
        df_mini = pd.concat([chunk, df_mini], axis = 0)
    df_mini.to_csv(out_f)
	
my_columns = ["card_id", "purchase_amount", "month_lag", "purchase_date", "subsector_id"]
other_columns = ['authorized_flag', 'city_id', 'category_1', 'installments', 'category_3', 'merchant_category_id', 'merchant_id','category_2', 'state_id']


#Fichiers historical_transactions.csv et new_merchant_transactions.csv fournis par Kaggle

#Pour le train
reduction_dataset_card_id_col("historical_transactions.csv", "hist_mini_20190221.csv", my_columns = my_columns, other_columns = other_columns, my_card_id = card_id_train)

eduction_dataset_card_id_col("new_merchant_transactions.csv", "new_mini_20190221.csv", my_columns = my_columns, other_columns = other_columns, my_card_id = card_id_train)

#Pour le test
reduction_dataset_card_id_col("historical_transactions.csv", "hist_mini_test_20190222.csv", my_columns = my_columns, other_columns = other_columns, my_card_id = card_id_test)

reduction_dataset_card_id_col("new_merchant_transactions.csv", "new_mini_test_20190222.csv", my_columns = my_columns, other_columns = other_columns, my_card_id = card_id_test)


	
