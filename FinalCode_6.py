#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from catboost import Pool, CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import warnings
import datetime
import time
import gc

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()


# In[2]:


# Importation du dataset new_merchant_transactions
df_new_merchant = pd.read_csv("../data_sets/new_merchant_transactions.csv",parse_dates=['purchase_date'])
print(df_new_merchant.shape)
df_new_merchant.head(5)


# In[3]:


# Importation du dataset historical_transactions
df_histo = pd.read_csv("../data_sets/historical_transactions.csv",parse_dates=['purchase_date'])
print(df_histo.shape)
df_histo.head(5)


# In[4]:


# Importation du dataset train
df_train = pd.read_csv('../data_sets/train.csv')
# Importation du dataset test
df_test = pd.read_csv('../data_sets/test.csv')


# In[5]:


# Remplacement des valeurs NaN
# Ces transformations sont pour le historical et le new merchant
for df in [df_histo,df_new_merchant]:
    df['category_2'].fillna(1.0,inplace=True)
    df['category_3'].fillna('A',inplace=True)
    df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)


# In[6]:


# Transformer 'purchase_date' en datetime, et modifier les valeurs de 'authorized_flag' et 'category_1' par 0 et 1
# Calcule du différentiel temporel entre aujourd'hui et la date d'achat du client en mois et lui ajouter le décalage associé
# Ces transformations sont pour le historical et le new merchant
for df in [df_histo,df_new_merchant]:
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})
    df['category_1'] = df['category_1'].map({'Y':1, 'N':0}) 
    #https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/73244
    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30
    df['month_diff'] += df['month_lag']


# In[7]:


df_histo.head(5)


# In[8]:


# Fonction pour le renommage des nouvelles colonnes
def get_new_columns(name,aggs):
    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]


# In[9]:


# Calcule des nouvelles colonnes pour le historical_transactions
# Ces nouvelles colonnes seront ajoutées dans le train et le test
aggs = {}
for col in ['subsector_id','merchant_id','merchant_category_id']:
    aggs[col] = ['nunique']

aggs['purchase_amount'] = ['sum','max','min','mean','var']
aggs['installments'] = ['sum','max','min','mean','var']
aggs['purchase_date'] = ['max','min']
aggs['month_lag'] = ['max','min','mean','var']
aggs['month_diff'] = ['mean']
aggs['authorized_flag'] = ['sum', 'mean']
aggs['category_1'] = ['sum', 'mean']
aggs['card_id'] = ['size']

for col in ['category_2','category_3']:
    df_histo[col+'_mean'] = df_histo.groupby([col])['purchase_amount'].transform('mean')
    aggs[col+'_mean'] = ['mean']    

new_columns = get_new_columns('hist',aggs)
print (new_columns)
df_hist_trans_group = df_histo.groupby('card_id').agg(aggs)
df_hist_trans_group.columns = new_columns
df_hist_trans_group.reset_index(drop=False,inplace=True)
df_hist_trans_group['hist_purchase_date_diff'] = (df_hist_trans_group['hist_purchase_date_max'] - df_hist_trans_group['hist_purchase_date_min']).dt.days
df_hist_trans_group['hist_purchase_date_average'] = df_hist_trans_group['hist_purchase_date_diff']/df_hist_trans_group['hist_card_id_size']
df_hist_trans_group['hist_purchase_date_uptonow'] = (datetime.datetime.today() - df_hist_trans_group['hist_purchase_date_max']).dt.days
df_train = df_train.merge(df_hist_trans_group,on='card_id',how='left')
df_test = df_test.merge(df_hist_trans_group,on='card_id',how='left')
del df_hist_trans_group;gc.collect()


# In[10]:


# Calcule des nouvelles colonnes pour le new_merchant_transactions
# Ces nouvelles colonnes seront ajoutées dans le train et le test
aggs = {}
for col in ['subsector_id','merchant_id','merchant_category_id']:
    aggs[col] = ['nunique']
aggs['purchase_amount'] = ['sum','max','min','mean','var']
aggs['installments'] = ['sum','max','min','mean','var']
aggs['purchase_date'] = ['max','min']
aggs['month_lag'] = ['max','min','mean','var']
aggs['month_diff'] = ['mean']
aggs['category_1'] = ['sum', 'mean']
aggs['card_id'] = ['size']

for col in ['category_2','category_3']:
    df_new_merchant[col+'_mean'] = df_new_merchant.groupby([col])['purchase_amount'].transform('mean')
    aggs[col+'_mean'] = ['mean']
    
new_columns = get_new_columns('new_merchant',aggs)
df_hist_trans_group = df_new_merchant.groupby('card_id').agg(aggs)
df_hist_trans_group.columns = new_columns
print (new_columns)
df_hist_trans_group.reset_index(drop=False,inplace=True)
df_hist_trans_group['new_merchant_purchase_date_diff'] = (df_hist_trans_group['new_merchant_purchase_date_max'] - df_hist_trans_group['new_merchant_purchase_date_min']).dt.days
df_hist_trans_group['new_merchant_purchase_date_average'] = df_hist_trans_group['new_merchant_purchase_date_diff']/df_hist_trans_group['new_merchant_card_id_size']
df_hist_trans_group['new_merchant_purchase_date_uptonow'] = (datetime.datetime.today() - df_hist_trans_group['new_merchant_purchase_date_max']).dt.days
df_train = df_train.merge(df_hist_trans_group,on='card_id',how='left')
df_test = df_test.merge(df_hist_trans_group,on='card_id',how='left')
del df_hist_trans_group;gc.collect()


# In[11]:


# Libérer de l'espace avec le garbage collector
del df_histo;gc.collect()
del df_new_merchant;gc.collect()
df_train.head(5)


# In[12]:


# Calcule de la nouvelle colonne 'outliers' pour le train
df_train['outliers'] = 0
df_train.loc[df_train['target'] < -30, 'outliers'] = 1
df_train['outliers'].value_counts()


# In[13]:


# Transformer 'first_active_month' en datetime, et calcule du temps passé entre aujourd'hui et la date du 1er achat
# Calcule du nombre de fois qu'une 'card_id' a été utilisé dans le historical et le new merchant
for df in [df_train,df_test]:
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days
    
    df['card_id_total'] = df['new_merchant_card_id_size']+df['hist_card_id_size']
    df['purchase_amount_total'] = df['new_merchant_purchase_amount_sum']+df['hist_purchase_amount_sum']

# Calcule des nouveaux features qui correspondent au mean par outliers par rapport aux features
# Ces transformations sont pour le train et le test
for f in ['feature_1','feature_2','feature_3']:
    order_label = df_train.groupby([f])['outliers'].mean()
    print(f,"order_label",order_label)
    df_train[f] = df_train[f].map(order_label)
    df_test[f] = df_test[f].map(order_label)


# In[14]:


print(df_train.shape)
df_train.head(5)


# In[15]:


# Retirer la colonnes target du train
df_train_columns = [c for c in df_train.columns if c not in ['card_id', 'first_active_month','target','outliers']]
print(df_train_columns)
target = df_train['target']
del df_train['target']


# In[16]:


print(df_test.shape)
df_test.head(5)


# In[17]:


df_train.columns.difference(df_test.columns)


# In[18]:


# Mettre le 'card_id' comme index pour le train
df_train = df_train.set_index("card_id")
df_train.head(5)


# In[19]:


# Mettre le 'card_id' comme index pour le test
df_test = df_test.set_index("card_id")
df_test.head(5)


# ## Nettoyage des données

# In[20]:


# Retirer les colonnes inutiles du train
listOfOuts = ["new_merchant_purchase_date_max","new_merchant_purchase_date_min",
              "new_merchant_merchant_category_id_nunique",
              "new_merchant_merchant_id_nunique","hist_authorized_flag_mean","hist_authorized_flag_sum",
              "hist_purchase_date_max","hist_purchase_date_min","hist_merchant_id_nunique",
              "hist_merchant_category_id_nunique"]

df_train = df_train.drop(listOfOuts, axis=1)
df_train = df_train.drop("outliers", axis=1)
df_train.head(5)


# In[21]:


df_train.shape


# In[22]:


# Affichage entier des colonnes
pd.set_option('display.max_columns', 100)
df_train.head(1)


# In[23]:


df_train.columns


# In[24]:


# Retrouver l'index d'une colonne
#df_train.columns.get_loc("new_merchant_purchase_date_max")


# In[25]:


# Retirer les colonnes inutiles du test
listOfOuts = ["new_merchant_purchase_date_max","new_merchant_purchase_date_min",
              "new_merchant_merchant_category_id_nunique",
              "new_merchant_merchant_id_nunique","hist_authorized_flag_mean","hist_authorized_flag_sum",
              "hist_purchase_date_max","hist_purchase_date_min","hist_merchant_id_nunique",
              "hist_merchant_category_id_nunique"]



df_test = df_test.drop(listOfOuts, axis=1)
df_test.head(3)


# In[26]:


df_test.shape


# In[27]:


# Vérification des valeurs manquantes pour le train
df_train.isnull().values.any()


# In[28]:


null_columns=df_train.columns[df_train.isnull().any()]
df_train[null_columns].isnull().sum()


# In[29]:


# Remplacer les valeur manquantes par 0
df_train = df_train.fillna(0)
df_train.isnull().values.any()


# In[30]:


df_train.shape


# In[31]:


# Verification des valeurs manquantes pour le test
df_test.isnull().values.any()


# In[32]:


null_columns=df_test.columns[df_test.isnull().any()]
df_test[null_columns].isnull().sum()


# In[ ]:





# In[33]:


# Ajouter une date pour pallier le manque
df_test["first_active_month"] = df_test["first_active_month"].fillna("2018-02-01")


# In[34]:


null_columns=df_test.columns[df_test.isnull().any()]
df_test[null_columns].isnull().sum()


# In[35]:


# Remplacer les valeur manquantes par 0
df_test = df_test.fillna(0)
df_test.isnull().values.any()


# In[36]:


df_test.shape


# In[37]:


# Transformer le 'first_active_month' en string pour l'utiliser avec notre model
df_train["first_active_month"] = df_train["first_active_month"].astype(str)


# In[39]:


df_train.isnull().values.any()


# In[40]:


df_train.dtypes


# # Notre modèle : CatBoostRegressor

# In[41]:


data = df_train
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=789)


# In[42]:


# Initialiser le Pool
train_pool = Pool(X_train, 
                  y_train, 
                  cat_features=[0,4,28])
test_pool = Pool(X_test, 
                 cat_features= [0,4,28])


# In[43]:


# Spécifier les parametres du trainning
model = CatBoostRegressor(iterations=20, 
                          depth=7, 
                          learning_rate=0.3, 
                          loss_function='RMSE')


# In[44]:


model.fit(train_pool)


# In[45]:


preds = model.predict(test_pool)
print(preds)


# In[46]:


print("score train :", model.score(X_train, y_train))
print("score test :", model.score(X_test, y_test))


# In[47]:


model_pred_train = model.predict(X_train)
model_pred_test = model.predict(X_test)

print("mse train:", mean_squared_error(model_pred_train, y_train))
print("mse test:", mean_squared_error(model_pred_test, y_test))


# # Prédictions pour le  Test

# In[48]:


df_test["first_active_month"] = df_test["first_active_month"].astype(str)


# In[49]:


predictions = model.predict(df_test)


# In[50]:


df_test = df_test.reset_index()


# In[51]:


df_test.head(3)


# In[52]:


sub_df = pd.DataFrame({"card_id":df_test["card_id"].values})
sub_df["target"] = predictions

sub_df.to_csv("submission_6.csv", index=False)


# In[53]:


sub_df.head(5)


# In[54]:


# la difference avec la submission 4 (learning_rate=1),est le 5(learning_rate=0.3)

