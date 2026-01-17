import sqlite3
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os

import datetime as dt
import squarify
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots
from joblib import dump

from Models.processdata import products_category_translation, products, order_items, order_payments, orders


products_eng = products.merge(products_category_translation, how = 'left', on = 'product_category_name' )
df_order_items_products = order_items.merge(products_eng, how='left',on='product_id')
df_order_items_products = df_order_items_products.merge(order_payments, how='left',on='order_id')
df_complete = df_order_items_products.merge(orders, how='left', on='order_id')
df_complete = df_complete.dropna()
df_complete = df_complete.drop_duplicates().reset_index()

# feature engineering
df_complete['order_purchase_timestamp'] = pd.to_datetime(df_complete['order_purchase_timestamp'], format="%Y-%m-%d %H:%M:%S")
df_complete['order_estimated_delivery_date'] = pd.to_datetime(df_complete['order_estimated_delivery_date'], format="%Y-%m-%d %H:%M:%S")
df_complete['estimated_delivery_time'] = df_complete.order_estimated_delivery_date - df_complete.order_purchase_timestamp
df_complete.estimated_delivery_time = list(map(lambda x: int(x.days), df_complete.estimated_delivery_time))
df_complete = df_complete.drop('index', axis=1).dropna()

df_intermediate = df_complete.drop(['order_id',
                                 'order_item_id',
                                 'customer_id',
                                 'shipping_limit_date',
                                 'product_description_lenght',
                                 'product_weight_g',
                                 'product_length_cm',
                                 'product_height_cm',
                                 'product_width_cm',
                                 'order_approved_at',
                                 'order_delivered_carrier_date',
                                 'order_delivered_customer_date',
                                 'order_estimated_delivery_date',
                                 'order_status'], axis=1)

df_predicter = df_intermediate.groupby('order_purchase_timestamp', as_index=False).sum()
df_predicter = df_predicter.dropna()

# Creating week and month information
oldest = min(df_predicter.order_purchase_timestamp)
df_predicter['month'] = list(map(lambda x: int(((x - oldest) // 30).days), df_predicter.order_purchase_timestamp))
df_predicter['week'] = list(map(lambda x: int(((x - oldest) // 7).days), df_predicter.order_purchase_timestamp))

df_predicter = df_predicter.drop('order_purchase_timestamp', axis=1)


X= df_predicter[['price', 'freight_value', 'product_name_lenght', 'product_photos_qty', 'estimated_delivery_time']]
y = df_predicter['payment_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)


regr = LinearRegression()

# Fit the model to the training data
regr.fit(X_train, y_train)

# Use the model to make predictions on the testing data
y_pred = regr.predict(X_test)

# save model
dump(regr, 'linear_regression_model.joblib')