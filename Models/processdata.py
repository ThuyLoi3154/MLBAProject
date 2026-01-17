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
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go

conn = sqlite3.connect('D:\\project\\dataset\\brazil_ecommerce.db')

#lấy dữ liệu thành các df
customers = pd.read_sql_query("SELECT * FROM customers", conn)
geolocation = pd.read_sql_query('select * from geolocation', conn)
order_items = pd.read_sql_query('select * from order_items', conn)
order_payments = pd.read_sql_query('select * from order_payments', conn)
order_reviews = pd.read_sql_query('select * from order_reviews', conn)
orders = pd.read_sql_query('select * from orders', conn)
products = pd.read_sql_query('select * from products', conn)
sellers = pd.read_sql_query('select * from sellers', conn)
products_category_translation = pd.read_sql_query('select * from product_category_name_translation', conn)
# Đóng kết nối
conn.close()

#review table
# drop the review_comment_title column
order_reviews.drop(['review_comment_title'], axis=1, inplace=True)
# replace missing review messages with string 'NONE'
order_reviews['review_comment_message'] = order_reviews['review_comment_message'].fillna('NONE')

#order table
orders["order_approved_at"] = orders["order_approved_at"].fillna(orders["order_purchase_timestamp"])
orders["order_delivered_carrier_date"] = orders["order_delivered_carrier_date"].fillna(orders["order_approved_at"])
orders["order_delivered_customer_date"] = orders["order_delivered_customer_date"].fillna(orders["order_estimated_delivery_date"])

#product table
products['product_description_lenght'].fillna(products['product_description_lenght'].median,inplace=True)
products['product_photos_qty'].fillna(products['product_photos_qty'].median(), inplace=True)
products['product_weight_g'].fillna(products['product_weight_g'].median(), inplace=True)
products['product_length_cm'].fillna(products['product_length_cm'].median(), inplace=True)
products['product_height_cm'].fillna(products['product_height_cm'].median(), inplace=True)
products['product_width_cm'].fillna(products['product_width_cm'].median(), inplace=True)

merged_df= pd.merge(customers, orders, on="customer_id")
merged_df= merged_df.merge(order_reviews, on="order_id")
merged_df= merged_df.merge(order_items, on="order_id")
merged_df= merged_df.merge(products, on="product_id")
merged_df= merged_df.merge(order_payments, on="order_id")
merged_df= merged_df.merge(sellers, on='seller_id')
merged_df= merged_df.merge(products_category_translation, on='product_category_name')

time_columns= ['order_purchase_timestamp', 'order_approved_at','order_delivered_carrier_date','order_delivered_customer_date',
               'order_estimated_delivery_date', 'review_creation_date', 'review_answer_timestamp', 'shipping_limit_date']
merged_df[time_columns]=merged_df[time_columns].apply(pd.to_datetime)