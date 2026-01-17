import sqlite3
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os
import re
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame

import datetime as dt
import squarify
import numpy as np

from matplotlib.gridspec import GridSpec
pd.set_option('display.max_columns', 100)
import plotly.offline as py
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
#for the millions format function
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as tkr
import matplotlib as mpl
from datetime import datetime, timedelta
from pandas import DataFrame
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go

from Models.processdata import customers, geolocation, order_items, order_payments, order_reviews, orders, products, sellers, products_category_translation

def thousand_count_y(y, pos):
    return '{:.0f} K'.format(y*1e-3)
formatter_thousand_count_y = FuncFormatter(thousand_count_y)
#ax2.yaxis.set_major_formatter(formatter_thousand_count_y)

def millons_count_y(y, pos):
    return '{:.0f} M'.format(y*1e-6)
formatter_millons_count_y = FuncFormatter(millons_count_y)
#ax2.yaxis.set_major_formatter(formatter_millons_count_y)

def thousand_count_x(x, pos):
    return '{:.0f} K'.format(x*1e-3)
formatter_thousand_count_x = FuncFormatter(thousand_count_x)
#ax2.yaxis.set_major_formatter(formatter_thousand_count_x)

def millons_count_x(x, pos):
    return '{:.0f} M'.format(x*1e-6)
formatter_millons_count_x = FuncFormatter(millons_count_x)
#ax2.yaxis.set_major_formatter(formatter_millons_count_x)

def thousand_real_y(y, pos):
    return 'R${:.0f} K'.format(y*1e-3)
formatter_thousand_real_y = FuncFormatter(thousand_real_y)
#ax2.yaxis.set_major_formatter(formatter_thousand_real_y)

def millons_real_y(y, pos):
    return 'R${:.1f} M'.format(y*1e-6)
formatter_millons_real_y = FuncFormatter(millons_real_y)
#ax2.yaxis.set_major_formatter(formatter_millons_real_y)

def thousand_real_x(x, pos):
    return 'R${:.0f} K'.format(x*1e-3)
formatter_thousand_real_x = FuncFormatter(thousand_real_x)
#ax2.yaxis.set_major_formatter(formatter_thousand_real_x)

def millons_real_x(x, pos):
    return 'R${:.1f} M'.format(x*1e-6)
formatter_millons_real_x = FuncFormatter(millons_real_x)
#ax2.yaxis.set_major_formatter(formatter_millons_real_x)

timestamp_cols = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 
                  'order_estimated_delivery_date']
for col in timestamp_cols:
    orders[col] = pd.to_datetime(orders[col])

# Extracting attributes for purchase date - Year and Month
orders['order_purchase_year'] = orders['order_purchase_timestamp'].apply(lambda x: x.year)
orders['order_purchase_month'] = orders['order_purchase_timestamp'].apply(lambda x: x.month)
orders['order_purchase_month_name'] = orders['order_purchase_timestamp'].apply(lambda x: x.strftime('%b'))
orders['order_purchase_year_month'] = orders['order_purchase_timestamp'].apply(lambda x: x.strftime('%Y%m'))
orders['order_purchase_date'] = orders['order_purchase_timestamp'].apply(lambda x: x.strftime('%Y%m%d'))

# Extracting attributes for purchase date - Day and Day of Week
orders['order_purchase_day'] = orders['order_purchase_timestamp'].apply(lambda x: x.day)
orders['order_purchase_dayofweek'] = orders['order_purchase_timestamp'].apply(lambda x: x.dayofweek)
orders['order_purchase_dayofweek_name'] = orders['order_purchase_timestamp'].apply(lambda x: x.strftime('%a'))

# Extracting attributes for purchase date - Hour and Time of the Day
orders['order_purchase_hour'] = orders['order_purchase_timestamp'].apply(lambda x: x.hour)
hours_bins = [-0.1, 6, 12, 18, 23]
hours_labels = ['Dawn', 'Morning', 'Afternoon', 'Night']
orders['order_purchase_time_day'] = pd.cut(orders['order_purchase_hour'], hours_bins, labels=hours_labels)

# Delete 2016 data
orders.drop(orders.loc[orders["order_purchase_year"]==2016].index, inplace=True)

customers = customers.drop_duplicates(subset=["customer_unique_id"])
customers["customer_city"] = customers["customer_city"].str.capitalize()

#The 10 cities with the most clients
clients_by_city = customers.groupby("customer_city").count()["customer_unique_id"].reset_index().sort_values(by="customer_unique_id",ascending=False).head(10)
clients_by_city.rename(columns = {"customer_unique_id":"total"}, inplace=True)

#The 10 states with the most clients
clients_by_state = customers.groupby(["customer_state"]).count()["customer_unique_id"].reset_index().sort_values(by="customer_unique_id",ascending=False).head(10)
clients_by_state.rename(columns = {"customer_unique_id":"total"}, inplace=True)

# merge elements of the data frame (customer, purchase date) to know the trend of how many customers made their first purchase
clients_x_date = pd.merge(customers, orders, on = "customer_id")

sellers["seller_city"] = sellers["seller_city"].str.capitalize()

#The 10 cities with the most sellers
sellers_by_city = sellers.groupby("seller_city").count()["seller_id"].reset_index().sort_values(by="seller_id",ascending=False).head(10)
sellers_by_city.rename(columns = {"seller_id":"total"}, inplace=True)

#The 10 states with the most sellers
sellers_by_states = sellers.groupby("seller_state").count()["seller_id"].reset_index().sort_values(by="seller_id",ascending=False).head(10)
sellers_by_states.rename(columns = {"seller_id":"total"}, inplace=True)

# merge elements of the data frame (customer, purchase date) to know the trend of how many customers made their first purchase
sellers_x_date = pd.merge(order_items,orders, on = "order_id")
sellers_x_date = sellers_x_date.drop_duplicates(subset=["seller_id"])

# CUSTOMERS AND SELLERS
figCS = plt.figure(constrained_layout=True, figsize=(15, 12))

# Axis definition
gs = GridSpec(5, 2, figure=figCS)
ax1 = figCS.add_subplot(gs[0, 0])
ax2 = figCS.add_subplot(gs[0, 1])
ax3 = figCS.add_subplot(gs[1, :])
#ax4 = fig.add_subplot(gs[1, 1])
ax5 = figCS.add_subplot(gs[2, 0])
ax6 = figCS.add_subplot(gs[2, 1])
#ax7 = fig.add_subplot(gs[3, 1])
ax8 = figCS.add_subplot(gs[3, :])

#Customer city
sns.barplot(x="total", y="customer_city", data=clients_by_city, ax=ax1, palette='viridis')
ax1.set_title("The 10 cities with the most clients", size=10, color='black')
ax1.set_xlabel("")
ax1.set_ylabel("")
for rect in ax1.patches:
    ax1.annotate('{:,.0f}'.format(rect.get_width()),(rect.get_width(),rect.get_y() + rect.get_height() / 2),
                xytext=(0, 0),textcoords='offset points', va='center', ha='left', size=12)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.axes.get_xaxis().set_visible(False) 

#Customer states
sns.barplot(x="total", y='customer_state', data=clients_by_state, ax=ax2, palette="YlGnBu")
ax2.set_title("The 10 states with the most clients", size=10, color='black')
ax2.set_xlabel("")
ax2.set_ylabel("")
for rect in ax2.patches:
    ax2.annotate('{:,.0f}'.format(rect.get_width()),(rect.get_width(),rect.get_y() + rect.get_height() / 2),
                xytext=(0, 0),textcoords='offset points', va='center', ha='left', size=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.axes.get_xaxis().set_visible(False)

#Customer per year
sns.lineplot(x="order_purchase_year_month", y="order_id", data=clients_x_date.groupby("order_purchase_year_month").agg({"order_id" : "count"}).reset_index(),ax=ax3, alpha=0.8,
             color='darkslateblue', linewidth=1, marker='o', markersize=3)
sns.barplot(x="order_purchase_year_month", y="order_id", data=clients_x_date.groupby("order_purchase_year_month").agg({"order_id" : "count"}).reset_index(),ax=ax3, alpha=0.1)
ax3.set_title("Customer Evolution", size=10, color="black")
ax3.set_xlabel("")
ax3.set_ylabel("")
ax3.set_ylim(0,9000)
#plt.setp(ax3.get_xticklabels(), rotation=45)
for p in ax3.patches:
        ax3.annotate('{:,.0f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),
                    ha="center", va="top", xytext=(0, 15), textcoords="offset points", 
                    color= "black", size=12)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.spines["left"].set_visible(False)
ax3.set_yticklabels([])
ax3.set_yticks([])

#Total de Customers
ax3.text(-1.5, 8000, "93,447", fontsize=13, ha='center', color="navy")
ax3.text(-1.5, 7200, "Total Customers", fontsize=8, ha='center')
ax3.text(-1.5, 5000, "41,067", fontsize=13, ha='center', color="navy")
ax3.text(-1.5, 4200, "Customers 2017", fontsize=6, ha='center')
ax3.text(-1.5, 2000, "52,410", fontsize=13, ha='center', color="navy")
ax3.text(-1.5, 1200, "Customers 2018", fontsize=6, ha='center')

# Sellers city
sns.barplot(x="total", y="seller_city", data=sellers_by_city, ax=ax5, palette='viridis')
ax5.set_title("The 10 cities with the most sellers", size=10, color='black')
ax5.set_xlabel("")
ax5.set_ylabel("")
for rect in ax5.patches:
    ax5.annotate('{:,.0f}'.format(rect.get_width()),(rect.get_width(),rect.get_y() + rect.get_height() / 2),
                xytext=(0, 0),textcoords='offset points', va='center', ha='left', size=8)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['left'].set_visible(False)
ax5.spines['bottom'].set_visible(False)
ax5.axes.get_xaxis().set_visible(False)

# Sellers states
sns.barplot(x="total", y="seller_state", data=sellers_by_states, ax=ax6, palette="YlGnBu")
ax6.set_title("The 10 states with the most sellers", size=11, color='black')
ax6.set_xlabel("")
ax6.set_ylabel("")
for rect in ax6.patches:
    ax6.annotate('{:,.0f}'.format(rect.get_width()),(rect.get_width(),rect.get_y() + rect.get_height() / 2),
                xytext=(0, 0),textcoords='offset points', va='center', ha='left', size=8)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.spines['left'].set_visible(False)
ax6.spines['bottom'].set_visible(False)
ax6.axes.get_xaxis().set_visible(False)

#Sellers per year
sns.lineplot(x="order_purchase_year_month", y="order_id", data=sellers_x_date.groupby("order_purchase_year_month").agg({"order_id" : "count"}).reset_index(),ax=ax8,
             color='darkslateblue', linewidth=1, marker='o', markersize=3)
sns.barplot(x="order_purchase_year_month", y="order_id", data=sellers_x_date.groupby("order_purchase_year_month").agg({"order_id" : "count"}).reset_index(),ax=ax8, alpha=0.1)
ax8.set_title("Seller Evolution", size=10, color="black")
ax8.set_xlabel("")
ax8.set_ylabel("")
ax8.set_ylim(0, 500)
plt.setp(ax8.get_xticklabels(), rotation=45)
for p in ax8.patches:
        ax8.annotate('{:,.0f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),
                    ha="center", va="top", xytext=(0, 15), textcoords="offset points", 
                    color= "black", size=8)
ax8.spines["top"].set_visible(False)
ax8.spines["right"].set_visible(False)
ax8.spines["left"].set_visible(False)
ax8.set_yticklabels([])
ax8.set_yticks([])
        
#Total de Sellers
ax8.text(-1.5, 460, "3068", fontsize=12, ha='center', color="navy")
ax8.text(-1.5, 420, "Total Sellers", fontsize=8, ha='center')
ax8.text(-1.5, 300, "1,236", fontsize=10, ha='center', color="navy")
ax8.text(-1.5, 260, "Sellers 2017", fontsize=6, ha='center')
ax8.text(-1.5, 140, "1,832", fontsize=10, ha='center', color="navy")
ax8.text(-1.5, 100, "Sellers 2018", fontsize=6, ha='center')

plt.suptitle("Customers and Sellers (2017-2018)", size=18)

# ORDERS IN OLIST ECOM
figO = plt.figure(constrained_layout=True, figsize=(15, 8))

# Axis definition
gs = GridSpec(2, 2, figure=figO)
ax1 = figO.add_subplot(gs[0, :])
ax2 = figO.add_subplot(gs[1, 0])
ax3 = figO.add_subplot(gs[1, 1])

# Lineplot - Evolution of e-commerce orders along time 
sns.lineplot(x="order_purchase_year_month", y="order_id", data=orders.groupby("order_purchase_year_month").agg({"order_id" : "count"}).reset_index(), ax=ax1, legend=False,
              marker='o',markersize=8)
sns.barplot(x="order_purchase_year_month", y="order_id", data=orders.groupby("order_purchase_year_month").agg({"order_id" : "count"}).reset_index(), ax=ax1, alpha=0.1)
#plt.setp(ax1.get_xticklabels(), rotation=45)
ax1.set_title("Orders in Brazilian e-commerce", size=12, color='black')
ax1.set_ylabel("")
ax1.set_xlabel("")
ax1.set_ylim(0,8500)
for p in ax1.patches:
        ax1.annotate('{:,.0f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),
                    ha='center', va='bottom', xytext=(0, 5), textcoords='offset points',
                    color= 'black', size=12)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["left"].set_visible(False)
ax1.set_yticklabels([])
ax1.set_yticks([])

#Total Orders
ax1.text(-1, 7600, "96,708", fontsize=12, ha='center', color="navy")
ax1.text(-1, 7200, "Total Customers", fontsize=8, ha='center')
ax1.text(-1, 6000, "42,697", fontsize=10, ha='center', color="navy")
ax1.text(-1, 5600, "Customers 2017", fontsize=6, ha='center')
ax1.text(-1, 4400, "54,011", fontsize=10, ha='center', color="navy")
ax1.text(-1, 4000, "Customers 2018", fontsize=6, ha='center')

# Barchart - Total of orders by day of week
day_order= ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
sns.countplot(x="order_purchase_dayofweek_name", data=orders, order=day_order, ax=ax2, palette="GnBu_r")
ax2.set_title("Orders by Day of Week", size=12, color='black')
ax2.set_xlabel("")
ax2.set_ylabel("")
for p in ax2.patches:
        ax2.annotate('{:,.0f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),
                    ha="center", va="bottom", xytext=(0, 1), textcoords="offset points",
                    color= "black")
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.yaxis.set_major_formatter(formatter_thousand_count_y)

# Barchart - Total of orders by time of the day
sns.countplot(x="order_purchase_time_day", data=orders,ax=ax3, palette="GnBu")
ax3.set_title("Orders by Time of the Day", size=12, color='black')
ax3.set_xlabel("")
ax3.set_ylabel("")
for p in ax3.patches:
        ax3.annotate('{:,.0f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),
                     ha="center", va="bottom", xytext=(0, 1), textcoords="offset points",
                    color= "black")
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.yaxis.set_major_formatter(formatter_thousand_count_y)

#We group order_id of items.csv to get the total cost of the order. (this because the purchase breakdown comes)
price = order_items.groupby("order_id").agg({ 'price': 'sum', 'freight_value': 'sum'}).reset_index()
price["total"] = price["price"] + price["freight_value"]
price

#We join the prices of the dataframe with the dataframe of orders, to calculate the sales by year, month, etc.
sales = pd.merge(price, orders, on="order_id")
sales_year_month = sales.groupby(by=["order_purchase_year", "order_purchase_year_month","order_purchase_month_name"]).agg({"order_id": "count","price": "sum","freight_value": "sum","total": "sum"}).reset_index()
sales_year = sales.groupby(by=["order_purchase_year"]).agg({"total": "sum"}).reset_index()

# ORDER VALUE AND SHIPPING COST
figOS = plt.figure(constrained_layout=True, figsize=(15, 10))

# Axis definition
gs = GridSpec(3, 2, figure=figOS)
ax1 = figOS.add_subplot(gs[0, :])
ax2 = figOS.add_subplot(gs[1, :])
ax3 = figOS.add_subplot(gs[2,:])



month_order= ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

sns.barplot(x="order_purchase_month_name", y="price", data=sales_year_month.loc[sales_year_month["order_purchase_year"]==2017], order=month_order, ax=ax1, color="skyblue", label="Price")
sns.barplot(x="order_purchase_month_name", y="freight_value", data=sales_year_month.loc[sales_year_month["order_purchase_year"]==2017], order=month_order, ax=ax1, color="yellow", label="Freight")

ax1.set_title("Order value and shipping cost 2017", size=12)
ax1.set_xlabel("")
ax1.set_ylabel("")
ax1.legend(loc="upper right")
for p in ax1.patches:
        ax1.annotate('R${:,.0f}'.format(p.get_height()+5.9), (p.get_x()+0.4, p.get_height()),
                    ha='center', va='top',
                    color= 'black')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.yaxis.set_major_formatter(formatter_millons_real_y)

ax1.text(1, 900000, f'R$7,142,672', fontsize=13, color='mediumseagreen', ha='center')
ax1.text(1, 820000, 'Amount sold', fontsize=6, ha='center')


sns.barplot(x="order_purchase_month_name", y="price", data=sales_year_month.loc[sales_year_month["order_purchase_year"]==2018], order=month_order, ax=ax2, color="skyblue", label="Price")
sns.barplot(x="order_purchase_month_name", y="freight_value", data=sales_year_month.loc[sales_year_month["order_purchase_year"]==2018], order=month_order, ax=ax2, color="yellow", label="Freight")
ax2.set_title("Order value and shipping cost 2018", size=12)
ax2.set_xlabel("")
ax2.set_ylabel("")
ax2.legend(loc="upper right")
for p in ax2.patches:
        ax2.annotate('R${:,.0f}'.format(p.get_height()+5.9), (p.get_x()+0.4, p.get_height()),
                    ha='center', va='top',
                    color= 'black')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.yaxis.set_major_formatter(formatter_millons_real_y)

ax2.text(9, 950000, f'R$8,643,697', fontsize=12, color='mediumseagreen', ha='center')
ax2.text(9, 800000, 'It has reached 21% more than in 2017 \nwithout ending 2018', fontsize=8, ha='center')

sns.lineplot(x="order_purchase_month_name", y='total', data=sales_year_month.loc[sales_year_month["order_purchase_year"]==2017], ax=ax3,linewidth=3.0, label="2017", marker='o', markersize=5)
sns.lineplot(x="order_purchase_month_name", y='total', data=sales_year_month.loc[sales_year_month["order_purchase_year"]==2018], ax=ax3,linewidth=3.0,label="2018", marker='o', markersize=5)
ax3.set_title("Total purchases per month", size=13, color='black')
ax3.set_xlabel("")
ax3.set_ylabel("")
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.yaxis.set_major_formatter(formatter_millons_real_y)

plt.suptitle("")

# merge elements of the data frame (order_payments, orders) to know the trend of the payment method of the customers
pay = pd.merge(order_payments, orders, on="order_id")
pay["payment_type"] = pay["payment_type"].str.capitalize()
#Grouping for pie chart, what is the most used payment method?
pay1 = pay.groupby(by=["payment_type"]).agg({"order_id": "count","payment_value": "sum"}).reset_index().sort_values(by="order_id",ascending=False)

#Group for bar chart to count transactions by payment type.
pay2 = pay.groupby(by=["payment_type", "order_purchase_year"]).agg({"order_id": "count"}).reset_index().sort_values(by=["order_purchase_year","order_id"],ascending=False)

#Group to know which are the trends of the payment methods
pay3 = pay.groupby(by=["order_purchase_year_month", "payment_type"]).agg({"order_id": "count"}).reset_index().sort_values(by=['order_purchase_year_month', 'order_id'], ascending=[True, False])

# PAYMENT METHOD
figPM = plt.figure(constrained_layout=True, figsize=(15, 10))

# Axis definition
gs = GridSpec(2, 3, figure=figPM)
ax1 = figPM.add_subplot(gs[0, 0])
ax2 = figPM.add_subplot(gs[0, 1:])
#ax3 = fig.add_subplot(gs[0, 2])
ax4 = figPM.add_subplot(gs[1,:])

colors_list1 = ['yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink']
explode1 = (0.1, 0.1, 0.1, 0.1, 0.8)

ax1.pie(pay1["order_id"], explode=explode1, autopct='%1.1f%%',shadow=True, startangle=40,pctdistance=0.8, colors=colors_list1)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.legend(labels=pay1["payment_type"], loc='best')
ax1.set_title("Most used payment method", size=13, color='black')

sns.barplot(x="order_id", y="payment_type", data=pay2,  ax=ax2, hue="order_purchase_year", palette="Set2")
ax2.legend(loc="best")
ax2.set_xlabel("")
ax2.set_ylabel("")
ax2.set_title("Numbers of transactions by payment type", size=13, color="black" )
for rect in ax2.patches:
    ax2.annotate('{:,.0f}'.format(rect.get_width()),(rect.get_width(),rect.get_y() + rect.get_height() / 2),
                xytext=(0, 0),textcoords='offset points', va='center', ha='left', size=10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.axes.get_xaxis().set_visible(False) 
#ax2.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    
sns.lineplot(x='order_purchase_year_month', y='order_id', data=pay3, ax=ax4, hue='payment_type',legend=False,
             style='payment_type', size='payment_type', palette=colors_list1, marker='o',markersize=5)
ax4.legend(labels=pay3["payment_type"], loc='upper left',fontsize=10)
ax4.set_title("Trend in the payment method", size=14, color="black")
ax4.set_xlabel("")
ax4.set_ylabel("")
plt.setp(ax4.get_xticklabels(), rotation=45)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

plt.suptitle("Payment method", size=13)

#merge data frame elements (products, product category) to know the English name of the products
products = pd.merge(products, products_category_translation, on="product_category_name")
#products

#Assemble a new dataframe with only product_id and the English name of the product
products = products[["product_id", "product_category_name_english"]]
#products.head()

#We put the product category in the items file
products = pd.merge(order_items, products, on="product_id")
#products

#We calculate the total that the customer paid for the product
products["total"] = products["price"] + products["freight_value"]
#new dataframe with the columns we just need
products = products[["order_id", "product_id", "price" , "freight_value", "total", "product_category_name_english"]]
#products

#We join our dataframe products with orders to obtain the information of the dates
products = pd.merge(products, orders, on="order_id")
#products

#new dataframe with the columns we just need
products = products[["product_id", "price" , "freight_value", "total", "product_category_name_english","order_purchase_year","order_purchase_month_name","order_purchase_year_month"]]
products["product_category_name_english"] = products["product_category_name_english"].str.capitalize()

#We group by category to generate the scatter plot
products_category = products.groupby("product_category_name_english").agg({"order_purchase_year_month" : "count","total" : "sum"}).reset_index().sort_values(by="order_purchase_year_month", ascending=False).head(20)

#We group by category to generate the bar plot
products_category_year = products.groupby(by=["product_category_name_english","order_purchase_year"]).agg({"order_purchase_year_month" : "count","total" : "sum"}).reset_index().sort_values(by=["order_purchase_year","order_purchase_year_month"], ascending=[True,False])

products_cat_trends = products.groupby(by=["product_category_name_english","order_purchase_year_month"]).agg({"order_purchase_year" : "count","total" : "sum"}).reset_index().sort_values(by=["order_purchase_year_month","order_purchase_year","product_category_name_english"], ascending=[True,False,True])
#We remove the data from 2016, we are only analyzing 2017 and 2018
products_cat_trends.drop(products_cat_trends.loc[products_cat_trends["order_purchase_year_month"]=="201609"].index, inplace=True)
products_cat_trends.drop(products_cat_trends.loc[products_cat_trends["order_purchase_year_month"]=="201610"].index, inplace=True)
products_cat_trends.drop(products_cat_trends.loc[products_cat_trends["order_purchase_year_month"]=="201612"].index, inplace=True)

#We filter the categories that are in the Top 5 by category in 2017 y 2018
products_cat_trends = products_cat_trends[(products_cat_trends["product_category_name_english"]=="Bed_bath_table") | 
                                  (products_cat_trends["product_category_name_english"]=="Furniture_decor")|
                                  (products_cat_trends["product_category_name_english"]=="Sports_leisure")|
                                  (products_cat_trends["product_category_name_english"]=="Health_beauty")|
                                  (products_cat_trends["product_category_name_english"]=="Computers_accessories")]

# PRODUCT CATEGORIES
figPC = plt.figure(constrained_layout=True, figsize=(15, 15))

# Axis definition
gs = GridSpec(9, 2, figure=figPC)
ax1 = figPC.add_subplot(gs[:3,0:])
ax2 = figPC.add_subplot(gs[3:5, 0])
ax3 = figPC.add_subplot(gs[3:5, 1])
ax4 = figPC.add_subplot(gs[5:,0:])

for product in products_category["product_category_name_english"].unique():
    data = products_category[products_category["product_category_name_english"] == product]
    ax1.scatter(data['total'],data['order_purchase_year_month'],
                s=0.5*data['order_purchase_year_month']**1,
                alpha = 0.5,
                label=product)
    
for index, row in products_category.iterrows():
    ax1.annotate(row['product_category_name_english'],
                 (row['total'], row['order_purchase_year_month']),
                 textcoords="offset points",
                 xytext=(0,5),
                 ha='left')

ax1.set_xlabel("")
ax1.set_ylabel("Number of pieces")
ax1.set_title("The 20 best-selling categories",size=12)
ax1.set_ylim(0,14000)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('$R{x:,.0f}'))
ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

sns.barplot(x="product_category_name_english", y="order_purchase_year_month", data=products_category_year.loc[products_category_year["order_purchase_year"]==2017].head(5), ax=ax2, palette="Blues_r")
ax2.set_xlabel("")
ax2.set_ylabel("Number of pieces")
ax2.set_yticklabels([])
ax2.set_yticks([])
ax2.set_title("Top 5 best-selling categories 2017", size=12)
#ax2.set_ylim(0,6000)
plt.setp(ax2.get_xticklabels(), rotation=30)
for p in ax2.patches:
        ax2.annotate('{:,.0f}'.format(p.get_height()+5.9), (p.get_x()+0.4, p.get_height()),
                    ha="center", va="bottom", xytext=(0, 1), textcoords="offset points",
                    color= 'black', size=10)
#ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2_twx = ax2.twinx()
sns.lineplot(x="product_category_name_english", y='total', data=products_category_year.loc[products_category_year["order_purchase_year"]==2017].head(5), ax=ax2_twx, linewidth=1.0, color="darkblue", marker="o", markersize=5)
#ax2_twx.set_ylim(0,990000)
ax2_twx.set_ylabel("Amount")
ax2_twx.spines['top'].set_visible(False)
ax2_twx.spines['left'].set_visible(False)
ax2_twx.yaxis.set_major_formatter(formatter_thousand_real_y)

sns.barplot(x="product_category_name_english", y="order_purchase_year_month", data=products_category_year.loc[products_category_year["order_purchase_year"]==2018].head(5), ax=ax3, palette="Blues_r")
ax3.set_xlabel("")
ax3.set_ylabel("Number of pieces")
ax3.set_yticklabels([])
ax3.set_yticks([])
ax3.set_title("Top 5 best-selling categories 2018", size=12)
for p in ax3.patches:
        ax3.annotate('{:,.0f}'.format(p.get_height()+5.9), (p.get_x()+0.4, p.get_height()),
                    ha="center", va="bottom", xytext=(0, 1), textcoords="offset points",
                    color= 'black', size=10)
ax3.spines['top'].set_visible(False)
ax3.spines['left'].set_visible(False)
plt.setp(ax3.get_xticklabels(), rotation=30)
ax3_twx = ax3.twinx()
sns.lineplot(x="product_category_name_english", y='total', data=products_category_year.loc[products_category_year["order_purchase_year"]==2018].head(5), ax=ax3_twx, linewidth=1.0, color="darkblue", marker="o", markersize=5)
#ax3_twx.set_ylim(0,990000)
ax3_twx.set_ylabel("Amount")
ax3_twx.yaxis.set_major_formatter(formatter_thousand_real_y)
ax3_twx.spines['top'].set_visible(False)
ax3_twx.spines['top'].set_visible(False)
ax3_twx.spines['left'].set_visible(False)

sns.lineplot(x='order_purchase_year_month', y='order_purchase_year', data=products_cat_trends, ax=ax4, hue='product_category_name_english',legend=False, style='product_category_name_english', size='product_category_name_english', marker='o', linewidth=3)
ax4.legend(labels=products_cat_trends["product_category_name_english"], loc='upper left', fontsize=10)
ax4.set_title("Top 5 trend 2017-2018", size=14)
ax4.set_ylabel("Number of pieces")
ax4.set_xlabel("")
ax4.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

#We group by product to generate the scatter plot
products_top = products.groupby(by=["product_id","product_category_name_english"]).agg({"order_purchase_month_name" : "count","total" : "sum"}).reset_index().sort_values(by="order_purchase_month_name", ascending=False).head(10)

#We group by product to generate the bar plot
products_top_year = products.groupby(by=["product_id","order_purchase_year"]).agg({"order_purchase_year_month" : "count","total" : "sum"}).reset_index().sort_values(by=["order_purchase_year","order_purchase_year_month"], ascending=[True,False])
#We remove the data from 2016, we are only analyzing 2017 and 2018
products_top_year.drop(products_top_year.loc[products_top_year["order_purchase_year"]==2016].index, inplace=True)

# SELLING
figS = plt.figure(constrained_layout=True, figsize=(15, 10))

# Axis definition
gs = GridSpec(4, 2, figure=figS)
ax1 = figS.add_subplot(gs[:2, 0])
ax2 = figS.add_subplot(gs[:2, 1])
ax3 = figS.add_subplot(gs[2:,0])
ax4 = figS.add_subplot(gs[2:,1])

sns.barplot(x="order_purchase_month_name", y="product_id",  data=products_top, ax=ax1, palette="rocket")
ax1.set_xlabel("")
ax1.set_ylabel("Product ID")
ax1.set_title("Numbers of pieces sold",size=12)
for rect in ax1.patches:
    ax1.annotate('{:,.0f}'.format(rect.get_width()),(rect.get_width(),rect.get_y() + rect.get_height() / 2),
                xytext=(0, 0),textcoords='offset points', va='center', ha='right', size=8, color="white")    
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.axes.get_xaxis().set_visible(False)
    
sns.barplot(x="total", y="product_id",  data=products_top, ax=ax2, palette="rocket")
ax2.set_xlabel("")
ax2.set_ylabel("")
ax2.set_title("Amount sold", size=12)
for rect in ax2.patches:
    ax2.annotate('R${:,.0f}'.format(rect.get_width()),(rect.get_width(),rect.get_y() + rect.get_height() / 2),
                xytext=(0, 0),textcoords='offset points', va='center', ha='right', size=12, color="white")    
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.axes.get_xaxis().set_visible(False)
ax2.axes.get_yaxis().set_visible(False)

sns.barplot(x="product_id", y="order_purchase_year_month", data=products_top_year.loc[products_top_year["order_purchase_year"]==2017].head(10), ax=ax3, palette="Purples_r")
ax3.set_xlabel("Product ID")
ax3.set_ylabel("Number Of Pieces Sold")
plt.setp(ax3.get_xticklabels(), rotation=90)
ax3.set_title("Most seller in 2017", size=12)
ax3.set_yticklabels([])
ax3.set_yticks([])
for p in ax3.patches:
        ax3.annotate('{:,.0f}'.format(p.get_height()+5.9), (p.get_x()+0.4, p.get_height()),
                    ha="center", va="bottom", xytext=(0, 1), textcoords="offset points",
                    color= 'black', size=10)
ax3.spines['top'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3_twx = ax3.twinx()
sns.lineplot(x="product_id", y='total', data=products_top_year.loc[products_top_year["order_purchase_year"]==2017].head(10), ax=ax3_twx, linewidth=1.0, color="darkblue", marker="o", markersize=5)
ax3_twx.set_ylabel("Amount Sold")
ax3_twx.spines['top'].set_visible(False)
ax3_twx.spines['left'].set_visible(False)
ax3_twx.yaxis.set_major_formatter(formatter_thousand_real_y)

sns.barplot(x="product_id", y="order_purchase_year_month", data=products_top_year.loc[products_top_year["order_purchase_year"]==2018].head(10), ax=ax4, palette="Purples_r")
ax4.set_xlabel("Product ID")
ax4.set_ylabel("Number Of Pieces Sold")
plt.setp(ax4.get_xticklabels(), rotation=90)
ax4.set_title("Most seller in 2018", size=12)
ax4.set_yticklabels([])
ax4.set_yticks([])
for p in ax4.patches:
        ax4.annotate('{:,.0f}'.format(p.get_height()+5.9), (p.get_x()+0.4, p.get_height()),
                    ha="center", va="bottom", xytext=(0, 1), textcoords="offset points",
                    color= 'black', size=10)
ax4.spines['top'].set_visible(False)
ax4.spines['left'].set_visible(False)
ax4_twx = ax4.twinx()
sns.lineplot(x="product_id", y='total', data=products_top_year.loc[products_top_year["order_purchase_year"]==2018].head(10), ax=ax4_twx, linewidth=1.0, color="darkblue", marker="o", markersize=5)
ax4_twx.set_ylabel("Amount Sold")
ax4_twx.spines['top'].set_visible(False)
ax4_twx.spines['left'].set_visible(False)
ax4_twx.yaxis.set_major_formatter(formatter_thousand_real_y)

plt.suptitle("Number of Pieces and Amount Sold \nby Products", size=13)

order_status = orders.groupby("order_status").size().reset_index().sort_values(by=0,ascending=False)
order_status["order_status"] = order_status["order_status"].str.capitalize()

# ORDER STATUS
# Tạo một biểu đồ pie
figORS, ax = plt.subplots()

# Thêm dữ liệu vào biểu đồ
ax.pie(order_status[0], labels=order_status["order_status"], colors=['#334668','#496595','#6D83AA','#91A2BF','#C8D0DF'], autopct='%1.1f%%')

# Đặt tiêu đề cho biểu đồ
plt.title("Orders Status (2016-2018)")

# Ẩn các trục
ax.axis('off')


#We group the orders to know the seller
freight_value = order_items.groupby(by=["order_id", "seller_id"]).agg({"price" : "count","freight_value" : "sum"}).reset_index().sort_values(by="price", ascending=False)
#freight_value
# We group the freight value and the sales to know the dates made to the freight
freight_value = pd.merge(freight_value, sales, on = "order_id")
#freight_value
# We group the freight value and the sellers to know the city and state of the seller
freight_value = pd.merge(freight_value, sellers, on = "seller_id")
#freight_value
# We select the columns necessary for the analysis
freight_value = freight_value[["seller_id", "order_purchase_year", "order_purchase_year_month","order_purchase_timestamp", "order_delivered_carrier_date" , "order_delivered_customer_date", "order_estimated_delivery_date", "freight_value_x", "seller_city", "seller_state"]]

# We eliminate the rows that do not contain data in the columns "order delivered carrier date" and "order delivered customer date"
freight_value = freight_value.dropna(subset=["order_delivered_customer_date","order_delivered_carrier_date"])
#freight_value.head()

#Convert column order_delivered_customer_date (object) to datetime
freight_value["order_delivered_customer_date"] = freight_value["order_delivered_customer_date"].apply(pd.to_datetime)

# Formatting dates to calculate delivery time
freight_value["order_purchase_timestamp"] = pd.to_datetime(freight_value["order_purchase_timestamp"].dt.strftime("%Y-%m-%d"))
freight_value["order_delivered_carrier_date"] = pd.to_datetime(freight_value["order_delivered_carrier_date"].dt.strftime("%Y-%m-%d"))
freight_value["order_delivered_customer_date"] = pd.to_datetime(freight_value["order_delivered_customer_date"].dt.strftime("%Y-%m-%d"))

# We take the difference of days between the columns
freight_value["time_delivery_customer"] = freight_value["order_delivered_customer_date"] - freight_value["order_purchase_timestamp"]
freight_value["time_delivery_estimated"] = freight_value["order_delivered_customer_date"] - freight_value["order_estimated_delivery_date"]
freight_value["time_purchase_estimated_day_time"] = freight_value["order_estimated_delivery_date"] - freight_value["order_purchase_timestamp"]

#freight_value

# We convert the two columns into str
freight_value["time_delivery_customer"] = freight_value["time_delivery_customer"].astype(str)
freight_value["time_delivery_estimated"] = freight_value["time_delivery_estimated"].astype(str)
freight_value["time_purchase_estimated_day_time"] = freight_value["time_purchase_estimated_day_time"].astype(str)

separator_c = freight_value["time_delivery_customer"].str.rsplit(" ", n=1, expand=True)
separator_c.columns = ["customer_day_time", "1"]
separator_c = separator_c.drop(columns = ["1"])
freight_value = pd.concat([freight_value, separator_c], axis=1)

separator_d = freight_value["time_delivery_estimated"].str.rsplit(" ", n=1, expand=True)
separator_d.columns = ["delivery_day_time", "2"]
separator_d = separator_d.drop(columns = ["2"])
freight_value = pd.concat([freight_value, separator_d], axis=1)

separator_pe = freight_value["time_purchase_estimated_day_time"].str.rsplit(" ", n=1, expand=True)
separator_pe.columns = ["purchase_estimated_day_time", "2"]
separator_pe = separator_pe.drop(columns = ["2"])
freight_value = pd.concat([freight_value, separator_pe], axis=1)

freight_value["customer_day_time"] = freight_value["customer_day_time"].astype(int)
freight_value["delivery_day_time"] = freight_value["delivery_day_time"].astype(int)
freight_value["purchase_estimated_day_time"] = freight_value["purchase_estimated_day_time"].astype(int)

conditionlist = [
    (freight_value["delivery_day_time"] <=0),(freight_value["delivery_day_time"] > 0)]

choicelist = ["On Time", "Out of Time"]
freight_value['delivery_status'] = np.select(conditionlist, choicelist, default='Not Specified')

delivery_status = freight_value.groupby("delivery_status").count()["seller_id"].reset_index()

delivery_status_year = freight_value.groupby(by=["order_purchase_year","delivery_status"]).count()["seller_id"].reset_index()

#The 5 states that take the most time compared to the estimated date
# Convert 'delivery_day_time' to numeric
freight_value["delivery_day_time"] = pd.to_numeric(freight_value["delivery_day_time"], errors='coerce')

# Filter rows where 'delivery_day_time' is greater than or equal to 1
delivery_low = freight_value[freight_value["delivery_day_time"] >= 1].copy()

# Convert 'delivery_day_time' to numeric again (if needed)
delivery_low["delivery_day_time"] = pd.to_numeric(delivery_low["delivery_day_time"], errors='coerce')

# Group by 'seller_state' and calculate mean 'delivery_day_time', then sort and select top 10
delivery_low = delivery_low.groupby("seller_state")["delivery_day_time"].mean().reset_index().sort_values(by="delivery_day_time", ascending=False).head(10)

#The 5 states that take less time to deliver compared to the estimated date
# Convert 'delivery_day_time' to numeric
freight_value["delivery_day_time"] = pd.to_numeric(freight_value["delivery_day_time"], errors='coerce')

# Filter rows where 'delivery_day_time' is less than or equal to 0
delivery_fast = freight_value[freight_value["delivery_day_time"] <= 0].copy()

# Convert 'delivery_day_time' to numeric again (if needed)
delivery_fast["delivery_day_time"] = pd.to_numeric(delivery_fast["delivery_day_time"], errors='coerce')

# Group by 'seller_state' and calculate mean 'delivery_day_time', then sort and select top 10
delivery_fast = delivery_fast.groupby("seller_state")["delivery_day_time"].mean().reset_index().sort_values(by="delivery_day_time", ascending=True).head(10)

delivery = freight_value.groupby("seller_state").agg({"freight_value_x":"mean", "customer_day_time":"mean", "purchase_estimated_day_time": "mean"}).reset_index().sort_values(by="freight_value_x", ascending=False)

# FREIGHT TIME AND COST
figFC = plt.figure(constrained_layout=True, figsize=(15, 20))

# Axis definition
gs = GridSpec(6, 2, figure=figFC)
ax1 = figFC.add_subplot(gs[0, 0])
ax2 = figFC.add_subplot(gs[0, 1:])
ax3 = figFC.add_subplot(gs[1, :])
ax4 = figFC.add_subplot(gs[2, 0])
ax5 = figFC.add_subplot(gs[2, 1])
ax6 = figFC.add_subplot(gs[3, :])

colors_list2 = ['lightskyblue', 'lightgreen']
explode2 = (0.0, 0.2)

ax1.pie(delivery_status["seller_id"], autopct='%1.1f%%',shadow=True, startangle=40,pctdistance=0.8, explode=explode2, colors=colors_list2)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.legend(labels=delivery_status["delivery_status"], loc='upper left')
ax1.set_title("Delivery Status", size=12, color='black')

sns.barplot(x="order_purchase_year", y="seller_id", data=delivery_status_year, ax=ax2, hue="delivery_status", palette="mako")
ax2.set_xlabel("")
ax2.set_ylabel("")
ax2.legend(loc="best")
ax2.set_title("Delivery Status \nPer Year", size=10, color="black")
for p in ax2.patches:
        ax2.annotate('{:,.0f}'.format(p.get_height()), (p.get_x()+0.2, p.get_height()),
                    ha="center", va="bottom", xytext=(0, 1), textcoords="offset points",
                    color= 'black')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.set_yticklabels([])
ax2.set_yticks([])


sns.histplot(data=freight_value, x=freight_value["delivery_day_time"], kde=True, ax=ax3, color=colors_list2, hue=freight_value["delivery_status"])
ax3.set_xlim(-40,20)
ax3.set_xlabel("Days")
ax3.set_ylabel("")
ax3.set_title("Difference between the estimated delivery date and the actual delivery date", size=14, color="black")
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

sns.barplot(x="delivery_day_time", y="seller_state",  data=delivery_low, ax=ax4, color="darkblue")
ax4.set_xlabel("Days")
ax4.set_ylabel("")
ax4.set_title("The 10 states that take longer to deliver compared to the estimated date",size=12, color="black")
for rect in ax4.patches:
    ax4.annotate('{:,.0f} days'.format(rect.get_width()),(rect.get_width(),rect.get_y() + rect.get_height() / 2),
                xytext=(0, 0),textcoords='offset points', va='center', ha='right', size=10, color="white")
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_visible(False)
ax4.spines['bottom'].set_visible(False)
ax4.axes.get_xaxis().set_visible(False)

sns.barplot(x="delivery_day_time", y="seller_state",  data=delivery_fast, ax=ax5, color="darkblue")
ax5.set_xlabel("Days")
ax5.set_ylabel("")
ax5.set_title("The 10 states that take less time to deliver compared to the estimated date",size=12, color="black")
for rect in ax5.patches:
    ax5.annotate('{:,.0f} days'.format(rect.get_width()),(rect.get_width(),rect.get_y() + rect.get_height() / 2),
                xytext=(-1, 0),textcoords='offset points', va='center', ha='left', size=10, color="white")
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['left'].set_visible(False)
ax5.spines['bottom'].set_visible(False)
ax5.axes.get_xaxis().set_visible(False)

sns.barplot(x="seller_state", y="freight_value_x", data=delivery, ax=ax6, color="dodgerblue")
#ax6.legend(loc="best")
ax6.set_xlabel("States")
ax6.set_ylabel("Average freight cost")
ax6.set_title("Average Delivery Cost and Shipping", size=11, color="black", loc='left')
for p in ax6.patches:
        ax6.annotate('R${:,.0f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),
                    ha="center", va="bottom", xytext=(0, 1), textcoords="offset points",
                    color= 'black', size=8)
ax6.spines['top'].set_visible(False)
ax6.spines['left'].set_visible(False)
ax6.set_yticklabels([])
ax6.set_yticks([])
ax6_twx = ax6.twinx()
sns.lineplot(x="seller_state", y='customer_day_time', data=delivery, ax=ax6_twx, linewidth=1.0, color="darkblue", marker="o", markersize=6)
sns.lineplot(x="seller_state", y='purchase_estimated_day_time', data=delivery, ax=ax6_twx, linewidth=1.0, color="darkgreen", marker="o", markersize=5)
ax6_twx.set_ylabel("Days")
ax6_twx.spines['top'].set_visible(False)
ax6_twx.spines['left'].set_visible(False)

ax6_twx.annotate(f'Average Estimated Delivery Time', ("AM", 48), xytext=(75, 25), 
             textcoords='offset points', bbox=dict(boxstyle="round4", fc="w", pad=.8),
             arrowprops=dict(arrowstyle='-|>', fc='w'), color='darkblue', ha='center')

ax6_twx.annotate(f'Average Delivery Time', ("AM", 39), xytext=(100, 25), 
             textcoords='offset points', bbox=dict(boxstyle="round4", fc="w", pad=.8),
             arrowprops=dict(arrowstyle='-|>', fc='w'), color='darkgreen', ha='center')


plt.suptitle("Freight times and costs", size=18)

conditionlist = [
    (order_reviews["review_score"] <=2),(order_reviews["review_score"] == 3),(order_reviews["review_score"] >= 4)]

choicelist = ["Negative", "Neutral", "Positive"]
order_reviews["score_classification"] = np.select(conditionlist, choicelist, default='Not Specified')

conditionlist = [
    (order_reviews["review_comment_message"].isnull())]

choicelist = ["No Comment"]
order_reviews["comment_classification"] = np.select(conditionlist, choicelist, default="With Comments")

#Filter what has information
comments = order_reviews[order_reviews.review_comment_message.notnull()]
comments = comments[["review_id", "review_comment_message"]]

# Defining a function to remove the stopwords and to lower the comments
def stopwords_removal(text, cached_stopwords=stopwords.words('portuguese')):
    return [c.lower() for c in text.split() if c.lower() not in cached_stopwords]

review_comment_message = [' '.join(stopwords_removal(review)) for review in comments["review_comment_message"]]
comments["review_comment_message"] = review_comment_message

def re_hiperlinks(text_list):
    """
    Args:
    ----------
    text_list: list object with text content to be prepared [type: list]
    """
    
    # Applying regex
    pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return [re.sub(pattern, ' ', r) for r in text_list]

# Applying RegEx
reviews_hiperlinks = re_hiperlinks(review_comment_message)
comments["review_comment_message"] = reviews_hiperlinks

def re_dates(text_list):
    """
    Args:
    ----------
    text_list: list object with text content to be prepared [type: list]
    """
    
    # Applying regex
    pattern = '([0-2][0-9]|(3)[0-1])(\/|\.)(((0)[0-9])|((1)[0-2]))(\/|\.)\d{2,4}'
    return [re.sub(pattern, ' ', r) for r in text_list]

# Applying RegEx
reviews_dates = re_dates(reviews_hiperlinks)
comments["review_comment_message"] = reviews_dates

def re_money(text_list):
    """
    Args:
    ----------
    text_list: list object with text content to be prepared [type: list]
    """
    
    # Applying regex
    pattern = '[R]{0,1}\$[ ]{0,}\d+(,|\.)\d+'
    return [re.sub(pattern, '  ', r) for r in text_list]

# Applying RegEx
reviews_money = re_money(reviews_dates)
comments["review_comment_message"] = reviews_money

def re_numbers(text_list):
    """
    Args:
    ----------
    text_series: list object with text content to be prepared [type: list]
    """
    
    # Applying regex
    return [re.sub('[0-9]+', '  ', r) for r in text_list]

# Applying RegEx
reviews_numbers = re_numbers(reviews_money)
comments["review_comment_message"] = reviews_numbers

def re_negation(text_list):
    """
    Args:
    ----------
    text_series: list object with text content to be prepared [type: list]
    """
    
    # Applying regex
    return [re.sub('([nN][ãÃaA][oO]|[ñÑ]| [nN] )', '  ', r) for r in text_list]

# Applying RegEx
reviews_negation = re_negation(reviews_numbers)
comments["review_comment_message"] = reviews_negation

def re_special_chars(text_list):
    """
    Args:
    ----------
    text_series: list object with text content to be prepared [type: list]
    """
    
    # Applying regex
    return [re.sub('\W', ' ', r) for r in text_list]

# Applying RegEx
reviews_special_chars = re_special_chars(reviews_negation)
comments["review_comment_message"] = reviews_special_chars

def re_whitespaces(text_list):
    """
    Args:
    ----------
    text_series: list object with text content to be prepared [type: list]
    """
    
    # Applying regex
    white_spaces = [re.sub('\s+', ' ', r) for r in text_list]
    white_spaces_end = [re.sub('[ \t]+$', '', r) for r in white_spaces]
    return white_spaces_end

# Applying RegEx
reviews_whitespaces = re_whitespaces(reviews_special_chars)
comments["review_comment_message"] = reviews_whitespaces

wordcloud = WordCloud(width=5280, height=720, background_color='white', max_words=100).generate(str(reviews_whitespaces))

# REVIEWS
figRE = plt.figure(constrained_layout=True, figsize=(15, 10))

# Axis definition
gs = GridSpec(2, 2, figure=figRE)
ax1 = figRE.add_subplot(gs[0, 0])
ax2 = figRE.add_subplot(gs[0, 1:])
ax3 = figRE.add_subplot(gs[1, :])
#ax4 = fig.add_subplot(gs[2, 0])
#ax5 = fig.add_subplot(gs[2, 1])
#ax6 = fig.add_subplot(gs[3, :])

sns.barplot(x="review_score", y="review_id", data=order_reviews.groupby("review_score").count()["review_id"].reset_index(), ax=ax1, color="dodgerblue")
ax1.set_xlabel("Score", size=12)
ax1.set_ylabel("")
ax1.set_title("Score Received by Reviews", size=12, color="black")
for p in ax1.patches:
        ax1.annotate('{:,.0f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),
                    ha="center", va="bottom", xytext=(0, 1), textcoords="offset points",
                    color= 'black', size=10)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_yticklabels([])
ax1.set_yticks([])

comment_classification = order_reviews.groupby("comment_classification").count()["review_id"].reset_index()
colors_list3 = ['lightskyblue', 'lightgreen']
#explode3 = (0.0, 0.05)  explode=explode3,

ax2.pie(comment_classification["review_id"], autopct='%1.1f%%',shadow=True, startangle=40,pctdistance=0.8,  colors=colors_list3)
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax2.legend(labels=comment_classification["comment_classification"], loc='upper right')
ax2.set_title("Percentage of Comments Received", size=12, color='black')

ax3.imshow(wordcloud)
ax3.axis('off')
ax3.set_title("Most Commented Words", size=12, color='black')

plt.suptitle("Reviews", size=13)

# GEOLOCATION

figGEO, ax = plt.subplots(figsize=(10, 6))
geometry = [Point(xy) for xy in zip(geolocation['geolocation_lng'], geolocation['geolocation_lat'])]
gdf = GeoDataFrame(geolocation, geometry=geometry)

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Plot the world map on the ax
world.plot(ax=ax, color='white', edgecolor='black')

# Plot the geolocation data on the same ax
gdf.plot(ax=ax, marker='o', color='red', markersize=12)

# Add gridlines
ax.grid(True)

# Set longitude labels
ax.set_xticklabels(np.arange(-180, 180, 20))

# Set latitude labels
ax.set_yticklabels(np.arange(-90, 90, 10))