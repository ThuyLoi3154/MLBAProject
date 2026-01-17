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

from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots


from Models.MLmodel import df_predicter

def plot_product_category(category, time):
    # Filter the DataFrame for the specific product category
    df_category = df_predicter[df_predicter['product_category_name_english'] == category]

    # Group the data by week and calculate the sum
    df_grouped = df_category.groupby(time, as_index=False).sum()

    # Create the linechart
    fig = plt.figure(figsize=(10, 3))
    plt.plot(df_grouped[time], df_grouped['payment_value'], label='Sales')
    plt.title(f'Sales of {category} Over Time')
    plt.xlabel(f'{time}', fontsize=8)
    plt.ylabel('Sales', fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.legend()
    return fig

def plot_product_specific(product_id, time):
    # Filter the DataFrame for the specific product category
    df_category = df_predicter[df_predicter['product_id'] == product_id]

    # Group the data by week and calculate the sum
    df_grouped = df_category.groupby(time, as_index=False).sum()

    # Create the linechart
    fig = plt.figure(figsize=(10, 3))
    plt.plot(df_grouped[time], df_grouped['payment_value'], label='Sales')
    plt.title(f'Sales of {product_id} Over Time')
    plt.xlabel(f'{time}', fontsize=8)
    plt.ylabel('Sales', fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.legend()
    return fig

def plot_all_categories(time):
    # Group the data by week and category, then calculate the sum
    df_grouped = df_predicter.groupby([time, 'product_category_name_english']).sum().reset_index()

    # Get the list of unique categories
    categories = df_grouped['product_category_name_english'].unique()

    # Create the linechart
    fig = plt.figure(figsize=(10, 3))

    # Plot a line for each category
    for category in categories:
        df_category = df_grouped[df_grouped['product_category_name_english'] == category]
        plt.plot(df_category[time], df_category['payment_value'], label=category)

    plt.title('Sales of All Categories Over Time')
    plt.xlabel(f'{time}', fontsize=8)
    plt.ylabel('Sales', fontsize=8)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.legend()
    return fig
