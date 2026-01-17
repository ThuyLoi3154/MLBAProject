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

import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import StandardScaler
from matplotlib.backends.backend_qt5agg import FigureCanvas
from plotly.subplots import make_subplots

from Models.processdata import merged_df
from Models.RFMcalculation import RFM_df2
from Models.kmean import RFM_df4

customer_data_with_recommendations = pd.read_csv('D:\\project\\dataset\\customer_data_with_recommendations.csv')

def filter_customer_data(df, customer_unique_id):
    return df[df['customer_unique_id'] == customer_unique_id]
