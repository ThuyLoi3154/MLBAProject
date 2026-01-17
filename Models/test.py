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


from chartoption import plot_product_category, plot_product_specific, plot_all_categories

category = 'sport_leisure'
fig = plot_product_category(category, 'week')
fig.show()