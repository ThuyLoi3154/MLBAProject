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
from matplotlib.backends.backend_qt5agg import FigureCanvas
from plotly.subplots import make_subplots

from Models.processdata import merged_df
from Models.RFMcalculation import RFM_df2

# Kmeans clustering
RFM_df3= RFM_df2.drop(["recency_score", "frequency_score", "monetary_score", "RFM_SCORE", "Segment"], axis=1)
# apply log transform for the Frequency and Monetary columns as they are very skewed
RFM_log= RFM_df3.copy()
for i in RFM_log.columns[1:]:
    RFM_log[i] = np.log10(RFM_log[i])

# apply StandardScaler
scaler= StandardScaler()
RFM_log_scaled= scaler.fit_transform(RFM_log)
RFM_log_scaled_df= pd.DataFrame(RFM_log_scaled)
RFM_log_scaled_df.columns = ['recency', 'frequency', 'monetary']


kmeans= KMeans(n_clusters=6)
kmeans.fit(RFM_log_scaled_df)

RFM_log_scaled_df['Cluster']= kmeans.labels_
RFM_df4= RFM_df3.copy()
RFM_df4['Cluster'] = kmeans.labels_

# Customer segments in each cluster
df_cluster_segment = pd.merge(RFM_df2, RFM_df4, left_index=True, right_index=True)
df_cluster_segment1 = df_cluster_segment.drop(["Recency_x", "Frequency_x", "Monetary_x","Recency_y", "Frequency_y", "Monetary_y"], axis=1)

# 3D plot
fig = plt.figure(figsize=(18, 11))
ax = fig.add_subplot(111, projection='3d')

# Plot the filtered DataFrame
scatter = ax.scatter(RFM_df4['Recency'], RFM_df4['Frequency'], RFM_df4['Monetary'], c=RFM_df4['Cluster'], cmap='viridis')

ax.set_xlabel('Recency', fontsize=6)
ax.set_ylabel('Frequency', fontsize=6)
ax.set_zlabel('Monetary', fontsize=6)
ax.tick_params(axis='both', which='major', labelsize=5)  

# Create a legend for the colors
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters", prop={'size': 5})
ax.add_artist(legend1)
legend1.set_bbox_to_anchor((1.5, 1))

# DISTRIBUTION

colors = ['#9966cc', '#9775fa', '#7048e8', '#e0b0ff', '#bf80ff', '#330066']
# Calculate the percentage of customers in each cluster
cluster_percentage = (RFM_df4['Cluster'].value_counts(normalize=True) * 100).reset_index()
cluster_percentage.columns = ['Cluster', 'Percentage']
cluster_percentage.sort_values(by='Cluster', inplace=True)

figD, ax = plt.subplots(figsize=(15, 10))

# Create a pie chart
wedges, texts, autotexts = ax.pie(cluster_percentage['Percentage'], labels=cluster_percentage['Cluster'], colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85, textprops={'fontsize': 6})

# Draw a circle at the center of pie to make it look like a donut
centre_circle = plt.Circle((0,0),0.40,fc='white')
figD.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle.
ax.axis('equal')  
