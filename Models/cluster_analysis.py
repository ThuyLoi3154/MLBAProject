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

import plotly.express as px
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import StandardScaler
from matplotlib.backends.backend_qt5agg import FigureCanvas
from plotly.subplots import make_subplots

from Models.processdata import merged_df
from Models.RFMcalculation import RFM_df2
from Models.kmean import RFM_df4

df_customer = RFM_df4.copy()

# Standardize the data (excluding the cluster column)
scaler = StandardScaler()
df_customer_standardized = scaler.fit_transform(df_customer.drop(columns=['Cluster'], axis=1))

# Create a new dataframe with standardized values and add the cluster column back
df_customer_standardized = pd.DataFrame(df_customer_standardized, columns=df_customer.columns[:-1], index=df_customer.index)
df_customer_standardized['Cluster'] = df_customer['Cluster']

# Calculate the centroids of each cluster
cluster_centroids = df_customer_standardized.groupby('Cluster').mean()

# Function to create a radar chart
def create_radar_chart(ax, angles, data, color, cluster):
    # Plot the data and fill the area
    ax.fill(angles, data, color=color, alpha=0.4)
    ax.plot(angles, data, color=color, linewidth=1, linestyle='solid')
    
    # Add a title
    ax.text(1.8, 0.5, f'Cluster {cluster}', size=6, color=color,  ha='right', transform=ax.transAxes)
    # Change the size of the tick labels
    ax.tick_params(axis='both', which='major', labelsize=4)

# Set data
labels=np.array(cluster_centroids.columns)
num_vars = len(labels)

# Compute angle of each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The plot is circular, so we need to "complete the loop" and append the start to the end
labels = np.concatenate((labels, [labels[0]]))
angles += angles[:1]

# Initialize the figure
figR, ax = plt.subplots(figsize=(12, 6), subplot_kw=dict(polar=True), nrows=2, ncols=3)

# Flatten the ax array
ax = ax.flatten()
colors = ['#9966cc', '#9775fa', '#7048e8', '#e0b0ff', '#bf80ff', '#330066']
# Create radar chart for each cluster
for i, color in enumerate(colors):
    data = cluster_centroids.loc[i].tolist()
    data += data[:1]  # Complete the loop
    create_radar_chart(ax[i], angles, data, color, i)

# Add input data
ax[0].set_xticks(angles[:-1])
ax[0].set_xticklabels(labels[:-1], fontsize=5)

ax[1].set_xticks(angles[:-1])
ax[1].set_xticklabels(labels[:-1], fontsize=5)

ax[2].set_xticks(angles[:-1])
ax[2].set_xticklabels(labels[:-1], fontsize=5)

ax[3].set_xticks(angles[:-1])
ax[3].set_xticklabels(labels[:-1], fontsize=5)

ax[4].set_xticks(angles[:-1])
ax[4].set_xticklabels(labels[:-1], fontsize=5)

ax[5].set_xticks(angles[:-1])
ax[5].set_xticklabels(labels[:-1], fontsize=5)

# Add a grid
ax[0].grid(color='grey', linewidth=0.5)
ax[1].grid(color='grey', linewidth=0.5)
# Display the plot
plt.tight_layout()

# HISTOGRAM
# Plot histograms for each feature segmented by the clusters
features = RFM_df4.columns[0:-1]
clusters = RFM_df4['Cluster'].unique()
clusters.sort()

# Setting up the subplots
n_rows = len(features)
n_cols = len(clusters)
figH, axes = plt.subplots(n_rows, n_cols, figsize=(8, 1.5*n_rows))

# Plotting histograms
for i, feature in enumerate(features):
    for j, cluster in enumerate(clusters):
        data = RFM_df4[RFM_df4['Cluster'] == cluster][feature]
        axes[i, j].hist(data, bins=20, color=colors[j], edgecolor='w', alpha=0.7)
        axes[i, j].set_title(f'Cluster {cluster} - {feature}', fontsize=6)
        axes[i, j].set_xlabel('')
        axes[i, j].set_ylabel('')
        axes[i, j].tick_params(axis='both', which='major', labelsize=4)
        axes[i, j].xaxis.labelpad = -1
        axes[i, j].yaxis.labelpad = -1

# Adjusting layout to prevent overlapping
plt.tight_layout()