
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.checkpoint
import numpy as np
import pickle 
import math
import copy
import pandas as pd
import datasets
import copy
from tqdm import tqdm

from collections import Counter
from datasets import load_dataset
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("/home/hpc/b207dd/b207dd11/test/Deu/labelled_data.csv")




def preprocess_scores(json_str):
    json_str = json_str.replace('false', 'False').replace('true', 'True')
    score_dict = eval(json_str)
    mean_score = sum(score_dict.values()) / len(score_dict)
    return mean_score

# Applying the preprocessing function to the relevant columns
df['anti_elitism'] = df['anti_elitism'].apply(preprocess_scores)
df['people_centrism'] = df['people_centrism'].apply(preprocess_scores)
df['left_wing'] = df['left_wing'].apply(preprocess_scores)
df['right_wing'] = df['right_wing'].apply(preprocess_scores)


df['speech_date'] = pd.to_datetime(df['speech_date'])
df['year_month'] = df['speech_date'].dt.to_period('M')
df['year'] = df['speech_date'].dt.to_period('Y')

df['text_length'] = df['text'].str.len()


weighted_avg = df.groupby(['faction', 'year']).apply(
    lambda x: (
        x[['anti_elitism', 'people_centrism', 'left_wing', 'right_wing']].multiply(x['text_length'], axis=0).sum() / 
        x['text_length'].sum()
    )
).reset_index()

weighted_avg.head()




# Creating the plot
plt.figure(figsize=(8,8))

# List of parties
parties = weighted_avg['faction'].unique()

# Dictionary for colors (for simplicity, using a colormap)
colors = plt.cm.get_cmap('tab20', len(parties))

# Dictionary for alphas (transparency) based on year
years = weighted_avg['year'].unique()
alphas = np.linspace(0.2, 1, len(years))

# Plot each party
for i, party in enumerate(parties):
    party_data = weighted_avg[weighted_avg['faction'] == party]
    for j, year in enumerate(years):
        year_data = party_data[party_data['year'] == year]
        if not year_data.empty:
            plt.scatter(year_data['left_wing'], year_data['right_wing'], color=colors(i), alpha=alphas[j], label=f'{party} {year}' if j == 0 else '')
            if j > 0:
                # Connect this point to the previous year's point
                prev_year_data = party_data[party_data['year'] == years[j-1]]
                if not prev_year_data.empty:
                    plt.plot([prev_year_data['left_wing'].values[0], year_data['left_wing'].values[0]], 
                             [prev_year_data['right_wing'].values[0], year_data['right_wing'].values[0]], 
                             color=colors(i), alpha=alphas[j])

# Adding labels and title
plt.xlabel('Left Wing')
plt.ylabel('Right Wing')
plt.title('Left-Right Political Spectrum of Parties Over Years')
plt.legend()
plt.grid(True)
file_path = '/home/hpc/b207dd/b207dd11/test/Deu/left_right_wing_scores_plot.png'
plt.savefig(file_path)
