TRANSFORMERS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/cache/huggingface/transformers"
DATASETS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/cache/huggingface/datasets"
WORK_DIR = '/home/atuin/b207dd/b207dd11/'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.checkpoint
import numpy as np
import pandas as pd
import pickle 
import math
import copy
import pandas as pd
import datasetsdf
import copy
from tqdm import tqdm

from collections import Counter
from transformers import LlamaTokenizer, LlamaModel
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import VarianceThreshold, f_classif

from scipy.spatial import distance


import matplotlib.pyplot as plt
import seaborn as sns





df = pd.read_csv("/home/hpc/b207dd/b207dd11/test/Deu/labelled_data.csv")

batch_output_dir=WORK_DIR+'test/DEU/llama2-7b/text'
all_pooled_hs = []
for batch_id in tqdm(range(440)):
    with open(batch_output_dir+'/all_pooled_hs_'+str(batch_id)+'.pkl', 'rb') as f:
        batch_all_pooled_hs = pickle.load(f)
    all_pooled_hs.append(batch_all_pooled_hs)

all_pooled_hs = torch.cat(all_pooled_hs, dim=1)
all_pooled_hs = all_pooled_hs.permute(1,0,2,3)
hs = []
for i in range(all_pooled_hs.shape[0]):
    hs.append(all_pooled_hs[i,:,:,0])

all_pooled_act = []
for batch_id in tqdm(range(440)):
    with open(batch_output_dir+'/all_pooled_act_'+str(batch_id)+'.pkl', 'rb') as f:
        batch_all_pooled_activations = pickle.load(f)
    all_pooled_act.append(batch_all_pooled_activations)

all_pooled_act = torch.cat(all_pooled_act, dim=1)
all_pooled_act = all_pooled_act.permute(1,0,2,3)
act_max = []
act_avg = []
for i in range(all_pooled_act.shape[0]):
    act_max.append(all_pooled_act[i,:,:,0])
    act_avg.append(all_pooled_act[i,:,:,1])


df['hs'] = hs
df['act_max'] = act_max
df['act_avg'] = act_avg


# Assuming 'faction' is already a categorical variable. If not, you'll need to encode it
# Let's encode the factions as integers if they aren't already
if df['faction'].dtype == 'object':
    faction_labels = df['faction'].astype('category').cat.codes
else:
    faction_labels = df['faction']

df['faction'].value_counts()
factions = list(dict(df['faction'].value_counts()).keys())









batch_output_dir=WORK_DIR+'test/DEU/llama2-7b/text_kollektiv'
all_pooled_hs = []
for batch_id in tqdm(range(1)):
    with open(batch_output_dir+'/all_pooled_hs_'+str(batch_id)+'.pkl', 'rb') as f:
        batch_all_pooled_hs = pickle.load(f)
    all_pooled_hs.append(batch_all_pooled_hs)

all_pooled_hs = torch.cat(all_pooled_hs, dim=1)
all_pooled_hs = all_pooled_hs.permute(1,0,2,3)
hs_kollektiv = all_pooled_hs[:,:,:,0].reshape(all_pooled_hs.shape[0], -1)

all_pooled_act = []
for batch_id in tqdm(range(1)):
    with open(batch_output_dir+'/all_pooled_act_'+str(batch_id)+'.pkl', 'rb') as f:
        batch_all_pooled_activations = pickle.load(f)
    all_pooled_act.append(batch_all_pooled_activations)

all_pooled_act = torch.cat(all_pooled_act, dim=1)
all_pooled_act = all_pooled_act.permute(1,0,2,3)
act_max_kollektiv = all_pooled_act[:,:,:,0].reshape(all_pooled_act.shape[0], -1)
act_avg_kollektiv = all_pooled_act[:,:,:,1].reshape(all_pooled_act.shape[0], -1)


batch_output_dir=WORK_DIR+'test/DEU/llama2-7b/text_individual'
all_pooled_hs = []
for batch_id in tqdm(range(1)):
    with open(batch_output_dir+'/all_pooled_hs_'+str(batch_id)+'.pkl', 'rb') as f:
        batch_all_pooled_hs = pickle.load(f)
    all_pooled_hs.append(batch_all_pooled_hs)

all_pooled_hs = torch.cat(all_pooled_hs, dim=1)
all_pooled_hs = all_pooled_hs.permute(1,0,2,3)
hs_individual = all_pooled_hs[:,:,:,0].reshape(all_pooled_hs.shape[0], -1)

all_pooled_act = []
for batch_id in tqdm(range(1)):
    with open(batch_output_dir+'/all_pooled_act_'+str(batch_id)+'.pkl', 'rb') as f:
        batch_all_pooled_activations = pickle.load(f)
    all_pooled_act.append(batch_all_pooled_activations)

all_pooled_act = torch.cat(all_pooled_act, dim=1)
all_pooled_act = all_pooled_act.permute(1,0,2,3)
act_max_individual = all_pooled_act[:,:,:,0].reshape(all_pooled_act.shape[0], -1)
act_avg_individual = all_pooled_act[:,:,:,1].reshape(all_pooled_act.shape[0], -1)







act_max_kollektiv_centroid = torch.mean(act_max_kollektiv, axis=0)
act_max_individual_centroid = torch.mean(act_max_individual, axis=0)



act_max_axis = act_max_kollektiv_centroid - act_max_individual_centroid
act_max_axis_norm = act_max_axis / np.linalg.norm(act_max_axis)















# Step 1: Flatten the tensors in the 'act' column to 1D arrays
flattened_act = df['act_max'].apply(lambda x: x.flatten())

# Step 2: Stack these arrays into a 2D array
act_matrix = torch.stack(flattened_act.tolist()).numpy()


projections = np.dot(act_matrix, act_max_axis_norm)
max_proj_value = np.max(np.abs(projections))
normalized_projections = projections / max_proj_value

df['normalized_projections'] = normalized_projections



# Assuming act_max_kollektiv_centroid and act_max_individual_centroid are your centroids
# and act_matrix is already created

# Calculate distances to both centroids
dist_to_kollektiv = [distance.euclidean(act_vector, act_max_kollektiv_centroid) for act_vector in act_matrix]
dist_to_individual = [distance.euclidean(act_vector, act_max_individual_centroid) for act_vector in act_matrix]

# Add these distances to your DataFrame
df['dist_to_kollektiv'] = dist_to_kollektiv
df['dist_to_individual'] = dist_to_individual




dates = pd.to_datetime(df['speech_date'])
date_min, date_max = dates.min(), dates.max()
transparency = (dates - date_min) / (date_max - date_min) * 0.5

colors = {
    'DIE LINKE.': 'purple', 
    'CDU/CSU': 'black', 
    'AfD': 'blue', 
    'SPD': 'red', 
    'Grüne': 'green', 
    'FDP': 'yellow', 
    'Fraktionslos': 'white'
    }
# Now plot using these new LDA dimensions
fig, ax = plt.subplots()
for faction, color in colors.items():
    faction_data = df[df['faction'] == faction]
    ax.scatter(faction_data['dist_to_kollektiv'], faction_data['dist_to_individual'], 
               color=color, 
               alpha=transparency, 
               label=faction)

ax.legend()
plt.xlabel('dist_to_kollektiv Dimension 1')
plt.ylabel('dist_to_individual Dimension 2')
plt.title('Results with Faction Separation')

file_path = '/home/hpc/b207dd/b207dd11/test/Deu/img/standard_act_PCA-2_scores_plot.png'
plt.savefig(file_path)



# Create a melted DataFrame for easy plotting with Seaborn
melted_df = df.melt(id_vars=['faction'], value_vars=['dist_to_kollektiv', 'dist_to_individual'],
                    var_name='Distance Type', value_name='Distance')

# Create a boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='faction', y='Distance', hue='Distance Type', data=melted_df)
plt.title('Boxplot of Distances to Kollektiv and Individual for Each Faction')
file_path = '/home/hpc/b207dd/b207dd11/test/Deu/img/standard_act_dist.png'
plt.savefig(file_path)








## LR
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch


# Prepare the data
X = torch.cat((act_max_kollektiv, act_max_individual), 0)
y = np.array([0] * len(act_max_kollektiv) + [1] * len(act_max_individual))  # 0 for kollektiv, 1 for individual

# Convert to numpy for compatibility with scikit-learn
X = X.numpy()

# Optional: Dimensionality Reduction (e.g., PCA)
# from sklearn.decomposition import PCA
# pca = PCA(n_components=500)  # for example, 500 components
# X = pca.fit_transform(X)

# 4-Fold Cross-Validation
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Predict and Evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print("Fold MSE:", mse)

trained_model = model

# Step 1: Flatten the tensors in the 'act' column to 1D arrays
#flattened_act = df['act_max'].apply(lambda x: x.flatten())
# Step 2: Stack these arrays into a 2D array
#act_matrix = torch.stack(flattened_act.tolist()).numpy()


# Use the trained model to score each item
df['scores'] = trained_model.predict(act_matrix)


# Create a boxplot
plt.figure(figsize=(6, 16))
sns.boxplot(x='faction', y='scores', data=df, palette=colors)
plt.title('Boxplot of LR Scores for Each Faction')

file_path = '/home/hpc/b207dd/b207dd11/test/Deu/img/standard_act_lr_scores_plot.png'
plt.savefig(file_path)
