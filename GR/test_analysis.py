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

import matplotlib.pyplot as plt




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






## PCA @ hs
# Step 1: Flatten the tensors in the 'hs' column to 1D arrays
flattened_hs = df['hs'].apply(lambda x: x.flatten())

# Step 2: Stack these arrays into a 2D array
hs_matrix = torch.stack(flattened_hs.tolist()).numpy()

# Step 3: Apply PCA
# You can choose the number of components based on your analysis needs
n_components = 2  # For example, reducing to 2 components
pca = PCA(n_components=n_components)
reduced_hs = pca.fit_transform(hs_matrix)


df['reduced_hs'] = list(reduced_hs)



dates = pd.to_datetime(df['speech_date'])
date_min, date_max = dates.min(), dates.max()
transparency = (dates - date_min) / (date_max - date_min)

# Add the reduced_hs dimensions to the DataFrame
df['hs_dim1'] = reduced_hs[:, 0]
df['hs_dim2'] = reduced_hs[:, 1]

# Plot
fig, ax = plt.subplots()

# Colors for each faction - replace with actual factions and desired colors
colors = {
    'DIE LINKE.': 'magenta', 
    'CDU/CSU': 'black', 
    'AfD': 'blue', 
    'SPD': 'red', 
    'Grüne': 'green', 
    'FDP': 'yellow', 
    'Fraktionslos': '#aaa'
    }

for faction, color in colors.items():
    faction_data = df[df['faction'] == faction]
    ax.scatter(faction_data['hs_dim1'], faction_data['hs_dim2'], 
               color=color, alpha=transparency, label=faction)

ax.legend()
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('PCA Results with Faction Colors and Date Transparency')

file_path = '/home/hpc/b207dd/b207dd11/test/Deu/hs_PCA-2_scores_plot.png'
plt.savefig(file_path)







## LDA @ hs
# Initialize LDA
lda = LDA(n_components=2)

# Fit LDA to the high-dimensional data and transform it
lda_hs = lda.fit_transform(hs_matrix, faction_labels)


# Add the LDA dimensions to the DataFrame for plotting
df['lda_dim1'] = lda_hs[:, 0]
df['lda_dim2'] = lda_hs[:, 1]











## LDA @ act_max
# Step 1: Flatten the tensors in the 'act' column to 1D arrays
flattened_act = df['act_max'].apply(lambda x: x.flatten())

# Step 2: Stack these arrays into a 2D array
act_matrix = torch.stack(flattened_act.tolist()).numpy()


# Initialize the VarianceThreshold object
selector = VarianceThreshold()

# Fit the selector to the data
selector.fit(act_matrix)

# Get the variances
variances = selector.variances_

# Sort features by variance
sorted_features = np.argsort(variances)[::-1]


plt.figure(figsize=(10, 5))
plt.plot(sorted_features)
plt.yscale('log')  # Use logarithmic scale to better visualize the differences
plt.title('Feature Variances')
plt.xlabel('Feature index (sorted by variance)')
plt.ylabel('Variance (log scale)')
plt.grid(True)
file_path = '/home/hpc/b207dd/b207dd11/test/Deu/act-max_variance_scores_plot.png'
plt.savefig(file_path)

# Select the top N features - you need to decide on N
N = 8000  # for example, adjust N based on your needs and resources
top_features = sorted_features[:N]

# Reduce your dataset to these top features
reduced_act_matrix = act_matrix[:, top_features]


# Initialize LDA
lda = LDA(n_components=2)

# Fit LDA to the high-dimensional data and transform it
lda_act = lda.fit_transform(reduced_act_matrix, faction_labels)


# Add the LDA dimensions to the DataFrame for plotting
df['lda_dim1'] = lda_act[:, 0]
df['lda_dim2'] = lda_act[:, 1]



transparency = 0.25
# Now plot using these new LDA dimensions
fig, ax = plt.subplots(figsize=(10, 10))
for faction, color in colors.items():
    faction_data = df[df['faction'] == faction]
    sizes = ((pd.to_datetime(faction_data['speech_date']) - date_min) / (date_max - date_min) * 20).values
    ax.scatter(faction_data['lda_dim1'], faction_data['lda_dim2'], 
               color=color, 
               alpha=transparency, 
               label=faction,
               s=sizes)

ax.legend()
plt.xlabel('LDA Dimension 1')
plt.ylabel('LDA Dimension 2')
plt.title('LDA Results with Faction Separation')

file_path = '/home/hpc/b207dd/b207dd11/test/Deu/act-max_LDA-2_scores_plot.png'
plt.savefig(file_path)








## F-statistic

f_values, p_values = f_classif(act_matrix, df['faction'])

# Sort the features by their F-value in descending order
sorted_indices = np.argsort(f_values)[::-1]

# Now you can select the top N features based on F-value
# N needs to be determined based on your needs, for example:
N = 50
top_features = sorted_indices[:N]
selected_features = act_matrix[:, top_features]





# Add the LDA dimensions to the DataFrame for plotting
df['f_1'] = selected_features[:, 1]
df['f_2'] = selected_features[:, 2]



transparency = (dates - date_min) / (date_max - date_min) * 0.5
# Now plot using these new LDA dimensions
fig, ax = plt.subplots()
for faction, color in colors.items():
    faction_data = df[df['faction'] == faction]
    ax.scatter(faction_data['f_1'], faction_data['f_2'], 
               color=color, alpha=transparency, label=faction)

ax.legend()
plt.xlabel('F-statistic Dimension 1')
plt.ylabel('F-statistic Dimension 2')
plt.title('LDA Results with Faction Separation')

file_path = '/home/hpc/b207dd/b207dd11/test/Deu/act-max_F-2_scores_plot.png'
plt.savefig(file_path)








N = 1000
top_features = sorted_indices[:N]
selected_features = act_matrix[:, top_features]

# Initialize LDA
lda = LDA(n_components=2)

# Fit LDA to the high-dimensional data and transform it
lda_act = lda.fit_transform(selected_features, faction_labels)


# Add the LDA dimensions to the DataFrame for plotting
df['lda_dim1'] = lda_act[:, 0]
df['lda_dim2'] = lda_act[:, 1]



transparency = (dates - date_min) / (date_max - date_min) * 0.5
# Now plot using these new LDA dimensions
fig, ax = plt.subplots()
for faction, color in colors.items():
    faction_data = df[df['faction'] == faction]
    ax.scatter(faction_data['lda_dim1'], faction_data['lda_dim2'], 
               color=color, 
               alpha=transparency, 
               label=faction)

ax.legend()
plt.xlabel('LDA Dimension 1')
plt.ylabel('LDA Dimension 2')
plt.title('LDA Results with Faction Separation')
file_path = '/home/hpc/b207dd/b207dd11/test/Deu/act-max_F-50+LDA-2_scores_plot.png'
plt.savefig(file_path)









N = 10
top_features = sorted_indices[:N]
selected_features = act_matrix[:, top_features]
# Step 3: Apply PCA
# You can choose the number of components based on your analysis needs
n_components = 2  # For example, reducing to 2 components
pca = PCA(n_components=n_components)
reduced_act = pca.fit_transform(selected_features)

df['pca_dim1'] = reduced_act[:, 0]
df['pca_dim2'] = reduced_act[:, 1]

transparency = (dates - date_min) / (date_max - date_min) * 0.5
# Now plot using these new LDA dimensions
fig, ax = plt.subplots()
for faction, color in colors.items():
    faction_data = df[df['faction'] == faction]
    ax.scatter(faction_data['pca_dim1'], faction_data['pca_dim2'], 
               color=color, 
               alpha=transparency, 
               label=faction)

ax.legend()
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.title('PCA Results with Faction Separation')
file_path = '/home/hpc/b207dd/b207dd11/test/Deu/act-max_F-50+PCA-2_scores_plot.png'
plt.savefig(file_path)
