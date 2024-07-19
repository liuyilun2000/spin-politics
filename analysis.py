import pandas as pd
import numpy as np
import torch
from sklearn.decomposition import PCA
import umap.umap_ as umap


import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

from tqdm import tqdm

# Load the DataFrame
df = pd.read_pickle("/home/hpc/b207dd/b207dd11/test/spin-politics/standard_text_with_embeddings.pkl")
WORK_DIR = "/home/hpc/b207dd/b207dd11/test/spin-politics"


# Define a function to convert categorical values to RGB values
def category_to_rgb(libertarian, collectivist, progressive):
    color_map = {
        "Neutral": 128,
        "Libertär": 225,
        "Restriktiv": 30,
        "Kollektivistisch": 225,
        "Individualistisch": 30,
        "Progressiv": 225,
        "Konservativ": 30
    }
    r = color_map[libertarian]
    g = color_map[collectivist]
    b = color_map[progressive]
    return f'rgb({r},{g},{b})'

df['Ideology'] = df['Libertarian'] + ' - ' + df['Collectivist'] + ' - ' + df['Progressive']
df['Color'] = df.apply(lambda row: category_to_rgb(row['Libertarian'], row['Collectivist'], row['Progressive']), axis=1)
# Create a color discrete map
unique_ideologies = df['Ideology'].unique()
color_discrete_map = {ideology: color for ideology, color in zip(unique_ideologies, df['Color'].unique())}


def pca_umap_plot(df, 
                  pca_cumul_var_ratio_thresh=None,
                  pca_n_components=None, 
                  umap_n_components=3, 
                  umap_n_neighbors=15, 
                  umap_min_dist=0.1, 
                  layer_list=None,
                  image_output_folder=""):
    embeddings = np.stack(df['Embedding'].values)
    n_layers = embeddings.shape[1]
    if layer_list is None:
        layer_list = range(n_layers)
    #
    for layer in tqdm(layer_list, desc="Processing layers"):
        layer_embeddings = embeddings[:, layer, :]
        # Standardize embeddings
        mean = np.mean(layer_embeddings, axis=0)
        std = np.std(layer_embeddings, axis=0)
        standardized_embeddings = (layer_embeddings - mean) / std
        # PCA
        if pca_n_components:
            num_components = pca_n_components
            pca = PCA(n_components=num_components)
            pca_embeddings = pca.fit_transform(standardized_embeddings)
        elif pca_cumul_var_ratio_thresh:
            pca = PCA()
            pca.fit(standardized_embeddings)
            cumul_var_ratio = np.cumsum(pca.explained_variance_ratio_)
            num_components = np.where(cumul_var_ratio >= pca_cumul_var_ratio_thresh)[0][0] + 1
            if num_components < umap_n_components:
                num_components = umap_n_components
            pca_embeddings = pca.transform(standardized_embeddings)[:, :num_components]
        print(f"Layer {layer} PCA num_components {num_components}")
        # UMAP
        if umap_n_components < num_components:
            umap_reducer = umap.UMAP(n_components=umap_n_components, 
                                    n_neighbors=umap_n_neighbors, 
                                    min_dist=umap_min_dist, 
                                    metric='cosine', 
                                    random_state=42)
            umap_embeddings = umap_reducer.fit_transform(pca_embeddings)
        else:
            umap_embeddings = pca_embeddings
        # df for plotting
        plot_df = df.copy()
        plot_df['UMAP1'] = umap_embeddings[:, 0]
        plot_df['UMAP2'] = umap_embeddings[:, 1]
        plot_df['UMAP3'] = umap_embeddings[:, 2] if umap_n_components == 3 else np.zeros(umap_embeddings.shape[0])
        # Plotting
        fig = px.scatter_3d(
            plot_df, x='UMAP1', y='UMAP2', z='UMAP3', 
            color='Ideology', 
            color_discrete_map=color_discrete_map,
            #color_continuous_scale=px.colors.sequential.Rainbow_r,
            #color_discrete_sequence=px.colors.sequential.Rainbow_r,
            title=f'Layer {layer} Embeddings',
            hover_data={
                'Libertarian': True,
                'Collectivist': True,
                'Progressive': True,
                'Topic': True,
                #'Response': True
            }
        )
        file_path = f'{image_output_folder}/scatter_layer{layer}_pca{num_components}_umap{umap_n_components}.html'
        fig.write_html(file_path)
        plt.close()
        plt.clf()

# Call the function
pca_umap_plot(
    df, 
    #pca_cumul_var_ratio_thresh=.5,
    pca_n_components=3, 
    umap_n_components=3, 
    umap_n_neighbors=100, 
    umap_min_dist=0.8, 
    #layer_list=[i for i in range(3)],
    image_output_folder=f"{WORK_DIR}/standard_img"
)













def calculate_mean_embeddings(df, axis):
    type_a_label = axis["type_a"]
    type_b_label = axis["type_b"]    
    type_a_embeddings = np.mean(np.stack(df[df[axis["column"]] == type_a_label]['Embedding'].values), axis=0)
    type_b_embeddings = np.mean(np.stack(df[df[axis["column"]] == type_b_label]['Embedding'].values), axis=0)
    return type_a_embeddings, type_b_embeddings

# Define the axes
axes = {
    "Libertarian": {"column": "Libertarian", "type_a": "Libertär", "type_b": "Restriktiv"},
    "Collectivist": {"column": "Collectivist", "type_a": "Kollektivistisch", "type_b": "Individualistisch"},
    "Progressive": {"column": "Progressive", "type_a": "Progressiv", "type_b": "Konservativ"}
}



# Calculate mean embeddings for each axis
mean_embeddings = {}
for axis_name, axis in axes.items():
    mean_embeddings[axis_name] = calculate_mean_embeddings(df, axis)

# Create a new DataFrame for plotting
plot_data = []

for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
    embedding = row['Embedding']
    libertarian_score = np.dot(embedding.flatten(), mean_embeddings["Libertarian"][0].flatten()) - np.dot(embedding.flatten(), mean_embeddings["Libertarian"][1].flatten())
    collectivist_score = np.dot(embedding.flatten(), mean_embeddings["Collectivist"][0].flatten()) - np.dot(embedding.flatten(), mean_embeddings["Collectivist"][1].flatten())
    progressive_score = np.dot(embedding.flatten(), mean_embeddings["Progressive"][0].flatten()) - np.dot(embedding.flatten(), mean_embeddings["Progressive"][1].flatten())
    #
    plot_data.append({
        'Libertarian': row['Libertarian'],
        'Collectivist': row['Collectivist'],
        'Progressive': row['Progressive'],
        'Topic': row['Topic'],
        "Libertarian": libertarian_score,
        "Collectivist": collectivist_score,
        "Progressive": progressive_score,
        "Ideology": row["Ideology"]
    })

plot_df = pd.DataFrame(plot_data)

# Plotting

fig = px.scatter_3d(
    plot_df, x='Libertarian', y='Collectivist', z='Progressive', 
    color='Ideology', 
    color_discrete_map=color_discrete_map,
    #color_continuous_scale=px.colors.sequential.Rainbow_r,
    #color_discrete_sequence=px.colors.sequential.Rainbow_r,
    title=f'Manual Axes Embeddings',
    hover_data={
        'Libertarian': True,
        'Collectivist': True,
        'Progressive': True,
        'Topic': True,
        #'Response': True
    }
)



image_output_folder=f"{WORK_DIR}/standard_img"
file_path = f'{image_output_folder}/scatter_manual.html'
fig.write_html(file_path)
