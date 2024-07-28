import pandas as pd
import numpy as np
import torch
from sklearn.decomposition import PCA
import umap.umap_ as umap


import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

pio.templates.default = "plotly_white"



from tqdm import tqdm

from scipy.stats import ttest_ind
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder


# Load the DataFrame
WORK_DIR = "/home/hpc/b207dd/b207dd11/test/spin-politics"

model_name = "llama3-8b"








def calculate_scores(embedding, significant_features, mean_embeddings, axes, layer=None):
    if layer is None:
        filtered_embedding = {axis: embedding * significant_features[axis] for axis in axes}
        return {
            axis: np.dot(filtered_embedding[axis].flatten(), mean_embeddings[axis].flatten())
            for axis in axes
        }
    else:
        filtered_embedding = {axis: embedding * significant_features[axis][layer] for axis in axes}
        return {
            axis: np.dot(filtered_embedding[axis], mean_embeddings[axis][layer])
            for axis in axes
        }


def normalize_df(df, axes):
    mean = df[axes].mean()
    std = df[axes].std()
    for axis in axes:
        df[axis] = (df[axis] - mean[axis]) / std[axis]
    scale_factor = 3
    df[axes] /= scale_factor
    return df




def perform_lda(X, y, n_components=3):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    return lda.fit_transform(X, y_encoded)


def create_plot_data(df, axes, significant_features=None, mean_embeddings=None, is_layerwise=False, use_lda=False):
    plot_data = []
    num_layers = df.iloc[0]['Embedding'].shape[0] if is_layerwise else 1
    #
    for layer in tqdm(range(num_layers), desc="Processing layers"):
        layer_plot_data = []        
        if use_lda:
            X = np.array([row['Embedding'][layer].flatten() if is_layerwise else row['Embedding'].flatten() for _, row in df.iterrows()])
            y = df['Partei'].values
            X_lda = perform_lda(X, y)
        for i, (_, row) in enumerate(df.iterrows()):
            embedding = row['Embedding'][layer] if is_layerwise else row['Embedding']
            if use_lda:
                scores = {f"LDA{j+1}": X_lda[i, j] for j in range(3)}
            else:
                scores = calculate_scores(embedding, significant_features, mean_embeddings, axes, layer if is_layerwise else None)
            #
            layer_plot_data.append({
                "Speaker": row["Speaker"],
                "Party": row["Partei"],
                **scores,
                "EntryCount": row["EntryCount"],
                "Category": 'Speaker',
                "Layer": layer if is_layerwise else 0
            })
        #
        layer_df = pd.DataFrame(layer_plot_data)
        # Normalization
        layer_df = normalize_df(layer_df, axes)
        # Calculate centroids
        centroid_data = layer_df.groupby('Party').agg({
            **{axis: 'mean' for axis in axes},
            'EntryCount': 'sum'
        }).reset_index()
        centroid_data['Category'] = 'Party'
        centroid_data['Layer'] = layer if is_layerwise else 0
        #
        plot_data.extend(layer_df.to_dict('records'))
        plot_data.extend(centroid_data.to_dict('records'))
    return pd.DataFrame(plot_data)




def create_scatter_plot(df, axes, party_color_map, is_animated=False, ranges=None):
    if len(axes) == 2:
        plot_func = px.scatter
    elif len(axes) == 3:
        plot_func = px.scatter_3d
    else:
        raise ValueError("axes must contain either 2 or 3 elements")
    #
    plot_args = {
        'x': axes[0], 
        'y': axes[1],
        'color': 'Party',
        'size': 'EntryCount',
        'size_max': 64,
        'symbol': 'Category',
        'symbol_map': {'Party':'circle', 'Speaker':'square'},
        'opacity': 0.56,
        'color_discrete_map': party_color_map,
        'hover_data': ['Speaker'],
        'animation_frame': 'Layer' if is_animated else None,
    }
    if len(axes) == 3:
        plot_args['z'] = axes[2]
    if ranges:
        range_keys = ['range_x', 'range_y', 'range_z']
        for i, range_val in enumerate(ranges):
            if i < len(axes):  
                plot_args[range_keys[i]] = range_val
    #
    fig = plot_func(df, **plot_args)
    # layout
    layout_args = {
        'font_family': "Helvetica",
        'legend': dict(orientation="h"),
        'margin': dict(t=10)
    }
    if len(axes) == 3:
        layout_args['scene'] = dict(
            xaxis_title=axes[0],
            yaxis_title=axes[1],
            zaxis_title=axes[2],
            aspectmode='cube'
        )
    else:
        layout_args.update({
            'xaxis_title': axes[0],
            'yaxis_title': axes[1],
        })
    fig.update_layout(**layout_args)
    #
    if is_animated:
        fig["layout"].pop("updatemenus", None)
        if 'sliders' in fig['layout']:
            fig['layout']['sliders'][0]['pad'] = dict(t=140)
    return fig




df_tmp = df_speaker_mean_embeddings.sort_values(by='EntryCount', ascending=False).iloc[6:]


image_output_folder=f"{WORK_DIR}/real_speaker_img"

# Speakers Projected on Assigned Axes
axes = ["Libertarian", "Collectivist", "Progressive"]
ranges = [[-0.99, 0.99], [-0.99, 0.99], [-0.99, 0.99]]

plot_df = create_plot_data(df_tmp, axes, significant_features, mean_embeddings)

fig = create_scatter_plot(plot_df, axes, party_color_discrete_map, ranges=ranges)
file_path = f'{image_output_folder}/scatter_proj_all_layers.html'
fig.write_html(file_path)

## layer-wise 
plot_df_layerwise = create_plot_data(df_tmp, axes, significant_features, mean_embeddings, is_layerwise=True)

fig = create_scatter_plot(plot_df_layerwise, axes, party_color_discrete_map, is_animated=True, ranges=ranges)
file_path = f'{image_output_folder}/scatter_proj_animated.html'
fig.write_html(file_path)




# Speakers LDA 3D
axes_lda = ["LDA1", "LDA2", "LDA3"]
ranges = [[-1.49, 1.49], [-1.49, 1.49], [-1.49, 1.49]]

plot_df_lda = create_plot_data(df_tmp, axes_lda, use_lda=True)

fig_lda = create_scatter_plot(plot_df_lda, axes_lda, party_color_discrete_map, ranges=ranges)
file_path = f'{image_output_folder}/scatter_lda_3d_all_layers.html'
fig_lda.write_html(file_path)


## layer-wise
ranges = [[-0.99, 0.99], [-0.99, 0.99], [-0.99, 1.49]]

plot_df_lda_layerwise = create_plot_data(df_tmp, axes_lda, is_layerwise=True, use_lda=True)

fig_lda = create_scatter_plot(plot_df_lda_layerwise, axes_lda, party_color_discrete_map, is_animated=True, ranges=ranges)
image_output_folder=f"{WORK_DIR}/real_speaker_img"
file_path = f'{image_output_folder}/scatter_lda_3d_animated.html'
fig_lda.write_html(file_path)





# Speakers LDA 2D
axes_lda = ["LDA1", "LDA2"]
ranges = [[-1.49, 1.49], [-1.49, 1.49]]

plot_df_lda = create_plot_data(df_tmp, axes_lda, use_lda=True)

fig_lda = create_scatter_plot(plot_df_lda, axes_lda, party_color_discrete_map, ranges=ranges)
file_path = f'{image_output_folder}/scatter_lda_2d_all_layers.html'
fig_lda.write_html(file_path)


## layer-wise
ranges = [[-0.99, 0.99], [-0.99, 0.99]]

plot_df_lda_layerwise = create_plot_data(df_tmp, axes_lda, is_layerwise=True, use_lda=True)

fig_lda = create_scatter_plot(plot_df_lda_layerwise, axes_lda, party_color_discrete_map, is_animated=True, ranges=ranges)
image_output_folder=f"{WORK_DIR}/real_speaker_img"
file_path = f'{image_output_folder}/scatter_lda_2d_animated.html'
fig_lda.write_html(file_path)