# utils.py

import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from typing import List, Callable, Optional
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder

def process_activations(
    sentences: List[str],
    model_name: str,
    output_dir: str,
    cache_dir: str = None,
    activation_type: str = "mlp",
    device: str = "cpu",
    batch_size: int = 1,
    skip_batch: int = 0,
) -> None:
    """
    Process sentences through a language model and store mean activations.

    Args:
    sentences (List[str]): List of sentences to process.
    model_name (str): Name of the pretrained model to use.
    cache_dir (str): Directory to cache the model.
    output_dir (str): Directory to save the activation tensors.
    activation_type (str): Type of activation to capture ('mlp' or 'attention').
    device (str): Device to run the model on ('cpu' or 'cuda').
    batch_size (int): Number of sentences to process at once.

    Returns:
    None
    """
    # Load model
    model = HookedTransformer.from_pretrained(model_name, cache_dir=cache_dir)
    model.eval()
    model.to(device)

    torch.set_grad_enabled(False)

    # Define hook function
    activations = []
    def store_activation(activation, hook):
        activations.append(activation.mean(dim=(0, 1)))  # Mean across batch and sequence length

    # Define hook filter
    if activation_type == "mlp":
        activation_filter = lambda name: ("mlp" in name) and ("hook_post" in name)
    elif activation_type == "attention":
        activation_filter = lambda name: ("attn" in name) and ("hook_post" in name)
    else:
        raise ValueError("activation_type must be 'mlp' or 'attention'")

    # Process sentences in batches
    for i in tqdm(range(0, len(sentences), batch_size)):
        if i < skip_batch:
            continue
        batch = sentences[i:i+batch_size]
        activations = []
        
        model.run_with_hooks(
            batch,
            return_type=None,
            fwd_hooks=[(activation_filter, store_activation)],
        )
        
        # Stack and save activations
        stacked_activations = torch.stack(activations)
        if batch_size==1:
            torch.save(stacked_activations.cpu(), f"{output_dir}/{i}.pt")
        else:
            for j, activation in enumerate(stacked_activations):
                torch.save(activation.cpu(), f"{output_dir}/{i*batch_size+j}.pt")
        
        torch.cuda.empty_cache()

def load_activations(output_dir: str, num_files: int) -> torch.Tensor:
    """
    Load saved activation tensors.

    Args:
    output_dir (str): Directory where activation tensors are saved.
    num_files (int): Number of files to load.

    Returns:
    torch.Tensor: Stacked tensor of all loaded activations.
    """
    activations = []
    for i in tqdm(range(num_files)):
        activation = torch.load(f"{output_dir}/{i}.pt")
        activations.append(activation)
    return torch.stack(activations)

# Example usage:
# process_activations(
#     sentences=df_speech_filtered_concat['Text'].tolist(),
#     model_name="meta-llama/Meta-Llama-3-8B",
#     cache_dir=TRANSFORMERS_CACHE_DIR,
#     output_dir=ACTIVATIONS_CACHE_DIR,
#     activation_type="mlp",
#     device="cuda" if torch.cuda.is_available() else "cpu",
#     batch_size=8
# )

# To load the activations later:
# all_activations = load_activations(ACTIVATIONS_CACHE_DIR, len(df_speech_filtered_concat))








def perform_lda_analysis(df: pd.DataFrame, 
                         n_components: int,
                         embedding_column: str,
                         party_column: str,
                         speaker_column: str,
                         entry_count_column: str,
                         additional_columns: List[str] = None) -> pd.DataFrame:
    """
    Perform LDA analysis on speaker embeddings across all layers.

    Args:
    df (pd.DataFrame): DataFrame containing speaker embeddings and metadata.
    n_components (int): Number of LDA components to compute.
    embedding_column (str): Name of the column containing embeddings.
    party_column (str): Name of the column containing party information.
    speaker_column (str): Name of the column containing speaker information.
    entry_count_column (str): Name of the column containing entry counts.
    additional_columns (List[str]): List of additional columns to include in the output.

    Returns:
    pd.DataFrame: DataFrame with LDA projections for each layer and speaker/party.
    """
    if additional_columns is None:
        additional_columns = []

    num_layers = df.iloc[0][embedding_column].shape[0]
    plot_data = []

    for layer in tqdm(range(num_layers), desc="Processing layers"):
        layer_plot_data = []
        
        # Prepare data for LDA
        X = np.array([row[embedding_column][layer].flatten() for _, row in df.iterrows()])
        y = df[party_column].values
        
        # Encode party labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Perform LDA
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_lda = lda.fit_transform(X, y_encoded)
        
        # Create plot data
        for i, (_, row) in enumerate(df.iterrows()):
            data_point = {
                party_column: row[party_column],
                speaker_column: row[speaker_column],
                entry_count_column: row[entry_count_column],
                "Symbol": 'Speaker',
                "Layer": layer
            }
            for j in range(n_components):
                data_point[f"LDA{j+1}"] = X_lda[i, j]
            for col in additional_columns:
                data_point[col] = row[col]
            layer_plot_data.append(data_point)

        # Calculate centroids for each party
        agg_dict = {entry_count_column: 'sum'}
        for j in range(n_components):
            agg_dict[f"LDA{j+1}"] = 'mean'
        
        centroid_data = pd.DataFrame(layer_plot_data).groupby(party_column).agg(agg_dict).reset_index()
        centroid_data[speaker_column] = centroid_data[party_column]
        centroid_data['Symbol'] = 'Party'
        centroid_data['Layer'] = layer

        # Append data to plot_data
        plot_data.extend(layer_plot_data)
        plot_data.extend(centroid_data.to_dict('records'))

    return pd.DataFrame(plot_data)

# Example usage:
# lda_df = perform_lda_analysis(df_speaker_mean_embeddings, 
#                               n_components=3,
#                               embedding_column='Embedding',
#                               party_column='Partei',
#                               speaker_column='Speaker',
#                               entry_count_column='EntryCount',
#                               additional_columns=['SpeakerStatus', 'Religion'])