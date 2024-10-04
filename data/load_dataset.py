
import torch
from typing import List, Callable, Optional
from collections import Counter


import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns



from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModel



'''
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", 
    token='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')
llama3_model = AutoModel.from_pretrained("meta-llama/Meta-Llama-3-8B", 
    #device_map='cpu',
    token='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')
'''


from datasets import load_dataset



# Function to load and process a dataset
def load_and_process_dataset(dataset):
    ds = load_dataset("classla/ParlaSent", dataset)['train']
    df = pd.DataFrame(ds)
    df['dataset'] = dataset
    return df[['dataset', 'sentence', 'country', 'date', 'name', 'party', 'gender', 'birth_year', 'ruling']]

# Load and process all datasets
datasets = ['EN', 'EN_additional_test', 'BCS', 'BCS_additional_test', 'CZ', 'SK', 'SL']

dfs = [load_and_process_dataset(dataset) for dataset in tqdm(datasets)]

# Combine all dataframes
df = pd.concat(dfs, ignore_index=True)




# Calculate sentence lengths
df['sentence_length'] = df['sentence'].str.len()

# Create a figure and axis
plt.figure(figsize=(8, 4))

# Create the histogram
sns.histplot(data=df, x='sentence_length', bins=50, kde=True)

# Set labels and title
plt.xlabel('Sentence Length (characters)')
plt.ylabel('Frequency')
plt.title('Distribution of Sentence Lengths')

# Add some statistics to the plot
mean_length = df['sentence_length'].mean()
median_length = df['sentence_length'].median()

plt.axvline(mean_length, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_length:.2f}')
plt.axvline(median_length, color='g', linestyle='dashed', linewidth=2, label=f'Median: {median_length:.2f}')

plt.legend()

# Adjust layout
plt.tight_layout()

# Save the figure to a file
plt.savefig('sentence_length_distribution.png', dpi=300, bbox_inches='tight')

# Close the plot to free up memory
plt.close()






from utils import process_activations, load_activations


ACTIVATIONS_CACHE_DIR='data/cache/ParlaSent'
MODEL_NAME="meta-llama/Meta-Llama-3-8B"
# Process the activations
process_activations(
    sentences=df['sentence'].tolist(),
    model_name=MODEL_NAME,
    output_dir=ACTIVATIONS_CACHE_DIR,
    activation_type="mlp",
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=1
)

# Later, when you want to load the activations:
all_activations = load_activations(ACTIVATIONS_CACHE_DIR, len(df_speech_filtered_concat))