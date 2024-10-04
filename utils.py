# utils.py

import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
from typing import List, Callable, Optional

def process_activations(
    sentences: List[str],
    model_name: str,
    output_dir: str,
    cache_dir: str = None,
    activation_type: str = "mlp",
    device: str = "cpu",
    batch_size: int = 1
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
        batch = sentences[i:i+batch_size]
        activations = []
        
        model.run_with_hooks(
            batch,
            return_type=None,
            fwd_hooks=[(activation_filter, store_activation)],
        )
        
        # Stack and save activations
        stacked_activations = torch.stack(activations)
        for j, activation in enumerate(stacked_activations):
            torch.save(activation.cpu(), f"{output_dir}/{i+j}.pt")
        
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
    for i in range(num_files):
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