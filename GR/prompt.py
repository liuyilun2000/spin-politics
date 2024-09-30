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
from transformers import LlamaTokenizer, LlamaModel, AutoModelForCausalLM
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.feature_selection import VarianceThreshold, f_classif

import matplotlib.pyplot as plt


tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', 
                                           cache_dir=TRANSFORMERS_CACHE_DIR, 
                                           token='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')

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", 
                                   cache_dir=TRANSFORMERS_CACHE_DIR, 
                                   device_map='sequential',
                                   token='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')

model.eval()

prompt = "Erarbeiten Sie bitte eine Reihe von Sätzen, die kollektivistische Werte widerspiegeln, um Kriterien für die kollektivistische und individualistische Dimension im Rahmen des politischen Koordinatensystems festzulegen."
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids.cuda(), max_new_tokens=1000)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


prompt = "Erarbeiten Sie bitte eine Reihe von Sätzen, die progressive Werte widerspiegeln, um Kriterien für die progressive und konservative Dimension im Rahmen des politischen Koordinatensystems festzulegen."
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids.cuda(), max_new_tokens=1000)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]