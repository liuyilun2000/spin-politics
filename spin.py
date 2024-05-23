from transformers import AutoTokenizer, AutoModel

import datetime
import pandas as pd
import numpy as np

import json
import xml.etree.ElementTree as ET

from tqdm.auto import tqdm

import plotly.express as px
import plotly.graph_objs as go

import torch



TRANSFORMERS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/cache/huggingface/transformers"


df_speech_filtered_concat = pd.read_pickle('df_speech_filtered_concat.pkl')[:1000]


res = []
index = 0
ACTIVATIONS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/test/DEU/activations/llama3-8b"
for sentence in tqdm(df_speech_filtered_concat['Text']):
    res.append(torch.load(ACTIVATIONS_CACHE_DIR+f'/{index}.pt'))
    index += 1

res = torch.stack(res)

num_layer = res.shape[1]
for layer in tqdm(range(num_layer)):
    
    res[:,]
