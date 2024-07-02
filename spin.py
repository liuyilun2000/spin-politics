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

from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, f1_score, r2_score, classification_report

import plotly.express as px

import importlib
import warnings
from collections import defaultdict
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)




TRANSFORMERS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/cache/huggingface/transformers"


df_speech_filtered_concat = pd.read_pickle('df_speech_filtered_concat.pkl')#[:1000]


res = []
index = 0
ACTIVATIONS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/test/DEU/activations/llama3-8b"
for sentence in tqdm(df_speech_filtered_concat['Text']):
    res.append(torch.load(ACTIVATIONS_CACHE_DIR+f'/{index}.pt'))
    index += 1

res = torch.stack(res)[:10000]

label_encoder = LabelEncoder()
y_text = df_speech_filtered_concat['Partei'][:10000]
y = label_encoder.fit_transform(y_text)
y = torch.tensor(y, dtype=torch.long).cpu()


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        outputs = self.linear(x)
        return outputs


num_layer = res.shape[1]
for layer in tqdm(range(num_layer)):
    X = res[:, layer, :].cpu()
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    print("Train set size:", X_train.shape, y_train.shape)
    print("Validation set size:", X_val.shape, y_val.shape)
    print("Test set size:", X_test.shape, y_test.shape)
    #
    input_dim = X.shape[1]
    output_dim = len(label_encoder.classes_)  # Number of classes
    # Model
    model = LogisticRegression(input_dim, output_dim)
    #
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # Training the model
    model.train()
    num_epochs = 100
    for epoch in range(num_epochs):        
        optimizer.zero_grad()
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_accuracy = (val_predicted == y_val).sum().item() / y_val.size(0)
                print(f'Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_accuracy:.4f}')
    model.eval()
    # Testing
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        _, test_predicted = torch.max(test_outputs.data, 1)
        test_accuracy = (test_predicted == y_test).sum().item() / y_test.size(0)
        print(f'Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_accuracy:.4f}')

