import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


class CNN1D(nn.Module):
    def __init__(self, n_features_init, n_conv_layers, first_conv_layer_size,  num_dense_layers, first_dense_layer_size,  num_labels):
        super(CNN1D, self).__init__()

        # filter_size = 50
        # kernel_size = 5
        # num_layers = 3
        # num_dense_layers = 2
        # dense_neurons = 100
        
        # learning_rate = 0.0001
        # decision_threshold = 0.5

        self.conv_layer = nn.ModuleList()

        self.kernel_size = 3
        self.max_pool = 2
        self.dropout_rate = 0.3
        
        last_layer_channels = 0
        dense_neurons = first_dense_layer_size
        
        for i in range(n_conv_layers):

            # PARA CONV1D: Se pading = 0 e stride = 1 |-> [batch, j, k] -> [batch, new_j, k - kernel + 1]
            if i == 0:
                self.conv_layer.append(nn.Conv1d(1, first_conv_layer_size, self.kernel_size))
                last_layer_channels = first_conv_layer_size
            else:
                # past_layer_out = self.get_feature_size(i-1, n_channels_init)
                self.conv_layer.append(nn.Conv1d(last_layer_channels, last_layer_channels*2, self.kernel_size))
                last_layer_channels *= 2

            self.conv_layer.append(nn.ReLU())
            # PARA MAXPOOL: Divide a metade |-> [batch, j, k] -> [batch, j, k/2]
            self.conv_layer.append(nn.MaxPool1d(self.max_pool))
            self.conv_layer.append(nn.Dropout(self.dropout_rate))
            
        self.flatten = nn.Flatten()
        
        # Simula n sequencias de (Conv1d(kenrnel_size) + MaxPool1D(max_pool)) e retorna o numero de features após essas operações
        last_layer_features = self.get_feature_size(n_conv_layers, n_features_init)

        # Calcular com quantos neuronios a 1ª camada densa deve ter -> nº de canais * nº de features da última camada
        self.first_dense_in = last_layer_channels * last_layer_features
        
        self.fc_layers = nn.ModuleList()
        for i in range(num_dense_layers):
            if i == 0:
                self.fc_layers.append(nn.Linear(self.first_dense_in, dense_neurons))
            else:
                self.fc_layers.append(nn.Linear(dense_neurons, dense_neurons//3))
                dense_neurons //= 3
            self.fc_layers.append(nn.ReLU())

        # Output Layer (softmax)
        self.output_layer = nn.Linear(dense_neurons, num_labels)

    def get_feature_size(self, k, init_val):
        def feature_sequence(i, a0):
            if i == 0:
                return a0
            else:
                return (feature_sequence(i-1, a0) - self.kernel_size + 1)//self.max_pool
        return feature_sequence(k, init_val)

    def forward(self, x):
        print("Input:", x.shape)
        for layer in self.conv_layer:
            x = layer(x)
            if layer._get_name() in ("Conv1d", "MaxPool1d"):
                print(layer._get_name(), x.shape)

        # x = x.view(x.size(0), -1)
        x = self.flatten(x)
        print("Flatten:", x.shape)

        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            if fc_layer._get_name() in ("Linear"):
                print(fc_layer._get_name(), x.shape)

        x = torch.softmax(x, dim=1)
        return x

a = torch.randn((32, 1, 1020), dtype=torch.float32)
print(a.shape)

model = CNN1D(
    n_features_init=1020,
    n_conv_layers=3,
    first_conv_layer_size=250,
    num_dense_layers=3,
    first_dense_layer_size=5000,
    num_labels=37
)
print(model)
model(a)

# for i in range(6):
# 	print(model.get_feature_size(i, 1020))
