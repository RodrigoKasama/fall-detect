import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split

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
        dense_layer_droprate = 4
        
        # Para cada seq de (Conv1d + ReLU + MaxPool1d + Dropout)
        for i in range(n_conv_layers):

            # PARA CONV1D: Se pading = 0 e stride = 1 |-> [batch, j, k] -> [batch, j*2, k - kernel + 1]
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
        
        # Camada Flatten
        self.flatten = nn.Flatten()
        
        # Simula n sequencias de (Conv1d(kenrnel_size) + MaxPool1D(max_pool)) e retorna o numero de features após essas operações
        last_layer_features = self.get_feature_size(n_conv_layers, n_features_init)

        # Calcular com quantos neuronios a 1ª camada densa deve ter -> nº de canais * nº de features da última camada
        self.first_dense_input = last_layer_channels * last_layer_features
        
        self.fc_layers = nn.ModuleList()
        for i in range(num_dense_layers):
            if i == 0:
                self.fc_layers.append(nn.Linear(self.first_dense_input, dense_neurons))
            else:
                self.fc_layers.append(nn.Linear(dense_neurons, dense_neurons//dense_layer_droprate))
                dense_neurons //= dense_layer_droprate
            self.fc_layers.append(nn.ReLU())

        # Output Layer
        self.output_layer = nn.Linear(dense_neurons, num_labels)

    def get_feature_size(self, k, init_val):
        def feature_sequence(i, a0):
            if i == 0:
                return a0
            else:
                return (feature_sequence(i-1, a0) - self.kernel_size + 1)//self.max_pool
        return feature_sequence(k, init_val)

    def forward(self, x):
        # print("Input:", x.shape)
        # print()
        for layer in self.conv_layer:
            x = layer(x)
            # if layer._get_name() in ("Conv1d", "MaxPool1d"):
            #     print(layer._get_name(), x.shape)
            #     if layer._get_name() in ("MaxPool1d"): print()

        x = self.flatten(x) # x = x.view(x.size(0), -1)
        # print("Flatten:", x.shape)
        # print()

        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            # if fc_layer._get_name() in ("Linear"):
            #     print(fc_layer._get_name(), x.shape)
                
        # print()
        x = self.output_layer(x)
        # print("Output:", x.shape)
        x = torch.softmax(x, dim=1)
        # x = torch.argmax(x, dim=1)
        # print("Argmax:", x.shape)
        return x


def fit(epochs, lr, model, train_dl, val_dl, criterion, opt_func=torch.optim.SGD):

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for data, target in train_dl:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data.float())

            # calculate the loss
            loss = criterion(output, target)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for data, target in val_dl:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data.float())
            # calculate the loss
            loss = criterion(output, target)
            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        
        train_loss = np.average(train_losses)
        avg_train_losses.append(train_loss)
        valid_loss = np.average(valid_losses)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

    return model, avg_train_losses, avg_valid_losses


batch_size=32

X = np.load("../labels_and_data/data/chest/magacc_time_domain_data_array.npy")
y = np.load("../labels_and_data/labels/chest/binary_class_label_1.npy")

X = torch.from_numpy(X)
y = torch.from_numpy(y)

print("Dataset:", X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=500, random_state=42)
# 30% + 30% para validação e teste
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=250, random_state=42)

X_train = torch.permute(X_train, (0, 2, 1))
X_val = torch.permute(X_val, (0, 2, 1))
X_test = torch.permute(X_test, (0, 2, 1))

print("Datasets Pivotados")
print("Treinamento:", X_train.shape, X_train.dtype)
print("Validação:", X_val.shape, X_val.dtype)
print("Teste:", X_test.shape, X_test.dtype)

print("Labels:")
print("Treinamento:", y_train.shape, y_train.dtype)
print("Validação:", y_val.shape, y_val.dtype)
print("Teste:", y_test.shape, y_test.dtype)

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# a = torch.randn((32, 1, 1020))

model = CNN1D(
    # Comprimento das features
    n_features_init=1020, # Representa 5s de registros de movimentos no peito (450 para os punhos)
    # 'n_conv_layers' Sessões de convolução que duplica o nº de canais a partir da 2ª camada 
    n_conv_layers=1,
    first_conv_layer_size=25,
    # 'n_conv_layers' Sessões de camadas densas que reduzem em 30% o nº de canais a partir da 2ª camada 
    num_dense_layers=1,
    first_dense_layer_size=6000,
    num_labels=2 # A depender de uma classificação binária ou multiclasse
)

# print("-"*90)
# print(model)
# print("-"*90)

epochs = 100
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.0001

model, train_loss, valid_loss = fit(epochs, learning_rate, model, train_dl, val_dl,loss_fn)


# Extract function from 
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(train_loss)+1), train_loss, label='Training Loss')
plt.plot(range(1,len(valid_loss)+1), valid_loss,label='Validation Loss')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.xlim(0, len(train_loss)+1)
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.show()
fig.savefig('loss_plot.png', bbox_inches='tight')

# Reduzir lr e criar amostragem 
# Plottagem de gráficos
# BCELoss