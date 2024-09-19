from torch.utils.data import DataLoader, TensorDataset
import os
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv

import matplotlib.pyplot as plt

from sklearn.metrics import matthews_corrcoef

import numpy as np
from training_imports import *

BATCH_SIZE = 32


def plot_loss_curve(train_loss: list, valid_loss: list, image_dir: str = "./", filename: str = "plot_loss_curve"):
    import os
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss)+1), train_loss, label="Training Loss")
    plt.plot(range(1, len(valid_loss)+1), valid_loss, label="Validation Loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xlim(0, len(train_loss)+1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(image_dir, filename)
    fig.savefig(path, bbox_inches="tight")


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
                self.conv_layer.append(
                    nn.Conv1d(1, first_conv_layer_size, self.kernel_size))
                last_layer_channels = first_conv_layer_size
            else:
                # past_layer_out = self.get_feature_size(i-1, n_channels_init)
                self.conv_layer.append(
                    nn.Conv1d(last_layer_channels, last_layer_channels*2, self.kernel_size))
                last_layer_channels *= 2

            self.conv_layer.append(nn.ReLU())

            # PARA MAXPOOL: Divide a metade |-> [batch, j, k] -> [batch, j, k/2]
            self.conv_layer.append(nn.MaxPool1d(self.max_pool))

            self.conv_layer.append(nn.Dropout(self.dropout_rate))

        # Camada Flatten
        self.flatten = nn.Flatten()

        # Simula n sequencias de (Conv1d(kenrnel_size) + MaxPool1D(max_pool)) e retorna o numero de features após essas operações
        last_layer_features = self.get_feature_size(
            n_conv_layers, n_features_init)

        # Calcular com quantos neuronios a 1ª camada densa deve ter -> nº de canais * nº de features da última camada
        self.first_dense_input = last_layer_channels * last_layer_features

        self.fc_layers = nn.ModuleList()
        for i in range(num_dense_layers):
            if i == 0:
                self.fc_layers.append(
                    nn.Linear(self.first_dense_input, dense_neurons))
            else:
                self.fc_layers.append(
                    nn.Linear(dense_neurons, dense_neurons//dense_layer_droprate))
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

        x = self.flatten(x)  # x = x.view(x.size(0), -1)
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

        print_msg = (f"[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] " +
                     f"train_loss: {train_loss:.5f} " +
                     f"valid_loss: {valid_loss:.5f}")

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

    return model, avg_train_losses, avg_valid_losses


def create_study_object(objective, input_shape, X_train, y_train, X_val, y_val, neural_network_type, neural_network_results_dir, number_of_labels, batch_size, training_epochs=25):

    study = optuna.create_study(direction="maximize")
    # Fará chamadas da função
    study.optimize(lambda trial: objective(trial, input_shape, X_train, y_train, X_val,
                                           y_val, neural_network_type, neural_network_results_dir, number_of_labels, training_epochs, batch_size), n_trials=5)

    best_trial = study.best_trial
    best_params = best_trial.params

    return best_trial, best_params


def objective(trial, input_shape, X_train, y_train, X_val, y_val, neural_network_type, output_dir, number_of_labels, training_epochs, batch_size):

    mcc = None

    if neural_network_type == "CNN1D":

        # Fixando momentaneamente os hiperparâmetros
        filter_size = 50
        kernel_size = 5
        num_layers = 3
        num_dense_layers = 2
        dense_neurons = 100
        dropout = 0.3
        learning_rate = 0.0001
        decision_threshold = 0.5

        # Criando a arquitetura da rede neural de acordo com os hiperparametros e retornando um modelo treinado
        model, historic = cnn1d_architecture(input_shape, X_train, y_train, X_val, y_val, filter_size,
                                             kernel_size, num_layers, num_dense_layers, dense_neurons, dropout, learning_rate, number_of_labels, training_epochs, batch_size)

        # SUSPEITA DE DATA LEAKAGE - O modelo treina com os dados de treinamento e validação. Após é coletado o mcc com base novamente nos dados de validaçãp
        # Coleta a predição do modelo
        y_pred_prob = model.predict(X_val)

        # Coleta com o threshold alterado
        y_pred = (y_pred_prob[:, 1] >= decision_threshold).astype(int)
        mcc = matthews_corrcoef(y_val.argmax(axis=1), y_pred)

        optimized_params = {
            "filter_size": filter_size,
            "kernel_size": kernel_size,
            "num_layers": num_layers,
            "num_dense_layers": num_dense_layers,
            "dense_neurons": dense_neurons,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "decision_threshold": decision_threshold
        }

        # Registra o score do conj de hiperparametros
        file_path = os.path.join(output_dir, "optimization_results.csv")
        file_exists = os.path.isfile(file_path)

        with open(file_path, "a", newline="") as csvfile:

            fieldnames = ["Trial", "MCC"] + list(optimized_params.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

                row = {"Trial": trial.number, "MCC": mcc}
                row.update(optimized_params)
                writer.writerow(row)

    return mcc


def cnn1d_architecture(input_shape, X_train, y_train, X_val, y_val, filter_size, kernel_size, num_layers, num_dense_layers, dense_neurons, dropout, learning_rate, number_of_labels, training_epochs, batch_size):

    # print(X_train.)
    print("X_train original", X_train)
    print("Shape Original:", X_train.shape)

    a = torch.permute(X_train, (0, 2, 1))
    print()
    print("X_train", a)
    print("X_train pivotado", a.shape)

    input()
    X_train = torch.permute(X_train, (0, 2, 1))
    X_val = torch.permute(X_val, (0, 2, 1))

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = CNN1D(input_shape=input_shape, filter_size=filter_size, kernel_size=kernel_size, num_layers=num_layers,
                  num_dense_layers=num_dense_layers, dense_neurons=dense_neurons, dropout=dropout, number_of_labels=number_of_labels)

    print(model)
    # input()
    # Retorna um modelo treinado
    model = fit(model, X_train, y_train, X_val, y_val,
                learning_rate, nn.CrossEntropyLoss, training_epochs)
    return model


if __name__ == "__main__":

    position, label_type, scenario, neural_network_type = parse_input()

    # Melhorar
    current_directory = os.path.dirname(__file__)

    output_dir = os.path.join(current_directory, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Diretórios dos datasets. X-> data/{position}/filename
    # Diretórios dos targets. X-> labels/{position}/filename
    data_dir = os.path.join(
        current_directory, "labels_and_data", "data", position)
    label_dir = os.path.join(
        current_directory, "labels_and_data", "labels", position)

    input_shape, num_labels, X_train, y_train, X_val, y_val, X_test, y_test, = collect_datasets_from_input(
        position, label_type, scenario, neural_network_type, label_dir, data_dir)

    # neural_network_results_dir = os.path.join(output_dir, neural_network_type)
    # if neural_network_type == "CNN1D":
    #     neural_network_results_dir = os.path.join(
    #         neural_network_results_dir, position)

    # if not os.path.exists(neural_network_results_dir):
    #     os.makedirs(neural_network_results_dir)

    # scenario_dir = os.path.join(
    #     neural_network_results_dir, scenario, label_type)
    # if not os.path.exists(scenario_dir):
    #     os.makedirs(scenario_dir)
        
    X_train, y_train = X_train[0:100], y_train[0:100]
    X_val, y_val = X_val[0:150], y_val[0:150]
    
    
    # # Criaçao da tarefa de otimização. objective ==> função objective que treina a rede neural com um conj de hiperparametros e retorna um "score" (mcc)
    # create_study_object(objective, input_shape, X_train, y_train, X_val, y_val, neural_network_type, scenario_dir, num_labels, batch_size=32, training_epochs=25)
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

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = CNN1D(
        # Comprimento das features
        # Representa 5s de registros de movimentos no peito (450 para os punhos)
        n_features_init=1020,
        # "n_conv_layers" Sessões de convolução que duplica o nº de canais a partir da 2ª camada
        n_conv_layers=1,
        first_conv_layer_size=25,
        # "n_conv_layers" Sessões de camadas densas que reduzem em 30% o nº de canais a partir da 2ª camada
        num_dense_layers=1,
        first_dense_layer_size=6000,
        num_labels=2  # A depender de uma classificação binária ou multiclasse
    )

    # print("-"*90)
    # print(model)
    # print("-"*90)

    epochs = 100
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 0.0001

    model, train_loss, valid_loss = fit(
        epochs, learning_rate, model, train_dl, val_dl, loss_fn)


    plot_loss_curve(train_loss, valid_loss)