import os
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv

from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_auc_score, roc_curve, classification_report, confusion_matrix

# Ainda não estou trabalhando com Batches
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from training_imports import *


class CNN1D(nn.Module):
    def __init__(self, input_shape, filter_size, kernel_size, num_layers, num_dense_layers, dense_neurons, dropout, number_of_labels):
        super(CNN1D, self).__init__()
        
        # filter_size = 50
        # kernel_size = 5
        # num_layers = 3
        # num_dense_layers = 2
        # dense_neurons = 100
        # dropout = 0.3
        # learning_rate = 0.0001
        # decision_threshold = 0.5
        
        self.conv_layers = nn.Sequential()
        # self.conv_layers.append(nn.Conv1d(in_channels=1, out_channels=filter_size, kernel_size=kernel_size))
        self.conv_layers.append(nn.Conv1d(in_channels=input_shape[0], out_channels=filter_size, kernel_size=kernel_size))
        
        # for i in range(num_layers):
            # if i == 0:
                # Primeira camada convolucional recebe input_shape como dimensão de entrada
				# self.conv_layers.append(nn.Conv1d(in_channels=input_shape, out_channels=filter_size, kernel_size=kernel_size)
            # else:
            #     if filter_size < kernel_size: filter_size = kernel_size
            #     filter_size *= 2

            #     # Próximas camadas convolucionais
            #     self.conv_layers.append(nn.Conv1d(in_channels=filter_size, out_channels=filter_size, kernel_size=kernel_size))
    
        self.dropout = nn.Dropout1d(dropout)
        self.pool = nn.MaxPool1d(2)
            
        # self.flatten_size = self.calculate_flatten_size(input_shape, filter_size, kernel_size, num_layers)
        self.flatten = nn.Flatten()

        # Fully Connected Layers (Dense Layers)
        self.fc_layers = nn.Sequential()
        
		# Nº de neuronios pós-flatten?
        self.fc_layers.append(nn.Linear(25400, dense_neurons))
        self.flatten_size = dense_neurons
        
        # for _ in range(num_dense_layers):
        #     self.fc_layers.append(nn.Linear(filter_size, dense_neurons))
        #     self.fc_layers.append(nn.ReLU())
            
            # Atualiza o tamanho da entrada para a próxima camada densa
            # self.flatten_size = dense_neurons

        # Output Layer (softmax)
        self.output_layer = nn.Linear(dense_neurons, number_of_labels)
        
    def forward(self, x):
        # 3x (Conv1D + MaxPool + Dropout)
        print("Entrada:", x.shape)
        x = self.conv_layers(x)
        x = F.relu(x)
        print("Pós Convolução:", x.shape)
        x = F.max_pool1d(x, 2)
        print("Pós MaxPool:", x.shape)
        x = F.dropout1d(x, .5)
        print("Pós Dropout:", x.shape)
        #  Flatten
        # x = x.view(x.shape[0], -1)
        x = self.flatten(x)
        print("Pós Flatten:", x.shape)
        # 3x (Dense)
        x = self.fc_layers(x)
        x = F.relu(x)
        print("Pós Dense Layers:", x.shape)
        #
        x = torch.softmax(self.output_layer(x), dim=1)
        print(x.shape)
        return x

    # def calculate_flatten_size(self, input_shape, filter_size, kernel_size, num_layers):
    #     # Calcula o tamanho do vetor flatten após as camadas convolucionais
    #     length = input_shape[1]
    #     for _ in range(num_layers):
    #         length = (length - kernel_size + 1) // 2  # Conv1D + MaxPool1D
    #     return length * filter_size


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
        file_path = os.path.join(output_dir, 'optimization_results.csv')
        file_exists = os.path.isfile(file_path)

        with open(file_path, "a", newline='') as csvfile:
            
            fieldnames = ["Trial", "MCC"] + list(optimized_params.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

                row = {"Trial": trial.number, "MCC": mcc}
                row.update(optimized_params)
                writer.writerow(row)
    
    return mcc


def cnn1d_architecture(input_shape, X_train, y_train, X_val, y_val, filter_size, kernel_size, num_layers, num_dense_layers, dense_neurons, dropout, learning_rate, number_of_labels, training_epochs, batch_size):

    def fit(model: CNN1D, X_train, y_train, X_val, y_val, lr, criterion, epochs: int):

        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []

        optimizer = optim.Adam(model.parameters(), lr)
        for epoch in range(epochs):
            ###################
            # train the model #
            ###################
            model.train()
            optimizer.zero_grad()
            output = model(X_train.float())
            loss = criterion(output, y_train.float())

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            ######################
            # validate the model #
            ######################
            model.eval()  # prep model for evaluation

            output = model(X_val.float())
            loss = criterion(output, y_val.float())
            valid_losses.append(loss.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            
            epoch_len = len(str(epochs))

            print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' + f'valid_loss: {valid_loss:.5f}')
            print(print_msg)

            train_losses = []
            valid_losses = []


    # print(X_train.)
    # print("X_train original", X_train[0], X_train[0].shape)
    # a = torch.permute(X_train, (0, 2, 1))
    # print()
    # print("X_train redimensionado", a[0], a[0].shape)
    
    X_train = torch.permute(X_train, (0, 2, 1))
    X_val = torch.permute(X_val, (0, 2, 1))
    
    # train_ds = TensorDataset(X_train, y_train)
    # val_ds = TensorDataset(X_val, y_val)

    # train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    # val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    model = CNN1D(input_shape=input_shape, filter_size=filter_size, kernel_size=kernel_size, num_layers=num_layers, num_dense_layers=num_dense_layers, dense_neurons=dense_neurons, dropout=dropout, number_of_labels=number_of_labels)
    
    
    print(model)
    # input()
    # Retorna um modelo treinado
    model = fit(model, X_train, y_train, X_val, y_val, learning_rate, optim.Adam, nn.CrossEntropyLoss, training_epochs)
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
    data_dir = os.path.join(current_directory, "labels_and_data", "data", position)
    label_dir = os.path.join(current_directory, "labels_and_data", "labels", position)

    input_shape, num_labels, X_train, y_train, X_val, y_val, X_test, y_test, = collect_datasets_from_input(
        position, label_type, scenario, neural_network_type, label_dir, data_dir)


    neural_network_results_dir = os.path.join(output_dir, neural_network_type)
    if neural_network_type == "CNN1D":
        neural_network_results_dir = os.path.join(neural_network_results_dir, position)

    if not os.path.exists(neural_network_results_dir):
        os.makedirs(neural_network_results_dir)
    
    scenario_dir = os.path.join(neural_network_results_dir, scenario, label_type)
    if not os.path.exists(scenario_dir):
        os.makedirs(scenario_dir)
    
	# Criaçao da tarefa de otimização. objective ==> função objective que treina a rede neural com um conj de hiperparametros e retorna um 'score' (mcc)
    create_study_object(objective, input_shape, X_train, y_train, X_val, y_val, neural_network_type, scenario_dir, num_labels, batch_size=32, training_epochs=25)
