import requests
import zipfile
import io
import os
from numpy import array

import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score




class Net(nn.Module):
    def __init__(self, in_channels):
        super(Net, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=60, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=60, out_channels=30, kernel_size=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(300, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        print(x.shape)
        x = self.conv1d(x)
        print(x.shape)
        # x = self.conv1d_2(x)
        # print(x.shape)
        # x = self.relu(x)
        x = x.view(x.shape[0], -1)
        print(x.shape)
        x = self.fc1(x)
        print(x.shape)
        x = self.relu(x)
        x = self.fc2(x)
        print(x.shape)
        input()
        # x = self.relu(x)
        # x = F.sigmoid(x)
        return x


def evaluate_model(model, test_loader):
    model.eval()  # Colocar o modelo em modo de avaliação
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Não precisamos calcular os gradientes
        for inputs, labels in test_loader:
            outputs = model(inputs.float())
            preds = torch.round(outputs)  # Arredondar a saída para 0 ou 1
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

        # Cálculo das métricas
        acc = accuracy_score(all_labels, all_preds)
        # precision = precision_score(all_labels, all_preds)
        # recall = recall_score(all_labels, all_preds)
        # f1 = f1_score(all_labels, all_preds)

        print(f'Acurácia: {acc:.4f}')
        # print(f'Precisão: {precision:.4f}')
        # print(f'Revocação: {recall:.4f}')
        # print(f'F1-Score: {f1:.4f}')


def fit(epochs, lr, model, train_dl, val_dl, opt_func=torch.optim.SGD):

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    criterion = nn.BCEWithLogitsLoss()
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for data, target in train_dl:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data.float())

            # calculate the loss
            loss = criterion(output, target.float())

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())
            # input()

        train_loss = np.average(train_losses)
        avg_train_losses.append(train_loss)

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for data, target in val_dl:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data.float())
            # calculate the loss
            loss = criterion(output, target.float())
            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch

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


def extract_dataset():

    if not os.path.exists("./jena_climate_2009_2016.csv"):
        print("Importando o dataset...", end="")
        zip_file_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
        r = requests.get(zip_file_url, stream=True)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()
        print("OK.")

    df = pd.read_csv('jena_climate_2009_2016.csv')
    return df


def window_generator(sequence, n_steps):

    x, y = list(), list()
    for i in range(len(sequence)):

        end_ix = i + n_steps

        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)


if __name__ == "__main__":

    BATCH_SIZE = 32
    temp_threshold = 5

    n_steps = 7

    df = extract_dataset()
    print("Shape Original:", df.shape)
    # Agrupar de hora em hora:
    df = df[5::6]
    print("Shape de Hora em Hora:", df.shape)

    # Univariável, somente temperatura
    df_temp = df[["T (degC)"]]

    # Intervalo do dataset
    train_set = df_temp[:20200]
    valid_set = df_temp[61000:64000]
    test_set = df_temp[64000:]

    # Janelamento (Windowing) da variável e criação do dataset
    X_train, y_train = window_generator(train_set.values, n_steps)
    X_valid, y_valid = window_generator(valid_set.values, n_steps)
    X_test, y_test = window_generator(test_set.values, n_steps)

    # Conversão para classificação binária
    y_train = np.where(y_train > temp_threshold, 1, 0)
    y_valid = np.where(y_valid > temp_threshold, 1, 0)
    y_test = np.where(y_test > temp_threshold, 1, 0)

    # Conversão para tensor
    X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
    X_valid, y_valid = torch.from_numpy(X_valid), torch.from_numpy(y_valid)
    X_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test)

    # Pivoteamento do tensor [batch, in_channels, time]
    X_train = torch.permute(X_train, (0, 2, 1))
    X_valid = torch.permute(X_valid, (0, 2, 1))
    X_test = torch.permute(X_test, (0, 2, 1))

    # Agrupamento de features e targets
    train_ds = TensorDataset(X_train, y_train)
    valid_ds = TensorDataset(X_valid, y_valid)
    test_ds = TensorDataset(X_test, y_test)

    # Criação de batches
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    n_epochs = 100
    model = Net(in_channels=1)
    print(model)

    # model = model.float()
    lr = 1e-5
    model, train_loss, valid_loss = fit(n_epochs, lr, model, train_dl, val_dl)

    evaluate_model(model, test_dl)

    # print(X_valid.shape)
    # print(X_test.shape)
