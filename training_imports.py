import os
import optuna
import csv
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import argparse

def generate_datasets(data: str = None, label: str = None):
    # Antigo generate_training_testing_and_validation_sets()
    # Carregando os dados e os targuets
    X = np.load(data)
    y = np.load(label)
    
    # Convertendo para tensores
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    # 40% para treinamento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    # 30% + 30% para validação e teste
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    
    # É necessário "pivotar" o datset devido a forma como o pytorch interpreta as camadas dos tensores ([batch, features, passo_de tempo])
    X_train = torch.permute(X_train, (0, 2, 1))
    X_val = torch.permute(X_val, (0, 2, 1))
    X_test = torch.permute(X_test, (0, 2, 1))

    return X_train, y_train, X_val, y_val, X_test, y_test

def parse_input():
    parser = argparse.ArgumentParser(
        description="Script for model training")
    parser.add_argument(
        "-s", "--scenario",
        type=str,
        choices=[
            # Cenários sem transformada de fourier
            # Univariada
            "Sc1_acc_T", "Sc1_gyr_T", 
            # Multivariada (x, y, z)
            "Sc_2_acc_T", "Sc_2_gyr_T", 
            # Multivariada (Aceleração Linear e Angular)
            "Sc_3_T", 
            # Multivariada ((x, y, z)-Linear e (x, y, z)-Angular)
            "Sc_4_T",
            
            # Cenários com transformada de fourier
            "Sc1_acc_F", "Sc1_gyr_F", "Sc_2_acc_F", "Sc_2_gyr_F", "Sc_3_F", "Sc_4_F"
        ],
        required=True,
        help="Possiveis Cenários a se trabalhar.\n Cenários com _F referem-se a transformada de fourier entrada, enquanto que _T são leituras sem transformação.",
    )
    parser.add_argument(
        "-p", "--position",
        type=str,
        choices=["left", "chest", "right"],
        required=True,
        help="Sensor position",
    )
    parser.add_argument(
        "-l", "--label_type",
        type=str,
        choices=["binary_one", "binary_two"],
        # choices=["multiple_one", "multiple_two","binary_one", "binary_two"],
        required=True,
        help="Type of classification problem Multi/Binary Classes",
    )
    parser.add_argument(
        "-nn", "--neural_network_type",
        type=str,
        choices=["CNN1D", "MLP"],
        required=True,
        default="CNN1D",
        help="Tipo de rede neural (CNN1D) **MLP abandonada**",
    )
    args = parser.parse_args()

    return args.position, args.label_type, args.scenario, args.neural_network_type

def set_data_filename_and_shape_input(data_dir, array_size, scenario, neural_network_type):

    ###################################################################################################################################
    # Para cada cenário de CNN1D, cria uma lista com o diretório do dado e o shape de entrada
    
    neural_network_scenarios = {
        
        # Leitura da magnitude (SQRT(x² + y² + z²)) da aceleração linear
        "Sc1_acc_T": [os.path.join(data_dir, "magacc_time_domain_data_array.npy"), (array_size, 1)],
        # Leitura da magnitude (SQRT(x² + y² + z²)) da aceleração angular
        "Sc1_gyr_T": [os.path.join(data_dir, "maggyr_time_domain_data_array.npy"), (array_size, 1)],
        # Leitura dos exios (x, y, z) da aceleração linear - > Passa a ter 3 features | Problema multivariado
        "Sc_2_acc_T": [os.path.join(data_dir, "acc_x_y_z_axes_time_domain_data_array.npy"), (array_size, 3)],
        # Leitura dos exios (x, y, z) da aceleração angular - > Passa a ter 3 features | Problema multivariado
        "Sc_2_gyr_T": [os.path.join(data_dir, "gyr_x_y_z_axes_time_domain_data_array.npy"), (array_size, 3)],
        # Leitura da magnitude (SQRT(x² + y² + z²)) da aceleração linear e da aceleração angular - > Passa a ter 2 features | Problema multivariado
        "Sc_3_T": [os.path.join(data_dir, "magacc_and_maggyr_time_domain_data_array.npy"), (array_size, 2)],
        # Leitura dos exios (x, y, z) da aceleração linear E (x, y, z) da aceleração angular - > Passa a ter 6 features | Problema multivariado
        "Sc_4_T": [os.path.join(data_dir, "acc_and_gyr_three_axes_time_domain_data_array.npy"), (array_size, 6)],
        
    	# Também foi realizado uma uma transformada de fourier que mostrou-se promissora na classificação 
     	# - Por conta da caracteristica da transformada, o resultado é uma função espelhada, para resolver esse problema segmentamos a duplicata da transformada
        "Sc1_acc_F": [os.path.join(data_dir, "magacc_frequency_domain_data_array.npy"), (int(array_size/2), 1)],
        "Sc1_gyr_F": [os.path.join(data_dir, "maggyr_frequency_domain_data_array.npy"), (int(array_size/2), 1)],
        "Sc_2_acc_F": [os.path.join(data_dir, "acc_x_y_z_axes_frequency_domain_data_array.npy"), (int(array_size/2), 3)],
        "Sc_2_gyr_F": [os.path.join(data_dir, "gyr_x_y_z_axes_frequency_domain_data_array.npy"), (int(array_size/2), 3)],
        "Sc_3_F": [os.path.join(data_dir, "magacc_and_maggyr_frequency_domain_data_array.npy"), (int(array_size/2), 2)],
        "Sc_4_F": [os.path.join(data_dir, "acc_and_gyr_three_axes_frequency_domain_data_array.npy"), (int(array_size/2), 6)],
    }

    # O nome do arquivo de dados será definido de acordo com o cenário.
    # O formato de entrada da RN será do tipo definido em neural_network_scenarios ou por array_sizes, a depender da arquitetura da RN
    data_filename, input_shape = neural_network_scenarios[scenario]
    if neural_network_type == "MLP":
        input_shape = array_size

    return data_filename, input_shape
    ###################################################################################################################################

def collect_datasets_from_input(position, target_type, scenario, neural_network_type, label_dir, data_dir):

    targets_filename_and_size = {
        # Nume do arquivo dos targets e quantidade de classes
        "multiple_one": ("multiple_class_label_1.npy", 37), # O problema multiclasse não funciona por enquanto
        "multiple_two": ("multiple_class_label_2.npy", 26), # O problema multiclasse não funciona por enquanto
        "binary_one": ("binary_class_label_1.npy", 2),
        "binary_two": ("binary_class_label_2.npy", 2),
    }

    # Quantidade de leituras a cada 5s -> Passo de tempo
    array_sizes = {"chest": 1020, "right": 450, "left": 450}

    label_filename, label_size = targets_filename_and_size.get(target_type)

    array_size = array_sizes[position]

    #  O arquivo de targets é label_dir + label_filename
    label_path = os.path.join(label_dir, label_filename)

    data_filename, input_shape = set_data_filename_and_shape_input(data_dir, array_size, scenario, neural_network_type)

    X_train, y_train, X_val, y_val, X_test, y_test = generate_datasets(data_filename, label_path)

    return input_shape, label_size, X_train, y_train, X_val, y_val, X_test, y_test

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

# Funções não utilizadas voltadas p otimização de hiperparametros
def create_study_object(objective, input_shape, X_train, y_train, X_val, y_val, neural_network_type, neural_network_results_dir, number_of_labels, batch_size, training_epochs=25):

    study = optuna.create_study(direction="maximize")
    # Fará chamadas da função
    study.optimize(lambda trial: objective(trial, input_shape, X_train, y_train, X_val,
                                           y_val, neural_network_type, neural_network_results_dir, number_of_labels, training_epochs, batch_size), n_trials=5)

    best_trial = study.best_trial
    best_params = best_trial.params

    return best_trial, best_params

def objective(trial, input_shape, X_train, y_train, X_val, y_val, neural_network_type, output_dir, number_of_labels, training_epochs, batch_size):
	from sklearn.metrics import matthews_corrcoef

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