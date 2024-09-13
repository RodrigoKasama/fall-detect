import optuna
import torch
import torch.nn as nn
import argparse
import os
import torch.optim as optim
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_auc_score, roc_curve, classification_report, confusion_matrix
import optuna
import csv

from sklearn.model_selection import train_test_split
import numpy as np


def generate_datasets(data: str = None, label: str = None):
	# Antigo generate_training_testing_and_validation_sets()
	# Carregando os dados e os targuets baseado no
	X = np.load(data)
	y = np.load(label)

	X = torch.from_numpy(X)
	y = torch.from_numpy(y)

	# 40% para treinamento
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.4, random_state=42)
	# 30% + 30% para validação e teste
	X_test, X_val, y_test, y_val = train_test_split(
		X_test, y_test, test_size=0.5, random_state=42)

	# to_categorical????  - semelhante a get_dummy?
	# y_train = to_categorical(y_train)
	# y_test = to_categorical(y_test)
	# y_val = to_categorical(y_val)

	return X_train, y_train, X_val, y_val, X_test, y_test


class CNN1D(nn.Module):
	def __init__(self, input_shape, filter_size, kernel_size, num_layers, num_dense_layers, dense_neurons, dropout, number_of_labels):
		super(CNN1D, self).__init__()

		self.conv_layers = nn.ModuleList()
		self.pool = nn.MaxPool1d(2)  # Equivalente ao pool_size = 2 (max_pool)
		self.dropout = nn.Dropout(dropout)

		for i in range(num_layers):
			if i == 0:
				# Primeira camada convolucional recebe input_shape como dimensão de entrada
				self.conv_layers.append(nn.Conv1d(
					in_channels=input_shape[0], out_channels=filter_size, kernel_size=kernel_size))
			else:
				# Checando a relação entre filter_size e kernel_size
				if filter_size < kernel_size:
					filter_size = kernel_size
				filter_size *= 2
				# Próximas camadas convolucionais
				self.conv_layers.append(nn.Conv1d(
					in_channels=filter_size // 2, out_channels=filter_size, kernel_size=kernel_size))

		# self.flatten_size = self.calculate_flatten_size(input_shape, filter_size, kernel_size, num_layers)
		self.flatten = nn.Flatten()

		# Fully Connected Layers (Dense Layers)
		self.fc_layers = nn.ModuleList()

		for _ in range(num_dense_layers):
			self.fc_layers.append(nn.Linear(filter_size, dense_neurons))
			# Atualiza o tamanho da entrada para a próxima camada densa
			self.flatten_size = dense_neurons

		# Output Layer (softmax)
		self.output_layer = nn.Linear(self.flatten_size, number_of_labels)

	def forward(self, x):
		# Para cada camada de convoluçaõ:
		for conv in self.conv_layers:
			x = torch.relu(conv(x))  # Ativação ReLU
			x = self.pool(x)         # MaxPooling1D
			x = self.dropout(x)      # Dropout

		x = self.flatten(x)          # Achatando os neuronios

		for fc in self.fc_layers:
			x = torch.relu(fc(x))   # Camada totalmente conectada (Dense Layer)

		x = torch.softmax(self.output_layer(x), dim=1)  # Saída com softmax
		return x

	def calculate_flatten_size(self, input_shape, filter_size, kernel_size, num_layers):
		# Calcula o tamanho do vetor flatten após as camadas convolucionais
		length = input_shape[1]
		for _ in range(num_layers):
			length = (length - kernel_size + 1) // 2  # Conv1D + MaxPool1D
		return length * filter_size
# Hiperparametros do best_score
# model = CNN1D(input_shape=(1020, 1), filter_size=16, kernel_size=5, num_layers=2, num_dense_layers=2, dense_neurons=202, dropout=0.5, number_of_labels=10)
# print(model)

# c = np.load("./labels_and_data/labels/chest/binary_class_label_1.npy")
# d = np.load("./labels_and_data/labels/chest/binary_class_label_2.npy")

# b = np.load("./labels_and_data/labels/chest/multiple_class_label_1.npy")
# a = np.load("./labels_and_data/labels/chest/multiple_class_label_2.npy")
# [print(set(x), x.shape) for x in [c, d, b, a]]
# exit(0)
# ##############################################################################################################


def create_study_object(objective, input_shape, X_train, y_train, X_val, y_val, neural_network_type, neural_network_results_dir, number_of_labels, training_epochs=25):

	study = optuna.create_study(direction="maximize")

	study.optimize(lambda trial: objective(trial, input_shape, X_train, y_train, X_val,
				   y_val, neural_network_type, neural_network_results_dir, number_of_labels, training_epochs), n_trials=5)

	best_trial = study.best_trial
	best_params = best_trial.params

	return best_trial, best_params


def parse_input():
	parser = argparse.ArgumentParser(
		description="Script for Bayesian optimization and model training")
	parser.add_argument(
		"-s", "--scenario",
		type=str,
		choices=[
			"Sc1_acc_T", "Sc1_gyr_T", "Sc1_acc_F", "Sc1_gyr_F",
			"Sc_2_acc_T", "Sc_2_gyr_T", "Sc_2_acc_F", "Sc_2_gyr_F",
			"Sc_3_T", "Sc_3_F", "Sc_4_T", "Sc_4_F"
		],
		required=True,
		help="Neural network scenario (e.g. Sc1_acc_F, Sc1_gyr_T, etc.)",
	)
	parser.add_argument(
		"-p", "--position",
		type=str,
		choices=["left", "chest", "right"],
		required=True,
		help="Sensor position (left, chest, right)",
	)
	parser.add_argument(
		"-l", "--label_type",
		type=str,
		choices=["multiple_one", "multiple_two",
				 "binary_one", "binary_two"],
		required=True,
		help="Label type (multiple_one, multiple_two, binary_one, binary_two)",
	)
	parser.add_argument(
		"-nn", "--neural_network_type",
		type=str,
		choices=["CNN1D", "MLP"],
		required=True,
		help="Tipo de rede neural (CNN1D ou MLP)",
	)
	args = parser.parse_args()

	return args.position, args.label_type, args.scenario, args.neural_network_type


def set_data_filename_and_shape_input(data_dir, array_size, scenario, neural_network_type):

	###################################################################################################################################
	# Para cada cenário de CNN1D, cria uma lista com o diretório do dado e o shape de entrada
	neural_network_scenarios = {
		# for Sc1_CNN1D_acc_T and Sc1_MLP_acc_T
		"Sc1_acc_T": [os.path.join(data_dir, "magacc_time_domain_data_array.npy"), (array_size, 1)],
		# for Sc1_CNN1D_gyr_T and Sc1_MLP_gyr_T
		"Sc1_gyr_T": [os.path.join(data_dir, "maggyr_time_domain_data_array.npy"), (array_size, 1)],
		# for Sc1_CNN1D_acc_F and Sc1_MLP_acc_F
		"Sc1_acc_F": [os.path.join(data_dir, "magacc_frequency_domain_data_array.npy"), (int(array_size/2), 1)],
		# for Sc1_CNN1D_gyr_F and Sc1_MLP_gyr_F
		"Sc1_gyr_F": [os.path.join(data_dir, "maggyr_frequency_domain_data_array.npy"), (int(array_size/2), 1)],

		# for Sc_2_CNN1D_acc_T and Sc_2_MLP_acc_T
		"Sc_2_acc_T": [os.path.join(data_dir, "acc_x_y_z_axes_time_domain_data_array.npy"), (array_size, 3)],
		# for Sc_2_CNN1D_gyr_T and Sc_2_MLP_gyr_T
		"Sc_2_gyr_T": [os.path.join(data_dir, "gyr_x_y_z_axes_time_domain_data_array.npy"), (array_size, 3)],
		# for Sc_2_CNN1D_acc_F and Sc_2_MLP_acc_F
		"Sc_2_acc_F": [os.path.join(data_dir, "acc_x_y_z_axes_frequency_domain_data_array.npy"), (int(array_size/2), 3)],
		# for Sc_2_CNN1D_gyr_F and Sc_2_MLP_gyr_F
		"Sc_2_gyr_F": [os.path.join(data_dir, "gyr_x_y_z_axes_frequency_domain_data_array.npy"), (int(array_size/2), 3)],

		# for Sc_3_CNN1D_T and Sc_3_MLP_T
		"Sc_3_T": [os.path.join(data_dir, "magacc_and_maggyr_time_domain_data_array.npy"), (array_size, 2)],
		# for Sc_3_CNN1D_F and Sc_3_MLP_F
		"Sc_3_F": [os.path.join(data_dir, "magacc_and_maggyr_frequency_domain_data_array.npy"), (int(array_size/2), 2)],

		# for Sc_4_CNN1D_T and Sc_4_MLP_T
		"Sc_4_T": [os.path.join(data_dir, "acc_and_gyr_three_axes_time_domain_data_array.npy"), (array_size, 6)],
		# for Sc_4_CNN1D_F and Sc_4_MLP_F
		"Sc_4_F": [os.path.join(data_dir, "acc_and_gyr_three_axes_frequency_domain_data_array.npy"), (int(array_size/2), 6)],
	}

	# O nome do arquivo de dados será definido de acordo com o cenário.
	# O formato de entrada da RN será do tipo definido em neural_network_scenarios ou por array_sizes, a depender da arquitetura da RN
	data_filename, input_shape = neural_network_scenarios[scenario]
	if neural_network_type == "MLP":
		input_shape = array_size

	return data_filename, input_shape
	###################################################################################################################################


def collect_datasets_from_input(position, target_type, scenario, neural_network_type):

	targets_filename_and_size = {
		# Nume do arquivo dos targets e qunatidade de classes
		"multiple_one": ("multiple_class_label_1.npy", 37),
		"multiple_two": ("multiple_class_label_2.npy", 26),
		"binary_one": ("binary_class_label_1.npy", 2),
		"binary_two": ("binary_class_label_2.npy", 2),
	}

	array_sizes = {"chest": 1020, "right": 450, "left": 450}

	label_filename, label_size = targets_filename_and_size.get(target_type)

	array_size = array_sizes[position]

	#  O arquivo de rótulos é label_dir + label_filename
	label_path = os.path.join(label_dir, label_filename)

	data_filename, input_shape = set_data_filename_and_shape_input(
		data_dir=data_dir, array_size=array_size, scenario=scenario, neural_network_type=neural_network_type)

	# X_train, X_test, y_train, y_test, X_val, y_val = generate_datasets(data_filename, label_path)
	return input_shape, label_size, generate_datasets(data_filename, label_path)


def objective(trial, input_shape, X_train, y_train, X_val, y_val, neural_network_type, output_dir, number_of_labels, training_epochs):

	mcc = None

	if neural_network_type == "CNN1D":

		# Definindo o espaço de busca dos hiperparâmetros
		filter_size = trial.suggest_int('filter_size', 8, 600, log=True)
		kernel_size = trial.suggest_int('kernel_size', 2, 6)
		num_layers = trial.suggest_int('num_layers', 2, 4)
		num_dense_layers = trial.suggest_int('num_dense_layers', 1, 3)
		dense_neurons = trial.suggest_int('dense_neurons', 60, 320, log=True)
		dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
		learning_rate = trial.suggest_categorical(
			'learning_rate', [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01])
		decision_threshold = trial.suggest_float(
			'decision_threshold', 0.5, 0.9, step=0.1)

		model, historic = cnn1d_architecture(input_shape, X_train, y_train, X_val, y_val, filter_size,
											 kernel_size, num_layers, num_dense_layers, dense_neurons, dropout, learning_rate, number_of_labels, training_epochs)

		y_pred_prob = model.predict(X_val)
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

	elif neural_network_type == "MLP":

		num_layers = trial.suggest_int('num_layers', 1, 5)
		dense_neurons = trial.suggest_int('dense_neurons', 20, 4000, log=True)
		dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
		learning_rate = trial.suggest_categorical(
			'learning_rate', [0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07])
		decision_threshold = trial.suggest_float(
			'decision_threshold', 0.5, 0.9, step=0.1)

		model, historic = mlp_architecture(input_shape, X_train, y_train, X_val,
										   y_val, num_layers, dense_neurons, dropout, learning_rate, number_of_labels)

		y_pred_prob = model.predict(X_val)
		y_pred = (y_pred_prob[:, 1] >= decision_threshold).astype(int)

		mcc = matthews_corrcoef(X_val.argmax(axis=1), y_pred)

		optimized_params = {
			"num_layers": num_layers,
			"dense_neurons": dense_neurons,
			"dropout": dropout,
			"learning_rate": learning_rate,
			"decision_threshold": decision_threshold
		}

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


def cnn1d_architecture(input_shape, X_train, y_train, X_val, y_val, filter_size, kernel_size, num_layers, num_dense_layers, dense_neurons, dropout, learning_rate, number_of_labels, training_epochs):
	model = CNN1D(input_shape=input_shape, filter_size=filter_size, kernel_size=kernel_size,
				  num_layers=num_layers, num_dense_layers=num_dense_layers,
				  dense_neurons=dense_neurons, dropout=dropout)
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	criterion = nn.CrossEntropyLoss()
 
	# # Convert data to PyTorch tensors and create data loaders
	# train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
	# val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

	# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
	# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


	# for epoch in range(training_epochs):
	# 	# Training loop (manual calculation of loss and accuracy)
	# 	model.train()  # Set the model to training mode
	# 	running_loss = 0.0
	# 	correct = 0
	# 	total = 0

	# 	for inputs, labels in train_loader:  # Assuming you have a DataLoader for training data
	# 		optimizer.zero_grad()  # Clear the gradients from the previous step

	# 		# Forward pass
	# 		outputs = model(inputs)
	# 		loss = criterion(outputs, labels)

	# 		# Backward pass and optimization step
	# 		loss.backward()
	# 		optimizer.step()

	# 		# Accumulate loss
	# 		running_loss += loss.item()

	# 		# Calculate accuracy
	# 		_, predicted = torch.max(outputs, 1)
	# 		total += labels.size(0)
	# 		correct += (predicted == labels).sum().item()
	# 	accuracy = 100 * correct / total
	# 	print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Accuracy: {accuracy}%')


if __name__ == "__main__":

	position, label_type, scenario, neural_network_type = parse_input()

	# Melhorar
	current_directory = os.path.dirname(__file__)
	data_dir = os.path.join(
		current_directory, "labels_and_data", "data", position)
	label_dir = os.path.join(
		current_directory, "labels_and_data", "labels", position)

	output_dir = os.path.join(current_directory, "output")
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	input_shape, num_labels, X_train, y_train, X_val, y_val, X_test, y_test, = collect_datasets_from_input(
		position=position, target_type=label_type, scenario=scenario, neural_network_type=neural_network_type)

	neural_network_results_dir = os.path.join(output_dir, neural_network_type)
	if neural_network_type == "CNN1D":
		neural_network_results_dir = os.path.join(
			neural_network_results_dir, position)

	if not os.path.exists(neural_network_results_dir):
		os.makedirs(neural_network_results_dir)

	scenario_dir = os.path.join(
		neural_network_results_dir, scenario, label_type)
	if not os.path.exists(scenario_dir):
		os.makedirs(scenario_dir)

	create_study_object(objective, input_shape, X_train, y_train,
						X_val, y_val, neural_network_type, scenario_dir, num_labels)

	print(X_train.shape, y_train.shape)
	print(X_val.shape, y_val.shape)
	print(X_test.shape, y_test.shape)
