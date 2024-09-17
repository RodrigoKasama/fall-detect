Resumo do Projeto;

-   Otimização de hiperparametros via optuna

Hiperparametros de CNN1D:
-   filter_size(8~600, log=True)
-   kernel_size(2~6)
-   num_layers(2~4)
-   num_dense_layers(1~3)
-   dense_neurons(60~320, log=True)
-   dropout(.1~.5, step=.1)
-   learning_rate([0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01])
-   decision_threshold(.5~.9, log=True)
-   matthews_corrcoef


Arquitetura CNN1D:
- 

Hiperparametros de MLP:
- num_layers,
- dense_neurons,
- dropout,
- learning_rate,
- decision_threshold
- matthews_corrcoef



## Etapa 1
 De acordo com a entrada via CLI define o cenário a ser treinado. 
	- Tamanho do 'data_array' (Tam. do janelamento?)
	- Nome do arquivo que contem o dataset
	- Nome do arquivo que contem os rótulos
	- Numero de targets (37, 26 ou 2)
 - Definir os melhores hiperparametros da rede neural baseados em um dataset de validação - Otimização Bayesiana


## Etapa 2
 - Efetuar um treinamento com os melhores hiperparâmetros selecionados