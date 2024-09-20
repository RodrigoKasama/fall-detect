# Fall-detect

### Antes de tudo
Com o objetivo de isolar o projeto, é recomendado a criação de um ambiente virtual do python:

```
cd fall-detect/
python3 -m venv nome-do-enviroment
```

## Importação dos dados

Para execução desse repositório, será necessário utilizar os requisitos descritos no arquivo requirements.txt. É possivel instalar todos os pacotes necessários com o unico comando abaixo:

```
cd fall-detect/
pip install -r requirements.txt
```

Após instalação dos pacotes necessários, importaremos a base de dados a ser utilizada. Essa a base de dados encontra-se diposnível publicamente [aqui](https://zenodo.org/records/12760391). 

## Geração dos datasets 

Uma vez com os pacotes necessários instalados e a base de dados baixada e **descompactada**. Será realizado a criação do dataset através do script:
```
python3 training_data_generator.py chest
python3 training_data_generator.py right
python3 training_data_generator.py left
```

Uma vez com o dataset de cada modalidade (chest, right e left) criado é possivel seguir para a etapa de treinamento da Rede Neural.

## Treinamento e Plotagem 
Para o treinamento, execute o script `training.py` com os parâmetros que deseja como **cenários**, **sensor**, **tipo de classificação**, etc. Em caso de dúvidas, verifique a Wiki do Projeto original e a seção de **--help** do script.

```
python3 training.py -s Sc1_acc_T -p chest -l binary_one -c 2 -d 3

Datasets | Labels
Treinamento: torch.Size([100, 1, 1020]) (torch.float64) | torch.Size([100]) (torch.int64)
Validação: torch.Size([150, 1, 1020]) (torch.float64) | torch.Size([150]) (torch.int64)
Teste: torch.Size([1208, 1, 1020]) (torch.float64) | torch.Size([1208]) (torch.int64)
------------------------------------------------------------------------------------------
Arquitetura da Rede Neural:
...
------------------------------------------------------------------------------------------
[ 1/50] train_loss: 1.01968 valid_loss: 0.44025
[ 2/50] train_loss: 0.45207 valid_loss: 0.42807
[ 3/50] train_loss: 0.44221 valid_loss: 0.42522
...
[49/50] train_loss: 0.42600 valid_loss: 0.42295
[50/50] train_loss: 0.43025 valid_loss: 0.42292
Gráfico de Perda gerado com sucesso. (Verifique o diretório ...)
```

Após o treinamento será gerado um grafico, no diretório indicado, para análise do desempenho da rede neural ao longo do treinamento. 

---

#### Observações
Alguns arquivos presentes no repositório servem apenas como comparação com o projeto original (`run_of_the_neural_network_model.py` ou `model_builders/`) ou auxilio (`commands.txt`).
