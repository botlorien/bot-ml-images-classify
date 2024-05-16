
![Logo do projeto](https://drive.google.com/uc?id=1_f9DWbFolAEFZ3ebPuLHj5ZZUVDQuzkg&export=view)

# Automatização de Processos de Rotulagem e Classificação de Imagens com Interface Gráfica

Este projeto Python desenvolve uma interface desktop interativa que facilita a rotulagem de imagens para treinamento de modelos de classificação binária supervisionada. Utilizando o framework Selenium para automação, o sistema automatiza o download de imagens do Google Imagens. As imagens baixadas são exibidas na interface, permitindo que o usuário as rotule de acordo com as categorias desejadas.

As imagens rotuladas são vetorizadas e podem ser armazenadas tanto em um arquivo CSV quanto em um banco de dados PostgreSQL. Esses dados rotulados alimentam o treinamento de vários modelos de machine learning através da biblioteca LazyPredict, que automatiza a comparação e seleção de modelos.

Os modelos treinados são salvos para análises futuras. Relatórios detalhados com métricas de desempenho são gerados e armazenados localmente, permitindo uma fácil consulta e análise dos modelos com melhor desempenho. Finalmente, a interface proporciona a visualização das imagens classificadas pelo modelo selecionado pelo usuário, facilitando a verificação e a aplicação prática dos resultados.

## 🔥 Introdução

A pipeline de execução na Interface segue essa sequência abaixo: 

**Treino:**
* BAIXAR IMAGENS
* CLASSIFICAR IMAGENS
* GRAVAR CLASSIFICAÇÃO
* TREINAR MODELOS
* AUDITAR IMAGENS

**Teste:**
* BAIXAR IMAGENS
* AUDITAR IMAGENS

**Reiniciar a base de dados:**
* LIMPAR DADOS (Opcional caso queira reiniciar a base de dados)

### ⚙️ Pré-requisitos

Esse aplicação foi desenvolvida em ambiente com essas especificações:
* Windows 10 Home
* Intel(R) Core(TM) i7-6500U CPU @ 2.50GHz   2.60 GHz
* 8GB RAM
* Sistema operacional de 64 bits
* Python >= 3.10
* git
  
Obs: Não foi testada em ambiente Linux!

### 🔨 Guia de instalação

Para instalar o repositorio siga o passo a passo abaixo

**Passo 1:**

```cmd
git clone https://github.com/botlorien/Bot_ml_images_classify.git
```

**Passo 2:**

```cmd
cd Bot_ml_images_classify
```

**Passo 3:**

```cmd
python -m venv venv
```

**Passo 4:**

```cmd
.\venv\Scripts\activate
```

**Passo 5:**

```cmd
pip install -r requirements.txt
```

**Passo 6:**

```cmd
python Bot_ml_images_classify.py
```

## 📦 Tecnologias usadas:

* ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
* ![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E)
* ![Selenium](https://img.shields.io/badge/-selenium-%43B02A?style=for-the-badge&logo=selenium&logoColor=white)
* ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
* ![Postgres](https://img.shields.io/badge/postgres-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white)
* ![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)

## 📄 Licença

Esse projeto está sob a licença (MIT) - acesse os detalhes [LICENSE.md](https://github.com/botlorien/Bot_ml_images_classify/blob/main/LICENSE).


## 📞​ Contato e Suporte
* Para suporte, mande um email para botlorien@gmail.com.
* [linkedin](https://www.linkedin.com/in/ben-hur-p-b-santos/)
