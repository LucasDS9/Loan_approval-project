# **🏦 Loan Approval Prediction**

Projeto completo de **Machine Learning** focado em prever a aprovação de empréstimos bancários.  
Além da previsão da aprovação, o projeto também realiza **regressão** para estimar a taxa de juros (*interest rate*) ideal para cada cliente aprovado, oferecendo uma solução completa para decisões de crédito.

O problema tratado envolve identificar, de forma automática e assertiva, quais clientes têm maior chance de ter seu empréstimo aprovado e também os fatores que mais agregam para a aprovação de crédito.  

A resolução auxilia instituições financeiras a **reduzir riscos**, **automatizar processos**, **otimizar taxas oferecidas** e **melhorar a eficiência operacional** em suas políticas de concessão de crédito.

---

## 🎯 **Objetivos do Projeto**
- Construir um modelo que classifique corretamente clientes em **aprovados** ou **não aprovados**.
- Realizar regressão para prever a **taxa de juros (interest rate)** de clientes aprovados.
- Identificar variáveis com maior impacto no processo de aprovação.
- Criar uma visao geral completa, desde EDA até modelos preditivos e avaliação.

---

## 📁 Estrutura do Projeto

```text
📦 loan_project
├── 📁 app
│   └── app.py (Aplicação em streamlit)
│
├── 📁 artifacts
│   ├── model_classifier.pkl
│   ├── model_regression.pkl
│   ├── regression_features.pkl
│   ├── regression_dtypes.pkl
│   ├── test_classifier.pkl
│   └── test_regression.pkl
│
├── 📁 data
│   └── Loan_dataset.xlsx (Dataset do projeto)
│
├── 📁 notebooks
│   ├── loan_full_project.ipynb (Projeto completo)
│   └── pipeline_visual.ipynb 
│
├── 📁 src
│   ├── pipeline.py 
│   ├── train.py
│   └── evaluate.py
│
├── README.md
└── __pycache__
```
---

## 🧱 **Etapas do Projeto**

### 1️⃣ **Importações e conhecimento inicial do dataset**
- Leitura dos dados
- Primeiras inspeções: tipos das variáveis, estatísticas iniciais, identificação de inconsistências
- Entendimento geral da estrutura do conjunto de dados

---

### 2️⃣ **Análise Exploratória de Dados (EDA)**
- Geração de gráficos das variáveis e suas relações com a variável *approval*
- Avaliação da distribuição das variáveis numéricas e categóricas
- Verificação de padrões, tendências e possíveis outliers
- Análises que ajudam a entender o comportamento dos aprovados vs não aprovados

---

### 3️⃣ **Pré-processamento**
- Limpeza e organização dos dados  
- Tratamento de valores ausentes, incluindo uso do **KNN Imputer** quando necessário
- Codificação de variáveis categóricas com **OrdinalEncoder**
- Análise da **matriz de correlação com o alvo** para identificar variáveis mais relevantes
- Padronização e preparação final do dataset para modelagem

---

### 4️⃣ **Treinamento e avaliação do modelo (Classificação)**

O modelo ideal foi selecionado através de **GridSearchCV** e avaliado pelas métricas clássicas de classificação:

```
              precision    recall  f1-score   support

           0       1.00      0.98      0.99      1150
           1       0.93      1.00      0.96       350

    accuracy                           0.98      1500
   macro avg       0.96      0.99      0.98      1500
weighted avg       0.98      0.98      0.98      1500
```

Com **98% de acurácia**, o modelo obteve excelente desempenho na classificação de aprovação.
- Foi também utilizado uma **matriz de confusão**, que representa os acertos e erros do modelo 
---

### 5️⃣ **Regressão para prever a Taxa de Juros (Interest Rate)**

Modelos utilizados:
- **Linear Regression**
- **Lasso Regression**

Resultados:

| Modelo             | R²        | MAE      |
|-------------------|-----------|----------|
| Linear Regression | 0.997680  | 0.077567 |

### Desempenho final do modelo:

- **Treino:** **R² = 0.9990 | MAE = 0.0708** 
- **Teste:** **R² = 0.9977 | MAE = 0.0776**

Também foi gerado um **gráfico Real vs Predito**, mostrando alto alinhamento entre os valores.  
Por fim, uma previsão real foi realizada usando um registro separado de teste, comparando o valor real e o predito e comprovando o baixo erro do modelo.

Além disso, no final, foi feito uma visão geral sobre o lucro da instituição, a porcentagem lucrativa, o valor investido e a trend de lucros por faixas semestrais.

---

### 5️⃣ Modularização e aplicação em streamlit
O projeto conta com uma aplicação em streamlit onde o usuário insere informações e recebe o resultado de aprovação ou rejeição, se aprovado também é calculado automaticamente a taxa de juros a ser paga

---

## 🧠 Principais Insights do Projeto




---


## 🚀 **Conclusão**
O projeto entrega uma solução completa que combina:
- Modelo robusto para **aprovação de empréstimos**
- Regressão eficiente para **definir taxas personalizadas**
- Análises profundas para entendimento dos fatores de decisão

Essa abordagem pode ser facilmente expandida e aplicada em cenários reais de instituições financeiras.

---
## 🛠 Tecnologias Utilizadas

| Tecnologia | Função |
|-----------|--------|
| 🐍 **Python** | Linguagem principal do projeto |
| 🧮 **Pandas / NumPy** | Manipulação e análise de dados |
| 📊 **Matplotlib / Seaborn** | Visualizações e gráficos |
| 🤖 **Scikit-learn** | Modelagem e métricas |
| 🌲 **RandomForestClassifier** | Classificador utilizado |
| 🚀 **Streamlit** | Criação da interface web e deploy do modelo |
| 📦 **Pickle** | Serialização e carregamento do modelo |
| ⚙️ **Joblib** | Persistência eficiente de modelos e objetos grandes |


---
