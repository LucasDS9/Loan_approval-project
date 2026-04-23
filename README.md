# 🏦 Loan Approval Prediction

> Projeto completo de **Machine Learning** focado em prever a aprovação de empréstimos bancários. Além da classificação, o projeto realiza **regressão** para estimar a taxa de juros ideal para cada cliente aprovado — oferecendo uma solução end-to-end para decisões de crédito.

🌐 **[Acesse o portfólio](https://LucasDS9.github.io)** · 🚀 **[Testar o modelo](https://loan-approval-project-yyfi.onrender.com)** · 📓 **[Ver notebook](https://github.com/LucasDS9/Loan_approval-project/blob/main/notebooks/loan_full_project.ipynb)**

---

## 💼 Aplicações Reais

| Caso de uso | Descrição |
|---|---|
| ⚠️ Redução de risco | Identificar automaticamente clientes com perfil de inadimplência |
| ⚙️ Automação de processos | Substituir análises manuais por decisões orientadas a dados |
| 💰 Personalização de taxas | Calcular a taxa de juros ideal para cada perfil aprovado |
| 📈 Eficiência operacional | Agilizar o fluxo de concessão de crédito em escala |

---

## 📌 Problema

O problema tratado envolve identificar, de forma automática e assertiva, quais clientes têm maior chance de ter seu empréstimo aprovado — e quais fatores mais contribuem para essa decisão.

Instituições financeiras precisam equilibrar dois objetivos conflitantes: **maximizar aprovações** para gerar receita e **minimizar inadimplência** para reduzir risco. Este projeto entrega uma solução que endereça os dois lados, combinando classificação de aprovação com previsão de taxa de juros personalizada.

---

## 📁 Estrutura do Projeto

```text
📦 loan_project
├── 📁 app
│   └── app.py                      # Aplicação Streamlit
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
│   └── Loan_dataset.xlsx           # Dataset do projeto
│
├── 📁 notebooks
│   ├── loan_full_project.ipynb     # EDA, insights e implementação completa
│   └── pipeline_visual.ipynb
│
├── 📁 src
│   ├── pipeline.py
│   ├── train.py
│   └── evaluate.py
├── dash.py                         # Dashboard Streamlit
├── requirements.txt
└── README.md

```
## 📊 Resultados

### Visão Geral de Aprovações

![Dashboard 1](images/dash1.png)

1. A distribuição de aprovação mostra que **23.7%** dos clientes (1.187) foram aprovados, enquanto 3.813 tiveram suas tentativas de empréstimo negadas. Isso implica que a instituição possui um sistema rigoroso de aprovação. Nos gráficos seguintes, serão apresentados os fatores que influenciam essa decisão.
2. O gráfico de aprovação por faixa etária mostra que a instituição não aprovou pessoas com menos de 24 anos. Para clientes acima dessa idade, há pouca influência da variável na decisão.
3. No gráfico de aprovação por score de crédito, observa-se explicitamente que a instituição impôs um **limite mínimo de 550** para aprovação, indicando a existência de um critério de corte bem definido.

### Fatores de Risco e Aprovação

![Dashboard 2](images/dash2.png)

A aprovação de crédito é fortemente impactada pelo histórico financeiro do cliente. Clientes sem delinquências concentram a grande maioria das aprovações (86,9%) e apresentam taxa significativamente superior (~54%), enquanto qualquer histórico negativo reduz drasticamente as chances — com nenhuma aprovação para 3 ou mais ocorrências.

O número de empréstimos atuais não demonstra influência relevante na decisão, mantendo taxas semelhantes entre os grupos, com exceção de clientes com 5 empréstimos, que apresentam queda acentuada na aprovação.

**Porcentagem aceita por quantidade de empréstimos:** 0: 33,3% · 1: 29,5% · 2: 28,4% · 3: 35,0% · 4: 38,5% · 5: 11,9%

Já a situação de emprego é determinante: clientes desempregados não são aprovados, enquanto empregados e autônomos possuem taxas muito próximas (~42%).

> **Conclusão:** o modelo prioriza fortemente baixo risco histórico e estabilidade financeira.

### Taxa de Aprovação por Debt-to-Income

![Dashboard 3](images/dash3.png)

O gráfico de taxa de aprovação por comprometimento de renda (debt-to-income) mostra uma relação clara entre o nível de endividamento e a probabilidade de aprovação:

1. Clientes com comprometimento entre **0% e 25%** apresentam taxa de aprovação de 38,97%, enquanto na faixa de **25% a 50%** a taxa é ligeiramente maior, com 40,51%.
2. A partir da faixa de **50% a 75%**, a taxa de aprovação cai significativamente para 14,68%, indicando aumento relevante no risco percebido.
3. Na faixa de **75% a 100%**, a taxa de aprovação é **0%**, demonstrando que níveis muito elevados de comprometimento de renda inviabilizam a concessão de crédito.

> **Conclusão:** quanto maior o comprometimento de renda, menor a probabilidade de aprovação, sendo este um dos principais fatores de risco analisados pela instituição.



## 🧠 Principais Insights

- **Regras de negócio identificadas:** idade mínima de 24 anos, score mínimo de 550, desempregados e clientes com mais de duas inadimplências não são aprovados, e comprometimento de renda acima de 50% reduz drasticamente as chances de aprovação.
- As **variáveis mais relevantes para a classificação** foram score de crédito, status de emprego, histórico de inadimplências e percentual de renda comprometida.
- A **taxa de juros** é influenciada principalmente pelo comprometimento de renda, existência de empréstimos ativos, número de parcelas definidas e score de crédito do cliente.



## 🧱 Etapas do Projeto

### 1️⃣ Conhecimento inicial do dataset
- Leitura dos dados e primeiras inspeções
- Tipos de variáveis, estatísticas iniciais e identificação de inconsistências
- Entendimento geral da estrutura do conjunto de dados

### 2️⃣ Análise Exploratória de Dados (EDA)
- Geração de gráficos das variáveis e suas relações com a variável `approval`
- Avaliação da distribuição das variáveis numéricas e categóricas
- Verificação de padrões, tendências e possíveis outliers
- Análise comparativa entre clientes aprovados e reprovados

### 3️⃣ Pré-processamento
- Limpeza e organização dos dados
- Tratamento de valores ausentes com **KNN Imputer**
- Codificação de variáveis categóricas com **OrdinalEncoder**
- Análise da matriz de correlação com o alvo para seleção de features
- Padronização e preparação final para modelagem

### 4️⃣ Classificação — Aprovação de Empréstimo
- Seleção do modelo ideal via **GridSearchCV**
- Avaliação com `classification_report` e `matriz de confusão`
- **98% de acurácia** na classificação de aprovação

### 5️⃣ Regressão — Taxa de Juros
- Modelos avaliados: **Linear Regression** e **Lasso Regression**
- Geração de gráfico **Real vs Predito** para validação visual
- Análise de lucratividade: porcentagem lucrativa e valor investido pela instituição
- Previsão real com registro de teste para comprovação do baixo erro

### 6️⃣ Aplicação Streamlit
- Interface onde o usuário insere informações do cliente
- Retorna aprovação ou rejeição em tempo real
- Se aprovado, calcula automaticamente a taxa de juros a ser paga

---

## 📊 Desempenho do Modelo

### Classificação — Aprovação

| Métrica | Valor |
|---|---|
| Acurácia | **98%** |
| F1 Macro | **98%** |

| Classe | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Reprovado | 1.00 | 0.98 | 0.99 | 1150 |
| Aprovado | 0.93 | 1.00 | 0.96 | 350 |

### Regressão — Taxa de Juros

| Métrica | Treino | Teste |
|---|---|---|
| R² | 0.9990 | **0.9977** |
| MAE | 0.0708 | **0.0776** |

**Equação do modelo:**
```
Taxa = 20.898 − 0.020 × credit_score − 0.041 × Loan_Term_Months
     + 0.598 × Existing_Loans_Count + 6.609 × debt_to_income
     − 0.007 × Savings_Account − 0.005 × gender
```

---

## 🚀 Conclusão

O projeto entrega uma solução completa que combina:
- Modelo robusto para **classificação de aprovação** com 98% de acurácia
- Regressão eficiente para **definir taxas personalizadas** com R² de 0.9977
- **EDA aprofundada** com insights estratégicos sobre o perfil de crédito
- Interface pronta para uso via Streamlit

Essa abordagem pode ser facilmente expandida e aplicada em cenários reais de instituições financeiras.

---

## 🛠 Tecnologias Utilizadas

| Tecnologia | Função |
|---|---|
| 🐍 **Python** | Linguagem principal do projeto |
| 🧮 **Pandas / NumPy** | Manipulação e análise de dados |
| 📊 **Matplotlib / Seaborn** | Visualizações e gráficos |
| 🤖 **Scikit-learn** | Modelagem, pipeline e métricas |
| 🌲 **RandomForestClassifier** | Classificador para aprovação |
| 📉 **Linear / Lasso Regression** | Regressão para taxa de juros |
| 🔧 **KNN Imputer** | Tratamento de valores ausentes |
| 📦 **Joblib / Pickle** | Serialização e persistência de modelos |
