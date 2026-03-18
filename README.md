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
│
└── README.md
```

---

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

## 🧠 Principais Insights

- O **credit score** e o **debt-to-income ratio** são os fatores com maior impacto na taxa de juros
- Clientes com mais empréstimos existentes recebem taxas maiores, refletindo maior risco percebido
- O modelo de regressão obteve R² de 0.9977 no teste, demonstrando altíssima precisão na precificação
- A análise de lucratividade ao final do notebook demonstra o impacto financeiro direto do modelo para a instituição

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