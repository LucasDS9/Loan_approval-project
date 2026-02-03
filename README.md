# End-to-End NLP for Product Review Analysis

## 📌 Visão Geral do Projeto

Este projeto apresenta um **pipeline completo de Processamento de Linguagem Natural (NLP)** voltado à extração de insights estruturados a partir de avaliações textuais de clientes.  
O objetivo principal é transformar **reviews não estruturadas** em **informações estratégicas em nível de produto**, combinando técnicas clássicas de NLP com **análise via Large Language Models (LLMs)**.

O pipeline cobre todas as etapas do processo:
- Ingestão dos dados
- Limpeza textual e feature engineering
- Vetorização de texto (TF-IDF e Word2Vec)
- Agregação de sentimento por produto
- Geração de métricas estruturadas para LLMs
- Criação automática de relatórios executivos utilizando LLM local (Ollama + LLaMA)

---

## 🌍 Por Que Este Projeto É Importante  
*(De Reviews a Inteligência de Negócio)*

Avaliações de clientes são uma das **fontes de dados mais valiosas e menos exploradas** pelas empresas.  
Elas contêm feedback direto sobre qualidade do produto, satisfação do cliente, problemas recorrentes e expectativas não atendidas — porém em um formato **não estruturado e difícil de analisar em escala**.

Este projeto demonstra como técnicas de NLP podem:
- Converter milhares de textos livres em **métricas estruturadas**
- Identificar **principais reclamações e pontos fortes** dos produtos
- **Quantificar o sentimento** dos clientes de forma objetiva
- **Apoiar decisões estratégicas** sem a necessidade de leitura manual das avaliações

Ao integrar NLP tradicional com LLMs, o projeto ilustra uma abordagem moderna para transformar dados textuais em **insights acionáveis e escaláveis**.

---

## 🎯 Objetivo do Projeto no Contexto Empresarial

Em um ambiente corporativo, este projeto pode ser aplicado para:

- **Monitoramento de Produtos**  
  Detectar quedas na percepção de qualidade a partir da análise de sentimento.

- **Análise da Experiência do Cliente**  
  Compreender dores e expectativas dos consumidores de forma automatizada.

- **Apoio à Tomada de Decisão**  
  Fornecer relatórios claros e objetivos para gestores e stakeholders.

- **Escalabilidade Analítica**  
  Substituir análises manuais por um pipeline automatizado de NLP + LLM.

O resultado final é um sistema projetado para gerar **informações prontas para consumo executivo**, indo além de análises puramente técnicas.

--- 
## Requisitos

Antes de executar o projeto, certifique-se de ter:

- Python 3.10+
- Ollama instalado e rodando localmente  
  https://ollama.com
- Bibliotecas Python listadas em `requirements.txt`

### Modelo LLM
Este projeto utiliza um LLM rodando localmente via **Ollama**.  
Após instalar o Ollama, baixe o modelo:

(bash): 
ollama pull llama3.1:8b


## Como usar

### Execução via script (análise automática)

1. **Instale as dependências**:
(bash):
pip install -r requirements.txt

**Execução via Streamlit (interface interativa)**
Inicie a aplicação: streamlit run app/app.py
Acesse o link exibido no terminal (ex: "http://localhost:8501").

Selecione o produto desejado e clique em Gerar análise.

A análise será exibida diretamente na interface.

**Execução no arquivo python :**
- Execute o arquivo:
llm/analyze_product.py
- Selecione um ProductId na lista exibida no terminal.

- O modelo irá gerar a análise automaticamente e o relatório será salvo em: reports/



--- 

--- 

## 📊 Dataset

O dataset utilizado neste projeto está disponível publicamente no Kaggle:

**Amazon Fine Food Reviews Dataset**  
🔗 https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

Esse conjunto de dados contém mais de **500 mil avaliações de clientes** sobre produtos alimentícios vendidos na Amazon, coletadas entre os anos de 1999 e 2012.

### Descrição das Colunas

| Coluna | Descrição |
|------|-----------|
| `Id` | Identificador único da avaliação |
| `ProductId` | Identificador único do produto |
| `UserId` | Identificador único do usuário |
| `ProfileName` | Nome do perfil do usuário |
| `HelpfulnessNumerator` | Número de usuários que consideraram a avaliação útil |
| `HelpfulnessDenominator` | Número total de usuários que avaliaram a utilidade |
| `Score` | Nota atribuída ao produto (1 a 5) |
| `Time` | Timestamp da avaliação (formato Unix) |
| `Summary` | Resumo curto da avaliação |
| `Text` | Texto completo da avaliação |

---


## 📁 Estrutura do Projeto

O projeto está organizado em módulos que representam cada etapa do pipeline de NLP, desde o tratamento dos dados até a geração automática de relatórios com LLM.
```
├── app
│   └── app.py        # Streamlit
│
├── data
│   ├── data_cleaned.csv
│   ├── text_preprocessed.csv
│   └── metrics_for_llm.csv
│
├── llm
│   ├── analyze_product.py
│   └── prompt_product_overview.txt
│
├── nlp
│   ├── 01_data_analysis_cleaning.ipynb
│   ├── 02_nlp_preprocessing.py
│   ├── 03_tfidf_and_word2vec.ipynb
│   └── 04_metrics_to_llm.py
│
├── reports
│   └── product_<product_id>.md
│
├── .gitignore
└── README.md
```

---

### 📂 Pasta `app/`

- **`app.py`**  
  Aplicação interativa desenvolvida com **Streamlit**, responsável por permitir a exploração dos produtos e a geração sob demanda das análises via LLM.

  **Fluxo da aplicação:**
  1. Carrega o arquivo `metrics_for_llm.csv`
  2. Exibe uma lista (selectbox) com os `ProductId`
  3. O usuário seleciona um produto e clica em **Gerar análise**
  4. Um spinner de carregamento é exibido durante a execução do LLM
  5. Os insights gerados são exibidos diretamente na interface

---

### 📂 Pasta `data/`

Esta pasta armazena os dados intermediários e finais gerados ao longo do pipeline.


- **`data_cleaned.csv`**  
  Dataset após a limpeza inicial, incluindo remoção de valores ausentes, ajustes de colunas e padronização dos dados.  
  Serve como base para as etapas posteriores de NLP.

- **`text_preprocessed.csv`**  
  Contém o texto das avaliações após o preprocessamento linguístico, como:
  - tokenização
  - lemmatização
  - remoção de stopwords
  - filtragem de tokens inválidos  

  Este arquivo é utilizado diretamente na etapa de vetorização textual.

- **`metrics_for_llm.csv`**  
  Dataset final com métricas agregadas por produto, incluindo:
  - quantidade de avaliações
  - nota média
  - distribuição percentual de sentimentos
  - palavras-chave extraídas via TF-IDF  

  Esse arquivo foi projetado para ser **consumido diretamente por um LLM**.

> ⚠️ O arquivo original **`Reviews.csv`**, que contém o dataset completo do Kaggle, é ignorado no `.gitignore` devido ao seu grande tamanho.

---

### 📂 Pasta `nlp/`

Responsável por todas as etapas relacionadas ao **processamento de linguagem natural e análise dos textos**.

- **`01_data_analysis_cleaning.ipynb`**  
  Notebook de análise exploratória dos dados (EDA), utilizado para entender a distribuição das avaliações, notas e características do texto, além da limpeza inicial.

- **`02_nlp_preprocessing.py`**  
  Script que executa o preprocessamento textual utilizando técnicas de NLP, garantindo consistência e reaproveitamento do código em pipelines automatizados.

- **`03_tfidf_and_word2vec.ipynb`**  
  Implementação das representações vetoriais:
  - TF-IDF para identificação de termos relevantes por produto
  - Word2Vec para captura de relações semânticas entre palavras

- **`04_metrics_to_llm.py`**  
  Script responsável por consolidar as análises e gerar métricas finais por produto, preparando os dados no formato ideal para uso por LLMs.

---

### 📂 Pasta `llm/`

Camada responsável pela **geração automática de análises textuais** a partir das métricas calculadas.

- **`prompt_product_overview.txt`**  
  Prompt estruturado que orienta o LLM a produzir análises objetivas, profissionais e baseadas exclusivamente nas métricas fornecidas.

- **`analyze_product.py`**  
  Script principal de execução do projeto.  
  Nele, o usuário seleciona um `ProductId`, o prompt é construído dinamicamente e o LLM é chamado via Ollama.

  O resultado é um relatório analítico salvo automaticamente em formato Markdown no diretório `reports/`.

---

### 📂 Pasta `reports/`

Armazena os **relatórios finais gerados pelo LLM**, um por produto analisado.

- **`product_<product_id>.md`**  
  Relatório executivo contendo:
  - análise geral de sentimento
  - interpretação da nota média
  - principais reclamações e pontos fortes
  - riscos e oportunidades de melhoria

---

Essa organização permite que o projeto funcione como um **pipeline reproduzível de NLP**, facilitando tanto o uso acadêmico quanto aplicações em ambientes corporativos.



### ✅ Exemplo gerado pelo ollama :


## 🛠️ Tecnologias Utilizadas

- **Python** — linguagem principal do projeto  
- **Pandas & NumPy** — manipulação e análise de dados  
- **spaCy** — preprocessamento de linguagem natural (tokenização, lemmatização, stopwords)  
- **Scikit-learn** — TF-IDF, vetorização e métricas  
- **Gensim** — Word2Vec para embeddings semânticos  
- **Ollama** — execução local de Large Language Models  
- **LLaMA 3.1** — geração de análises textuais automatizadas  
- **Jupyter Notebook** — análise exploratória e experimentação  
- **Markdown** — geração de relatórios executivos automatizados

## 🏁 Conclusão

Este projeto demonstra como técnicas clássicas de NLP podem ser combinadas com **Large Language Models** para transformar avaliações textuais em **insights estratégicos e acionáveis**.

Ao invés de aplicar LLMs diretamente sobre textos brutos, a abordagem adotada prioriza a geração de **métricas estruturadas**, garantindo:
- maior confiabilidade das análises
- menor risco de alucinação
- melhor controle sobre o conteúdo gerado

O pipeline desenvolvido é totalmente reproduzível, escalável e aplicável a contextos reais de negócio, como monitoramento de produtos, análise de experiência do cliente e apoio à tomada de decisão.

Esse projeto reforça a importância de unir **engenharia de dados, NLP e LLMs** de forma integrada, indo além de experimentos isolados e focando em soluções práticas e interpretáveis.
