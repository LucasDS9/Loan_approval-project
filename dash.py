import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Análise de Concessão de Crédito",
    layout="wide",
    initial_sidebar_state="collapsed",
)

plt.rcParams["axes.grid"] = False

C_DARK   = "#171f25"
C_CREAM  = "#f2eab7"
C_RED    = "#752e2b"
C_RED2   = "#8d312e"
COLORS   = [C_DARK, C_CREAM, C_RED]
custom_cmap = LinearSegmentedColormap.from_list("custom", COLORS)

BAR_PALETTE = [C_RED, C_DARK]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f1419;
    color: #e8dfc8;
}

.main { background-color: #0f1419; }
.block-container { padding: 1.5rem 2.5rem 3rem 2.5rem; }

/* ── HEADER ── */
.dash-header {
    background: linear-gradient(135deg, #171f25 0%, #1e2b35 60%, #2a1a19 100%);
    border-bottom: 3px solid #752e2b;
    padding: 2.2rem 3rem 1.8rem 3rem;
    margin: -1.5rem -2.5rem 2.5rem -2.5rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.dash-header-icon { font-size: 2.8rem; }
.dash-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.5rem;
    font-weight: 900;
    color: #f2eab7;
    letter-spacing: -0.5px;
    line-height: 1.1;
    margin: 0;
}
.dash-subtitle {
    color: #9aaa9a;
    font-size: 0.92rem;
    font-weight: 300;
    margin-top: 0.3rem;
}

/* ── KPI CARDS ── */
.kpi-row { display: flex; gap: 1.2rem; margin-bottom: 2rem; }
.kpi-card {
    flex: 1;
    background: #171f25;
    border: 1px solid #2e3d4a;
    border-top: 3px solid #752e2b;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    transition: box-shadow 0.2s;
}
.kpi-card:hover { box-shadow: 0 6px 24px rgba(117,46,43,0.25); }
.kpi-label { font-size: 0.78rem; font-weight: 500; color: #7a8a7a; text-transform: uppercase; letter-spacing: 1px; }
.kpi-value { font-family: 'Playfair Display', serif; font-size: 2.1rem; font-weight: 700; color: #f2eab7; line-height: 1.2; }
.kpi-delta { font-size: 0.82rem; color: #9aaa9a; margin-top: 0.2rem; }

/* ── SECTION HEADERS ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #f2eab7;
    border-left: 4px solid #752e2b;
    padding-left: 0.8rem;
    margin: 2.2rem 0 1rem 0;
}

/* ── CHART CARDS ── */
.chart-card {
    background: #171f25;
    border: 1px solid #2e3d4a;
    border-radius: 12px;
    padding: 1.2rem 1.2rem 0.8rem 1.2rem;
    margin-bottom: 0.2rem;
}

/* ── METRICS TABLE ── */
.metric-table { width: 100%; border-collapse: collapse; }
.metric-table th {
    background: #752e2b;
    color: #f2eab7;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 0.6rem 0.8rem;
    text-align: left;
}
.metric-table td {
    padding: 0.55rem 0.8rem;
    font-size: 0.88rem;
    border-bottom: 1px solid #2e3d4a;
    color: #d4c9a8;
}
.metric-table tr:nth-child(even) td { background: #1e2b35; }

/* ── DIVIDER ── */
hr.section-hr {
    border: none;
    border-top: 1px solid #2e3d4a;
    margin: 2rem 0;
}

/* Streamlit overrides */
[data-testid="stMetric"] {
    background: #171f25;
    border: 1px solid #2e3d4a;
    border-top: 3px solid #752e2b;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: #9aaa9a !important; font-size: 0.8rem; }
[data-testid="stMetricValue"] { color: #f2eab7 !important; font-family: 'Playfair Display', serif; }
</style>
""", unsafe_allow_html=True)

# ─── MATPLOTLIB GLOBAL STYLE ─────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#171f25",
    "axes.facecolor":    "#171f25",
    "axes.edgecolor":    "#2e3d4a",
    "axes.labelcolor":   "#c8bfa0",
    "axes.titlecolor":   "#f2eab7",
    "xtick.color":       "#8a9a8a",
    "ytick.color":       "#8a9a8a",
    "text.color":        "#c8bfa0",
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.facecolor":  "#1e2b35",
    "legend.edgecolor":  "#2e3d4a",
    "legend.fontsize":   8,
})

@st.cache_data
def load_data():
    df = pd.read_excel("data/Loan_dataset.xlsx")
    df.drop(columns=['ID', 'cpf'], inplace=True, errors='ignore')
    return df

@st.cache_data
def run_models(df_raw):
    df = df_raw.copy()
    df['dti_group'] = pd.cut(
        df['debt_to_income'],
        bins=[0, 0.25, 0.5, 0.75, 1],
        labels=['0-25%', '25-50%', '50-75%', '75-100%']
    )
    enc = OrdinalEncoder()
    df[['gender', 'Savings_Account', 'employment_status']] = enc.fit_transform(
        df[['gender', 'Savings_Account', 'employment_status']]
    )
    df.drop(columns=['dti_group'], inplace=True, errors='ignore')

    corr = df.corr()[['approved']].sort_values(by='approved', ascending=False)

    x = df.drop(columns=['approved', 'interest_rate'])
    y = df['approved']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    approved_df = df[df['approved'] == 1].copy()
    Xr = approved_df.drop(columns=['approved', 'interest_rate'])
    yr = approved_df['interest_rate']
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.3, random_state=0)
    Xr_train['age'].fillna(df['age'].median(), inplace=True)
    Xr_test['age'].fillna(df['age'].median(), inplace=True)
    imp = KNNImputer(n_neighbors=5)
    cols_imp = ['credit_score', 'Previous_Delinquencies', 'debt_to_income']
    Xr_train[cols_imp] = imp.fit_transform(Xr_train[cols_imp])
    Xr_test[cols_imp] = imp.transform(Xr_test[cols_imp])
    reg = LinearRegression()
    reg.fit(Xr_train, yr_train)
    yr_pred = reg.predict(Xr_test)

    return corr, cm, report, yr_test, yr_pred

try:
    df = load_data()
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.info("Certifique-se de que o arquivo `data/Loan_dataset.xlsx` está na pasta correta.")
    st.stop()

corr, cm, report, yr_test, yr_pred = run_models(df)

st.markdown("""
<div class="dash-header">
  <div class="dash-header-icon">💳</div>
  <div>
    <p class="dash-title">Análise de Concessão de Crédito</p>
    <p class="dash-subtitle">Exploração de dados · Modelagem preditiva · Regressão de taxa de juros</p>
  </div>
</div>
""", unsafe_allow_html=True)

total      = len(df)
approved   = df['approved'].sum()
denied     = total - approved
aprov_rate = approved / total * 100
avg_score  = df['credit_score'].mean()
avg_dti    = df['debt_to_income'].mean()
avg_salary = df['salary'].mean()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total de Solicitações", f"{total:,}".replace(",", "."))
with c2:
    st.metric("Aprovações", f"{int(approved):,}".replace(",", "."))
with c3:
    st.metric("Reprovações", f"{int(denied):,}".replace(",", "."))
with c4:
    st.metric("Taxa de Aprovação", f"{aprov_rate:.1f}%")

st.markdown('<hr class="section-hr">', unsafe_allow_html=True)

st.markdown('<p class="section-title">📊 Visão Geral de Aprovações</p>', unsafe_allow_html=True)

col_pie, col_age, col_score = st.columns([1, 2, 2])

with col_pie:
    with st.container():
        fig, ax = plt.subplots(figsize=(4, 4))
        counts = df['approved'].value_counts().sort_index()
        wedge_colors = [C_RED, C_CREAM]
        wedges, texts, autotexts = ax.pie(
            counts,
            autopct='%1.1f%%',
            wedgeprops=dict(width=0.25, edgecolor="#0f1419", linewidth=2),
            colors=wedge_colors,
            startangle=90,
        )
        for at in autotexts:
            at.set_color("#f2eab7")
            at.set_fontsize(11)
            at.set_fontweight("bold")
        legend_patches = [
            mpatches.Patch(color=C_RED, label="Não aprovado"),
            mpatches.Patch(color=C_CREAM, label="Aprovado"),
        ]
        ax.legend(handles=legend_patches, loc="lower center",
                  bbox_to_anchor=(0.5, -0.12), ncol=2, framealpha=0.3)
        ax.set_title("Taxa de Aprovação", pad=14)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

with col_age:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(
        data=df, x='age', bins=20, hue='approved',
        multiple='stack',
        palette=[C_RED, C_CREAM],
        edgecolor="#0f1419", alpha=1, ax=ax
    )
    legend_patches = [
        mpatches.Patch(color=C_RED, label="Não aprovado"),
        mpatches.Patch(color=C_CREAM, label="Aprovado"),
    ]
    ax.legend(handles=legend_patches, title="Aprovado", loc="lower center")
    ax.set_title("Aprovação por Faixa Etária")
    ax.set_xlabel("Idade")
    ax.set_ylabel("Contagem")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col_score:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(
        data=df, x='credit_score', bins=20, hue='approved',
        multiple='stack',
        palette=[C_RED, C_CREAM],
        edgecolor="#0f1419", alpha=1, ax=ax
    )
    legend_patches = [
        mpatches.Patch(color=C_RED, label="Não aprovado"),
        mpatches.Patch(color=C_CREAM, label="Aprovado"),
    ]
    ax.legend(handles=legend_patches, title="Aprovado")
    ax.set_title("Aprovação por Score de Crédito")
    ax.set_xlabel("Credit Score")
    ax.set_ylabel("Contagem")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

st.markdown("""
<div style="
    background-color: #171f25;
    border: 1px solid #333333;
    padding: 18px;
    border-radius: 8px;
    margin: 15px 0;
">

<h4 style="color:#ffffff;">📊 Análise Geral</h4>

<p style="color:#ffffff;">
1. A distribuição de aprovação mostra que <b>23.7%</b> dos clientes (1.187) foram aprovados, enquanto 
<b>3.813</b> tiveram suas tentativas de empréstimo negadas. Isso implica que a instituição possui 
um sistema rigoroso de aprovação. Nos gráficos seguintes, serão apresentados os fatores que influenciam essa decisão.
</p>

<p style="color:#ffffff;">
2. O gráfico de aprovação por faixa etária mostra que a instituição não aprovou pessoas com menos de 
<b>24 anos</b>. Para clientes acima dessa idade, há pouca influência da variável na decisão.
</p>

<p style="color:#ffffff;">
3. No gráfico de aprovação por score de crédito, observa-se explicitamente que a instituição impôs 
um limite mínimo de <b>550</b> para aprovação, indicando a existência de um critério de corte bem definido.
</p>

</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="section-hr">', unsafe_allow_html=True)

st.markdown('<p class="section-title">⚠️ Fatores de Risco e Aprovação</p>', unsafe_allow_html=True)

col_delinq, col_loans, col_emp = st.columns(3)

with col_delinq:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.countplot(
        data=df, x='Previous_Delinquencies', hue='approved',
        palette=[C_RED, C_CREAM],
        edgecolor="#0f1419", ax=ax
    )
    for container in ax.containers:
        ax.bar_label(container, fontsize=7, color="#c8bfa0", padding=2)
    legend_patches = [
        mpatches.Patch(color=C_RED, label="Não aprovado"),
        mpatches.Patch(color=C_CREAM, label="Aprovado"),
    ]
    ax.legend(handles=legend_patches)
    ax.set_title("Inadimplências Anteriores vs Aprovação")
    ax.set_xlabel("Nº de Inadimplências")
    ax.set_ylabel("Contagem")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with col_loans:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.countplot(
        data=df, x='Existing_Loans_Count', hue='approved',
        palette=[C_RED, C_CREAM],
        edgecolor="#0f1419", ax=ax
    )
    for container in ax.containers:
        ax.bar_label(container, fontsize=7, color="#c8bfa0", padding=2)
    legend_patches = [
        mpatches.Patch(color=C_RED, label="Não aprovado"),
        mpatches.Patch(color=C_CREAM, label="Aprovado"),
    ]
    ax.legend(handles=legend_patches)
    ax.set_title("Qtd. Empréstimos Ativos vs Aprovação")
    ax.set_xlabel("Empréstimos Existentes")
    ax.set_ylabel("Contagem")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with col_emp:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.countplot(
        data=df, x='employment_status', hue='approved',
        palette=[C_RED, C_CREAM],
        edgecolor="#0f1419", ax=ax
    )
    for container in ax.containers:
        ax.bar_label(container, fontsize=7, color="#c8bfa0", padding=2)
    legend_patches = [
        mpatches.Patch(color=C_RED, label="Não aprovado"),
        mpatches.Patch(color=C_CREAM, label="Aprovado"),
    ]
    ax.legend(handles=legend_patches)
    ax.set_title("Vínculo Empregatício vs Aprovação")
    ax.set_xlabel("Status de Emprego")
    ax.set_ylabel("Contagem")
    plt.xticks(rotation=10)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

st.markdown("""
<div style="
    background-color: #171f25;
    border: 1px solid #333333;
    padding: 18px;
    border-radius: 8px;
    margin: 15px 0;
">

<h4 style="color:#ffffff;">📊 Resumo Executivo</h4>

<p style="color:#ffffff;">
A aprovação de crédito é fortemente impactada pelo histórico financeiro do cliente.
Clientes <b>sem delinquências</b> concentram a grande maioria das aprovações (86,9%) e apresentam
taxa significativamente superior (~54%), enquanto qualquer histórico negativo reduz drasticamente as chances — 
com <b>nenhuma aprovação</b> para 3 ou mais ocorrências.
</p>

<p style="color:#ffffff;">
O número de empréstimos atuais <b>não demonstra influência relevante</b> na decisão, mantendo taxas semelhantes entre os grupos,
com exceção de clientes com 5 empréstimos, que apresentam queda acentuada na aprovação.
</p>

<p style="color:#ffffff;">
<b>Porcentagem aceita por quantidade de empréstimos:</b><br>
0: 33,3% , 1: 29,5% , 2: 28,4% , 3: 35,0% , 4: 38,5% , 5: 11,9%
</p>

<p style="color:#ffffff;">
Já a <b>situação de emprego</b> é determinante: clientes desempregados não são aprovados,
enquanto empregados e autônomos possuem taxas muito próximas (~42%).
</p>

<p style="color:#ffffff;">
Conclusão: o modelo prioriza fortemente <b>baixo risco histórico</b> e <b>estabilidade financeira</b>.
</p>

</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="section-hr">', unsafe_allow_html=True)

st.markdown('<p class="section-title">💰 Taxa de Aprovação por Debt-to-Income</p>', unsafe_allow_html=True)

df_dti = df.copy()
df_dti['dti_group'] = pd.cut(
    df_dti['debt_to_income'],
    bins=[0, 0.25, 0.5, 0.75, 1],
    labels=['0–25%', '25–50%', '50–75%', '75–100%']
)
approval_rate = df_dti.groupby('dti_group')['approved'].mean().reset_index()

col_dti_l, col_dti, col_dti_r = st.columns([1, 4, 1])
with col_dti:
    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colors = [C_DARK, C_CREAM, C_RED, C_RED2]
    bars = ax.bar(
        approval_rate['dti_group'].astype(str),
        approval_rate['approved'],
        color=bar_colors,
        edgecolor="#0f1419",
        linewidth=1.2,
        width=0.55
    )
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.005,
            f"{h:.1%}",
            ha='center', va='bottom',
            fontsize=11, fontweight='bold',
            color="#f2eab7"
        )
    ax.set_title("Taxa de Aprovação por Faixa de Comprometimento de Renda",
                 fontsize=13, pad=16)
    ax.set_xlabel("Faixa de Comprometimento de Renda", fontsize=10)
    ax.set_ylabel("Taxa de Aprovação", fontsize=10)
    ax.set_ylim(0, approval_rate['approved'].max() * 1.2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

st.markdown("""
<div style="
    background-color: #171f25;
    border: 1px solid #333333;
    padding: 18px;
    border-radius: 8px;
    margin: 15px 0;
">

<h4 style="color:#ffffff;">📊 Análise de Debt-to-Income</h4>

<p style="color:#ffffff;">
O gráfico de taxa de aprovação por comprometimento de renda (debt-to-income) mostra uma relação clara 
entre o nível de endividamento e a probabilidade de aprovação:
</p>

<p style="color:#ffffff;">
1. Clientes com comprometimento entre <b>0% e 25%</b> apresentam taxa de aprovação de <b>38,97%</b>, 
enquanto na faixa de <b>25% a 50%</b> a taxa é ligeiramente maior, com <b>40,51%</b>.
</p>

<p style="color:#ffffff;">
2. A partir da faixa de <b>50% a 75%</b>, a taxa de aprovação cai significativamente para <b>14,68%</b>, 
indicando aumento relevante no risco percebido.
</p>

<p style="color:#ffffff;">
3. Na faixa de <b>75% a 100%</b>, a taxa de aprovação é <b>0%</b>, demonstrando que níveis muito elevados 
de comprometimento de renda inviabilizam a concessão de crédito.
</p>

<p style="color:#ffffff;">
Conclusão: quanto maior o comprometimento de renda, menor a probabilidade de aprovação, sendo este um dos principais fatores de risco analisados pela instituição.
</p>

</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="section-hr">', unsafe_allow_html=True)

st.markdown('<p class="section-title">🤖 Análise de Correlação e Modelo Preditivo</p>', unsafe_allow_html=True)

col_corr, col_cm, col_imp = st.columns(3)

with col_corr:
    fig, ax = plt.subplots(figsize=(5, 5.5))
    sns.heatmap(
        corr, annot=True, cmap=custom_cmap, fmt='.3f',
        linewidths=0.5, linecolor="#0f1419",
        cbar_kws={'shrink': 0.8},
        ax=ax
    )
    ax.set_title("Correlação com Aprovação", pad=14)
    ax.tick_params(axis='y', labelsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col_cm:
    fig, ax = plt.subplots(figsize=(5, 5.5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap=custom_cmap,
        linewidths=0.5, linecolor="#0f1419",
        cbar_kws={'shrink': 0.8},
        ax=ax
    )
    ax.set_xlabel("Predito", fontsize=9)
    ax.set_ylabel("Real", fontsize=9)
    ax.set_title("Matriz de Confusão (Random Forest)", pad=14)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with col_imp:
    fig, ax = plt.subplots(figsize=(5, 5.5))

    data = {
        'debt_to_income': 6.609,
        'Existing_Loans_Count': 0.598,
        'Loan_Term_Months': -0.041,
        'credit_score': -0.020
    }

    df_plot = pd.DataFrame.from_dict(
        data, orient='index', columns=['impact']
    )

    sns.heatmap(
        df_plot,
        annot=True,
        cmap=custom_cmap,
        center=0,
        fmt='.3f',
        linewidths=0.5,
        linecolor="#0f1419",
        cbar_kws={'shrink': 0.8},
        ax=ax
    )

    ax.set_title("Impacto das Variáveis na Taxa de Juros", pad=14)

    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)

acc     = report['accuracy']
prec_1  = report.get('1', report.get('1.0', {})).get('precision', 0)
rec_1   = report.get('1', report.get('1.0', {})).get('recall', 0)
f1_1    = report.get('1', report.get('1.0', {})).get('f1-score', 0)
r2      = r2_score(yr_test, yr_pred)
mae     = mean_absolute_error(yr_test, yr_pred)

with col_m1:
    st.metric("Acurácia (Classificação)", f"{acc:.1%}")
with col_m2:
    st.metric("Precisão (Aprovado)", f"{prec_1:.1%}")
with col_m3:
    st.metric("R² — Regressão", f"{r2:.4f}")
with col_m4:
    st.metric("MAE — Regressão", f"{mae:.4f}")

st.markdown("""
<div style="
    background-color: #171f25;
    border: 1px solid #333333;
    padding: 18px;
    border-radius: 8px;
    margin: 15px 0;
">

<h4 style="color:#ffffff;">📊 Análise Integrada do Modelo</h4>

<p style="color:#ffffff;">
A análise conjunta dos gráficos permite entender tanto os fatores de decisão quanto o desempenho do modelo:
</p>

<p style="color:#ffffff;">
1. Na <b>correlação com aprovação</b>, o <b>interest rate</b> apresenta valor elevado (0.899), porém não é relevante para decisão,
pois está presente apenas após a aprovação, sendo removido do treinamento para evitar vazamento de informação.
O <b>credit score</b> se destaca como principal variável que influencia a decisão positivamente (Quanto maior o score, mais chance de aprovação), enquanto <b>debt-to-income</b>, 
<b>previous delinquencies</b> e <b>employment status</b> apresentam correlação negativa (maior dívida em relação ao rendimento e mais delinquências reduzem a probabilidade de aprovação e também a Situação profissional)
, indicando maior risco.
</p>

<p style="color:#ffffff;">
2. A <b>matriz de confusão</b> mostra um modelo com bom desempenho, com <b>350 verdadeiros positivos</b> e 
<b>1123 verdadeiros negativos</b>, além de <b>zero falsos negativos</b>, indicando alta capacidade de identificar clientes aprovados.
</p>

<p style="color:#ffffff;">
3. No <b>impacto na taxa de juros</b>, o <b>debt-to-income</b> é a variável mais influente, aumentando significativamente a taxa conforme o risco.
As demais variáveis menos impacto mas influenciam no número final.
</p>

<p style="color:#ffffff;">
Conclusão: o modelo prioriza variáveis de risco financeiro tanto na aprovação quanto na definição da taxa de juros.
</p>     
                   
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; padding: 2.5rem 0 1rem 0; color: #4a5a4a; font-size: 0.78rem;">
    Análise de Concessão de Crédito &nbsp;·&nbsp; Dashboard Streamlit &nbsp;·&nbsp; Paleta: #171f25 · #f2eab7 · #752e2b
</div>
""", unsafe_allow_html=True)