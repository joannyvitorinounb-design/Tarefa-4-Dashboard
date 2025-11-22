
# streamlit_app.py
# -------------------------------------------------------------
# Dashboard interativo para Previsão de Reclamações (Complain)
# -------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, precision_score,
    recall_score, f1_score, classification_report
)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

# Tenta importar XGBoost/LightGBM se estiverem instalados
HAS_XGB = False
HAS_LGB = False
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    pass
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    pass

st.set_page_config(page_title="Previsão de Reclamações", layout="wide")
st.title("Tarefa 4 – Dashboard Interativo: Previsão de Reclamações (Complain)")

st.markdown("""
**Como usar:**
1. Faça upload do arquivo `marketing_campaign.csv` (Kaggle, separador TAB `\\t`) ou coloque-o no mesmo diretório do app.
2. Selecione filtros, variáveis e algoritmo.
3. Ajuste o *threshold* (ou use otimizador) para otimizar a métrica desejada.
4. Veja métricas, curva ROC, matriz de confusão e importância das variáveis.
""")

# -------------------------------
# Carregamento de dados
# -------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str):
    df = pd.read_csv(path, sep='\t', encoding='latin1')
    # features derivadas (iguais às usadas no seu HTML)
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')
    df['Age'] = pd.Timestamp('today').year - df['Year_Birth']
    spend_cols = ['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']
    df['Total_Spent'] = df[spend_cols].sum(axis=1)
    purchase_cols = ['NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases']
    df['Total_Purchases'] = df[purchase_cols].sum(axis=1)
    df['Total_AcceptedCmp'] = df[['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Response']].sum(axis=1)
    df['Income'] = df['Income'].fillna(df['Income'].median())
    return df

uploaded = st.file_uploader("Carregue o arquivo marketing_campaign.csv (separador TAB)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded, sep='\t', encoding='latin1')
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')
    df['Age'] = pd.Timestamp('today').year - df['Year_Birth']
    spend_cols = ['MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']
    df['Total_Spent'] = df[spend_cols].sum(axis=1)
    purchase_cols = ['NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases']
    df['Total_Purchases'] = df[purchase_cols].sum(axis=1)
    df['Total_AcceptedCmp'] = df[['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Response']].sum(axis=1)
    df['Income'] = df['Income'].fillna(df['Income'].median())
else:
    default_path = os.path.join(os.getcwd(), 'marketing_campaign.csv')
    if os.path.exists(default_path):
        df = load_data(default_path)
    else:
        st.warning("⚠️ Forneça o arquivo para continuar.")
        st.stop()

# -------------------------------
# Exploração & Filtros
# -------------------------------
st.subheader("Exploração & Filtros")
col1, col2, col3 = st.columns(3)
with col1:
    min_income, max_income = float(df['Income'].min()), float(df['Income'].max())
    income_range = st.slider("Faixa de Income", min_value=min_income, max_value=max_income,
                             value=(min_income, max_income))
with col2:
    min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
    age_range = st.slider("Faixa de Age", min_value=min_age, max_value=max_age, value=(min_age, max_age))
with col3:
    edu_opts = sorted(df['Education'].dropna().unique().tolist())
    sel_edu = st.multiselect("Education", edu_opts, default=edu_opts)

df_f = df[(df['Income']>=income_range[0]) & (df['Income']<=income_range[1]) &
          (df['Age']>=age_range[0]) & (df['Age']<=age_range[1]) &
          (df['Education'].isin(sel_edu))]
st.markdown(f"**Registros após filtros:** {len(df_f)}")

# -------------------------------
# Seleção dinâmica de variáveis
# -------------------------------
available_num = ['Age','Income','Kidhome','Teenhome','Recency','Total_Spent','Total_Purchases','Total_AcceptedCmp','NumWebVisitsMonth']
available_cat = ['Education','Marital_Status']

st.subheader("Variáveis para modelagem")
sel_num = st.multiselect("Numéricas", available_num, default=available_num)
sel_cat = st.multiselect("Categóricas", available_cat, default=available_cat)

model_df = df_f[sel_num + sel_cat + ['Complain']].dropna()

# -------------------------------
# Pré-processamento + SMOTE
# -------------------------------
num_cols = sel_num
cat_cols = sel_cat

# Compatibilidade OneHotEncoder (sklearn >=1.2 usa sparse_output; versões antigas usam sparse)
try:
    categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
except TypeError:
    categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

numeric_transformer = Pipeline([('scaler', StandardScaler())])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

X = model_df.drop(columns='Complain')
y = model_df['Complain']

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                            random_state=42, stratify=y)
X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

# SMOTE com k_neighbors adaptado ao tamanho da classe minoritária
pos_count = int((y_train==1).sum())
k_smote = 3 if pos_count >= 3 else 1
sm = SMOTE(random_state=42, k_neighbors=k_smote)
X_res, y_res = sm.fit_resample(X_train, y_train)
st.caption(f"SMOTE aplicado com k_neighbors={k_smote}. Classe positiva no treino: {int((y_train==1).sum())}")

# -------------------------------
# Algoritmo & Hiperparâmetros
# -------------------------------
st.subheader("Algoritmo & Hiperparâmetros")
alg_opts = {
    'Logistic Regression': LogisticRegression(max_iter=4000),
    'KNN': KNeighborsClassifier(),
    'SVM (RBF)': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'MLPClassifier (Rede Neural)': MLPClassifier(hidden_layer_sizes=(50,30), max_iter=500, learning_rate='adaptive', random_state=42)
}
if HAS_XGB:
    alg_opts['XGBoost'] = xgb.XGBClassifier(eval_metric='logloss')
if HAS_LGB:
    alg_opts['LightGBM'] = lgb.LGBMClassifier()

alg_name = st.selectbox("Algoritmo", list(alg_opts.keys()), index=0)
model = alg_opts[alg_name]

with st.expander("Ajuste de hiperparâmetros"):
    if alg_name == 'KNN':
        n_neighbors = st.slider("n_neighbors", 1, 25, 5)
        model.set_params(n_neighbors=n_neighbors)
    elif alg_name == 'SVM (RBF)':
        C = st.number_input("C", 0.01, 100.0, 1.0)
        gamma = st.select_slider("gamma", options=['scale','auto'])
        model.set_params(C=C, gamma=gamma)
    elif alg_name == 'Decision Tree':
        max_depth = st.slider("max_depth", 1, 20, 5)
        model.set_params(max_depth=max_depth)
    elif alg_name == 'Random Forest':
        n_estimators = st.slider("n_estimators", 50, 500, 200, step=50)
        max_depth = st.slider("max_depth", 2, 30, 10)
        model.set_params(n_estimators=n_estimators, max_depth=max_depth)
    elif alg_name == 'AdaBoost':
        n_estimators = st.slider("n_estimators", 50, 500, 100, step=50)
        learning_rate = st.slider("learning_rate", 0.01, 2.0, 1.0)
        model.set_params(n_estimators=n_estimators, learning_rate=learning_rate)
    elif alg_name == 'Gradient Boosting':
        n_estimators = st.slider("n_estimators", 50, 500, 100, step=50)
        learning_rate = st.slider("learning_rate", 0.01, 1.0, 0.1)
        max_depth = st.slider("max_depth", 2, 10, 3)
        model.set_params(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    elif alg_name == 'XGBoost' and HAS_XGB:
        n_estimators = st.slider("n_estimators", 50, 500, 200, step=50)
        max_depth = st.slider("max_depth", 2, 12, 6)
        learning_rate = st.slider("learning_rate", 0.01, 0.5, 0.1)
        subsample = st.slider("subsample", 0.5, 1.0, 0.8)
        colsample_bytree = st.slider("colsample_bytree", 0.5, 1.0, 0.8)
        model.set_params(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                         subsample=subsample, colsample_bytree=colsample_bytree, eval_metric='logloss')
    elif alg_name == 'LightGBM' and HAS_LGB:
        n_estimators = st.slider("n_estimators", 50, 1000, 200, step=50)
        num_leaves = st.slider("num_leaves", 8, 128, 31)
        learning_rate = st.slider("learning_rate", 0.01, 0.5, 0.05)
        model.set_params(n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=learning_rate)
    elif alg_name == 'MLPClassifier (Rede Neural)':
        h1 = st.slider("Camada 1 (neurônios)", 10, 200, 50, 10)
        h2 = st.slider("Camada 2 (neurônios)", 10, 200, 30, 10)
        max_iter = st.slider("max_iter", 100, 2000, 500, 100)
        alpha = st.number_input("alpha (regularização)", 0.0001, 0.01, 0.0001, format="%.4f")
        model.set_params(hidden_layer_sizes=(h1, h2), max_iter=max_iter, alpha=alpha)

# -------------------------------
# Treino
# -------------------------------
model.fit(X_res, y_res)
# Alguns modelos (SVC) precisam predict_proba; já setamos probability=True
proba = model.predict_proba(X_test)[:,1]

# -------------------------------
# Otimizador de threshold (opcional) + threshold manual
# -------------------------------
st.subheader("Threshold de decisão")
opt = st.checkbox("Otimizar automaticamente o threshold", value=True)
metric_target = st.selectbox("Métrica alvo para otimizar", ["F1", "Recall", "Precisão", "Youden J (TPR-FPR)"], index=0)

if opt:
    thresholds = np.linspace(0, 1, 101)
    best_th, best_val = 0.5, -1
    for th in thresholds:
        y_tmp = (proba >= th).astype(int)
        if metric_target == "F1":
            val = f1_score(y_test, y_tmp, zero_division=0)
        elif metric_target == "Recall":
            val = recall_score(y_test, y_tmp, zero_division=0)
        elif metric_target == "Precisão":
            val = precision_score(y_test, y_tmp, zero_division=0)
        else:
            fpr, tpr, _ = roc_curve(y_test, proba >= th)
            # Youden J = max(tpr - fpr)
            val = float(np.max(tpr - fpr))
        if val > best_val:
            best_val, best_th = val, th
    th = best_th
    st.success(f"Threshold otimizado ({metric_target}): {th:.2f}")
else:
    th = st.slider("Defina o limiar (0–1)", 0.0, 1.0, 0.5, 0.01)

y_pred = (proba >= th).astype(int)

# -------------------------------
# Métricas
# -------------------------------
auc = roc_auc_score(y_test, proba)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

colA, colB, colC, colD = st.columns(4)
colA.metric("AUC", f"{auc:.3f}")
colB.metric("Precisão", f"{prec:.3f}")
colC.metric("Recall", f"{rec:.3f}")
colD.metric("F1-Score", f"{f1:.3f}")

st.subheader("Matriz de Confusão")
st.write(pd.DataFrame(cm, index=["Classe 0","Classe 1"], columns=["Pred 0","Pred 1"]))

# Curva ROC
st.subheader("Curva ROC")
fpr, tpr, _ = roc_curve(y_test, proba)
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
ax.plot([0,1],[0,1], 'k--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# -------------------------------
# Seleção de Variáveis (RFE) – opcional
# -------------------------------
st.subheader("Seleção de Variáveis (RFE) – opcional")
use_rfe = st.checkbox("Aplicar RFE (com Logistic Regression)", value=False)
if use_rfe:
    lr_base = LogisticRegression(max_iter=3000)
    rfe_n = st.slider("Número de variáveis", 5, min(25, X_res.shape[1]), 10)
    rfe = RFE(estimator=lr_base, n_features_to_select=rfe_n)
    rfe.fit(X_res, y_res)
    cat_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols)
    feat_names = num_cols + list(cat_names)
    selected = [f for f, s in zip(feat_names, rfe.support_) if s]
    st.write("Selecionadas:", selected)

# -------------------------------
# Importância de Variáveis (Permutation Importance)
# -------------------------------
st.subheader("Importância de Variáveis (Permutation Importance)")
from sklearn.inspection import permutation_importance

cat_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols)
feature_names = num_cols + list(cat_names)

perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
pi_df = pd.DataFrame({"feature": feature_names, "importance_mean": perm.importances_mean}).sort_values(
    "importance_mean", ascending=False)

st.dataframe(pi_df.head(15))
fig2, ax2 = plt.subplots(figsize=(6,6))
ax2.barh(pi_df.head(15)['feature'], pi_df.head(15)['importance_mean'])
ax2.invert_yaxis()
ax2.set_xlabel("Importância média")
ax2.set_title("Top 15 features")
st.pyplot(fig2)

# -------------------------------
# Interpretação Gerencial
# -------------------------------
st.subheader("Interpretação Gerencial")
st.markdown("""
**Sugestões de ação (exemplos):**
- Priorizar contato proativo com clientes de alta `Total_Spent` e alta `Recency` (tempo grande desde a última compra).
- Monitorar famílias com `Kidhome`/`Teenhome` > 0 para ofertas e suporte dedicado.
- Otimizar canais digitais para reduzir `NumWebVisitsMonth` com baixa conversão, evitando frustração.
""")
st.caption("Obs.: recomendações baseadas nas variáveis mais importantes e métricas atuais. Ajuste o threshold para metas (ex.: maximizar recall).")

# -------------------------------
# Relatório de Classificação + export
# -------------------------------
st.subheader("Relatório de Classificação")
report_text = classification_report(y_test, y_pred, digits=3)
st.text(report_text)
st.download_button("⬇️ Baixar relatório (txt)", report_text, file_name="classification_report.txt")
st.download_button("⬇️ Baixar importância (csv)", pi_df.to_csv(index=False), file_name="permutation_importance.csv")

st.markdown(":sparkles: **Feito para a Tarefa 4 (UnB/FT/EPR) – Bônus de Inovação**")
