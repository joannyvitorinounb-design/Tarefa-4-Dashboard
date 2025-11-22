import gradio as gr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

def process_file(file):
    df = pd.read_csv(file.name, sep='\t')

    return df.head().to_html()

def train_model(file, target, features, model_choice):
    df = pd.read_csv(file.name)

    X = df[features]
    y = df[target]

    sm = SMOTE()
    X_res, y_res = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)

    if model_choice == "Regressão Logística":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = LogisticRegression()
    else:
        model = RandomForestClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return f"""
    <b>AUC:</b> {auc}<br>
    <b>Precisão:</b> {prec}<br>
    <b>Recall:</b> {rec}<br>
    <b>F1:</b> {f1}<br>
    <b>Matriz de confusão:</b><br>{cm}
    """

with gr.Blocks() as demo:
    gr.Markdown("# 📊 Dashboard da Tarefa 4 — Bônus")
    file = gr.File(label="Envie seu CSV")
    
    show_head = gr.Button("Mostrar primeiras linhas")
    head_output = gr.HTML()

    with gr.Row():
        target = gr.Textbox(label="Variável alvo (Y)")
        features = gr.Textbox(label="Features (X) separadas por vírgula")

    model_choice = gr.Dropdown(["Regressão Logística", "Random Forest"], label="Modelo")

    train_button = gr.Button("Treinar Modelo")
    train_output = gr.HTML()

    show_head.click(process_file, inputs=file, outputs=head_output)
    train_button.click(train_model, inputs=[file, target, features, model_choice], outputs=train_output)

demo.launch()
