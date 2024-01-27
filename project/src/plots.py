# Visualizzazione dei grafici
from io import BytesIO

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go
import matplotlib.ticker as mtick

# Gestione dei dati
import numpy as np

# Libreria per calcolare l'importanza delle features
from sklearn.inspection import permutation_importance

# Libreria per calcolare la matrice di confusione
from sklearn.metrics import confusion_matrix


def draw_candlestick_plot(stock, mode = None):

    # Crea un grafico a candela per visualizzare il prezzo delle azioni
    candlestick = go.Candlestick(x=stock.index,
                                 open=stock['Open'],
                                 high=stock['High'],
                                 low=stock['Low'],
                                 close=stock['Adj Close'],
                                 increasing=dict(line=dict(color='black')),
                                 decreasing=dict(line=dict(color='red')),
                                 showlegend=False)

    # Layout
    layout = go.Layout(
        title='Adjusted Berkshire Hathaway Class B Shares Price - 1996 to 2023',
        yaxis=dict(title='Price (USD)'),
        xaxis=dict(title='Date'),
        template='ggplot2',
        xaxis_rangeslider_visible=False,
        yaxis_gridcolor='white',
        xaxis_gridcolor='white',
        yaxis_tickfont=dict(color='black'),
        xaxis_tickfont=dict(color='black'),
        margin=dict(t=50, l=50, r=50, b=50)
    )

    fig = go.Figure(data=[candlestick], layout=layout)

    # Plotting annotation
    fig.add_annotation(text='Berkshire Hathaway Class B (BRK-B)',
                       font=dict(color='gray', size=30),
                       xref='paper', yref='paper',
                       x=0.5, y=0.5,
                       showarrow=False,
                       opacity=.85)

    if mode is None:
        fig.show()
    else:
        # Ottieni l'URL del plot
        url = fig.to_html(full_html=False)
        return url


def draw_scatter_plot(y_test, y_pred, r2, rmse, mode = None):
    plt.scatter(y_test, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs. Predicted Values')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    box = dict(boxstyle="round, pad=0.3", fc="white", ec="gray", lw=1)
    plt.text(plt.xlim()[1], plt.ylim()[0] + 0.02, f"R²: {r2:.2f}", ha='right', va='bottom', wrap=True, bbox=box)
    plt.text(plt.xlim()[1], plt.ylim()[0] * 0.85 + 0.02, f"RMSE: {rmse:.3f}", ha='right', va='bottom', wrap=True,
             bbox=box)

    if mode is None:
        plt.show()
    else:
        # Salva il plot in un oggetto BytesIO
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)

        # Pulisci la figura per evitare sovrapposizioni in futuri plot
        plt.clf()
        return img_buf


def draw_frequency_plot(y_test, y_pred, mode = None):

    # Daily returns plot y_pred x y_test
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_test, mode='lines', name='True Values'))
    fig.add_trace(go.Scatter(x=np.arange(len(y_test)), y=y_pred, mode='lines', name='Predicted Values'))
    fig.update_layout(title='True vs. Predicted Values', xaxis_title='Index', yaxis_title='Values')
    if mode is None:
        fig.show()
    else:
        # Ottieni l'URL del plot
        url = fig.to_html(full_html=False)
        return url


def draw_feature_importance_plot(model, X_test, y_test, mode = None):

    result = permutation_importance(model, X_test, y_test, n_repeats=10,
                                    random_state=42)  # Computing feature importance

    # Computing mean scores and obtaining features' names
    importances = result.importances_mean
    feature_names = X_test.columns

    # Sorting Features importances and names
    indices = importances.argsort()[::1]
    sorted_features = feature_names[indices]
    sorted_importances = importances[indices]

    # Plotting Feature Importance plot
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.barh(sorted_features, sorted_importances)
    ax.set_yticklabels(sorted_features)
    ax.set_ylabel('Features')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    if mode is None:
        plt.show()
    else:
        # Salva il plot in un oggetto BytesIO
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)

        # Pulisci la figura per evitare sovrapposizioni in futuri plot
        plt.clf()
        return img_buf


def draw_confusion_matrix(y_test_class, y_pred_class, mode = None):
    # Creiamo la matrice di confusione
    conf_matrix = confusion_matrix(y_test_class, y_pred_class)

    # Trasformiamo i valori in %
    conf_matrix = conf_matrix / np.sum(conf_matrix) * 100

    # PDisegniamo la matrice
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues',
                fmt='.2f', xticklabels=[-1, 1], yticklabels=[-1, 1])

    plt.title('Confusion Matrix')
    plt.xlabel('y_pred')
    plt.ylabel('y_test')
    if mode is None:
        plt.show()
    else:
        # Salva il plot in un oggetto BytesIO
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)

        # Pulisci la figura per evitare sovrapposizioni in futuri plot
        plt.clf()
        return img_buf


