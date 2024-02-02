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


def draw_candlestick_plot(stock, mode=None):
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


def draw_scatter_plot(y_test, y_pred, r2, rmse, mode=None):
    plt.figure(figsize=(6.40, 4.80))
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


def draw_frequency_plot(y_test, y_pred, mode=None):
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


def draw_feature_importance_plot(model, X_test, y_test, mode=None):
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
        plt.close(fig)
    else:
        # Salva il plot in un oggetto BytesIO
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)

        # Pulisci la figura per evitare sovrapposizioni in futuri plot
        plt.clf()
        return img_buf


def draw_confusion_matrix(y_test_class, y_pred_class, mode=None):
    # Creiamo la matrice di confusione
    conf_matrix = confusion_matrix(y_test_class, y_pred_class)

    # Trasformiamo i valori in %
    conf_matrix = conf_matrix / np.sum(conf_matrix) * 100

    # Disegniamo la matrice
    plt.figure(num=120, figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues',
                fmt='.2f', xticklabels=[-1, 1], yticklabels=[-1, 1])

    plt.title('Confusion Matrix')
    plt.xlabel('y_pred')
    plt.ylabel('y_test')
    if mode is None:
        plt.show()
        plt.clf()
    else:
        # Salva il plot in un oggetto BytesIO
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)

        # Pulisci la figura per evitare sovrapposizioni in futuri plot
        plt.clf()
        return img_buf


# param: data table -> columuns: 'Date' (index), 'Adj Close'
#        planning_df -> columns: 'Date' (index), 'Value', 'Action'
def plot_strategy(data_brk, planning_df, mode = None):
    # Calcolo del momentum
    data_brk['momentum'] = data_brk['Adj Close'].pct_change()

    # Trovare i punti di inversione del trend
    buy_signals = data_brk[(data_brk['momentum'] < 0) & (data_brk['momentum'].shift(-1) > 0)]
    sell_signals = data_brk[(data_brk['momentum'] > 0) & (data_brk['momentum'].shift(-1) < 0)]

    # Unire i giorni in cui ci sono segnali di acquisto o vendita
    marked_days = buy_signals.index.union(sell_signals.index).union(planning_df.index)

    # Creare il grafico
    figure = go.Figure()

    # Aggiungere il grafico dei prezzi di chiusura
    figure.add_trace(go.Scatter(x=data_brk.index,
                                y=data_brk['Adj Close'],
                                name='Adj Close Price',
                                mode='lines'))

    # Aggiungere il grafico dei prezzi di chiusura
    figure.add_trace(go.Scatter(x=planning_df.index,
                                y=planning_df['Value'],
                                name='Plan'))

    # Aggiungere i triangolini per i segnali di acquisto
    figure.add_trace(go.Scatter(x=buy_signals.index,
                                y=buy_signals['Adj Close'],
                                mode='markers', name='Salita',
                                marker=dict(color='green', symbol='triangle-up', size=13)))

    # Aggiungere i triangolini per i segnali di vendita
    figure.add_trace(go.Scatter(x=sell_signals.index,
                                y=sell_signals['Adj Close'],
                                mode='markers', name='Discesa',
                                marker=dict(color='red', symbol='triangle-down', size=13)))

    # Aggiungere annotazioni per gli indicatori del tempo
    for index, row in planning_df.iterrows():
        if row['Action'] == "C":
            figure.add_trace(go.Scatter(x=[index],
                                        y=[row['Value']],
                                        mode='text',
                                        text=['Compra'],
                                        textposition="top center",
                                        textfont=dict(size=14, color='black', family='Arial'),
                                        showlegend=False))
        else:
            figure.add_trace(go.Scatter(x=[index],
                                        y=[row['Value']],
                                        mode='text',
                                        text=['Vendi'],
                                        textposition="top center",
                                        textfont=dict(size=14, color='black', family='Arial'),
                                        showlegend=False))

    # Aggiungere l'asse x con i giorni marcanti
    figure.update_xaxes(
        tickvals=marked_days,
        ticktext=marked_days.strftime('%d %b %Y'),
        tickmode='array'
    )

    # Configurare il layout del grafico
    figure.update_layout(title='Grafico di investimento',
                         xaxis_title='Data',
                         yaxis_title='Prezzo')
    figure.update_yaxes(title="Prezzo")

    if mode is None:
        figure.show()
    else:
        # Ottieni l'URL del plot
        url = figure.to_html(full_html=False)
        return url