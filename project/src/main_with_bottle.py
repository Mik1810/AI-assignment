# Importing Libraries

# Bottle to set up a python server
from bottle import route, run, template, response

# Financial Data Analysis
import yfinance as yf

# Data Handling
import pandas as pd
import numpy as np



import ta
import quantstats as qs

# Machine Learning Metrics
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix

# Models
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

# Feature Selection
from sklearn.feature_selection import SelectKBest, f_regression

# Feature Importance
from sklearn.inspection import permutation_importance

# Hyperparameter Tuning
import optuna

# Hiding warnings
import warnings
warnings.filterwarnings("ignore")

# DataFrame that will contain the data downloaded from yfinance
df = None
brk = None


@route("/test")
def test():
    return "test"


@route("/get_raws")
def get_assets():
    # Scarica i dati
    brk = yf.download('BRK-B', end='2023-05-13')

    # Costruisci la stringa HTML riga per riga
    html_rows = ''
    for _, row in brk.iterrows():
        html_row = '<tr>'
        for value in row:
            html_row += f'<td>{value}</td>'
        html_row += '</tr>'
        html_rows += html_row

    # Costruisci la stringa HTML completa
    html_string = f"""
        <table border="1" class="dataframe table table-striped">
          <thead>
            <tr style="text-align: right;">
              {"".join(f'<th>{col}</th>' for col in brk.columns)}
            </tr>
          </thead>
          <tbody>
            {html_rows}
          </tbody>
        </table>
        """

    # Imposta l'intestazione Content-Type su "text/html"
    response.headers['Content-Type'] = 'text/html'

    # Restituisci la risposta come HTML
    return html_string


@route("/get_json")
def get_json():
    # Converti il DataFrame in JSON
    json_data = df.to_json(orient='records')

    # Restituisci la risposta come HTML
    return {'data': json_data}


# Crea una route per visualizzare la tabella nel browser
@route('/data_table')
def show_data_table():
    # Converti il DataFrame in un formato HTML utilizzando DataTables
    table_html = df.to_html(classes='table table-striped table-bordered', index=False, escape=False, table_id='myTable')

    # Costruisci la pagina HTML direttamente in Python
    html_content = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Data Table</title>
        <!-- Includi le librerie DataTables -->
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.25/css/jquery.dataTables.css">
        <script type="text/javascript" charset="utf8" src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.10.25/js/jquery.dataTables.js"></script>
        <!-- Inizializza DataTables sulla tabella -->
        <script>
            $(document).ready( function () {{
                $('#myTable').DataTable();
            }} );
        </script>
    </head>
    <body>
        <h1>Data Table</h1>
        {table_html}
    </body>
    </html>
    '''

    return template(html_content)


def download_actions():
    global df, brk
    brk = yf.download('BRK-B', end='2023-05-13')

    # Converti i dati in un DataFrame
    df = pd.DataFrame(brk)

    # Aggiungi la colonna 'Date' al DataFrame
    df['Date'] = df.index

    # Riorganizza le colonne con 'Date' come prima colonna
    df = df[['Date'] + [col for col in df.columns if col != 'Date']]


@route("/plot_candlestick")
def plot_candlestick():
    global brk
    # Creating a Candlestick chart for Berkshire Hathaway stocks
    candlestick = go.Candlestick(x=brk.index,
                                 open=brk['Open'],
                                 high=brk['High'],
                                 low=brk['Low'],
                                 close=brk['Adj Close'],
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

    # Ottieni l'URL del plot
    url = fig.to_html(full_html=False)
    return url


if __name__ == "__main__":
    download_actions()
    run(host="0.0.0.0", debug=True, port=8080)

"""matplotlib
numpy
ta
quantstats
seaborn
matplotlib
scikit-learn
xgboost
lightgbm
catboost
optuna
IPython
"""
