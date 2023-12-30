# Per caricare un modello preaddestrato
import pickle

# Gestione dei dati
import pandas as pd
import numpy as np

# Financial Data Analysis
import yfinance as yf
import ta
import quantstats as qs

# Machine Learning Metrics
from sklearn.metrics import r2_score, mean_squared_error

# Models
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

# Feature Selection
from sklearn.feature_selection import SelectKBest, f_regression

# Hyperparameter Tuning
import optuna

# Hiding warnings
import warnings
warnings.filterwarnings("ignore")

import plots


# Funzione che calcola ulteriori indici utili da quelli delle azioni
def feature_engineering(df):

    # high_low_ratio indica la volatilità misurata come il rapporto tra i prezzi più alti e più bassi.
    df['high_low_ratio'] = df['High'] / df['Low']

    # open_adjclose cattura la direzione complessiva del mercato confrontando i prezzi di apertura e chiusura.
    df['open_adjclose_ratio'] = df['Open'] / df['Adj Close']

    # candle_to_wick_ratio rappresenta la porzione dell'intervallo di prezzo coperto dal corpo della candela,
    # che rappresenta la distanza tra i prezzi di apertura e chiusura di ogni giorno.
    df['candle_to_wick_ratio'] = (df['Adj Close'] - df['Open']) / (df['High'] - df['Low'])
    df['candle_to_wick_ratio'] = df['candle_to_wick_ratio'].replace([np.inf, -np.inf], 0)

    # I seguenti indici rappresentano il prezzo di chiusura spostato idietro nel tempo di uno, due, tre e cinque giorni.
    df['Close_lag1'] = df['Adj Close'].shift(1)
    df['Close_lag2'] = df['Adj Close'].shift(2)
    df['Close_lag3'] = df['Adj Close'].shift(3)
    df['Close_lag5'] = df['Adj Close'].shift(5)

    # I sequenti valori sono utili per catturare momentum e trend su un determinato arco temporale.
    # Ad esempio, un rapporto superiore a 1 suggerirebbe un momentum rialzista.
    df['Close_lag1_ratio'] = df['Adj Close'] / df['Close_lag1']
    df['Close_lag2_ratio'] = df['Adj Close'] / df['Close_lag2']
    df['Close_lag3_ratio'] = df['Adj Close'] / df['Close_lag3']
    df['Close_lag5_ratio'] = df['Adj Close'] / df['Close_lag5']

    # Indici che rappresentano la media mobile semplice calcota con finestre di grandezza 10, 20, 80 e 100
    df['sma10'] = ta.trend.sma_indicator(df['Adj Close'], window=10)
    df['sma20'] = ta.trend.sma_indicator(df['Adj Close'], window=20)
    df['sma80'] = ta.trend.sma_indicator(df['Adj Close'], window=80)
    df['sma100'] = ta.trend.sma_indicator(df['Adj Close'], window=100)

    # Rappresentano i rapporti tra il prezzo di chiusura e ciascuna media mobile, indicando se il prezzo è al di sopra
    # o al di sotto della media.
    df['Close_sma10_ratio'] = df['Adj Close'] / df['sma10']
    df['Close_sma20_ratio'] = df['Adj Close'] / df['sma20']
    df['Close_sma80_ratio'] = df['Adj Close'] / df['sma80']
    df['Close_sma100_ratio'] = df['Adj Close'] / df['sma100']

    # Ulteriori indici che rappresentano i rapporti tra le varie medie mobili
    df['sma10_sma20_ratio'] = df['sma10'] / df['sma20']
    df['sma20_sma80_ratio'] = df['sma20'] / df['sma80']
    df['sma80_sma100_ratio'] = df['sma80'] / df['sma100']
    df['sma10_sma80_ratio'] = df['sma10'] / df['sma80']
    df['sma20_sma100_ratio'] = df['sma20'] / df['sma100']

    # Indicatori tecnici del mondo finanziario (analizzati all'interno del documento)
    df['rsi'] = ta.momentum.RSIIndicator(df['Adj Close']).rsi()
    df['rsi_overbought'] = (df['rsi'] >= 70).astype(int)
    df['rsi_oversold'] = (df['rsi'] <= 30).astype(int)
    df['cci'] = ta.trend.cci(df['High'], df['Low'], df['Adj Close'], window=20, constant=0.015)
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['Adj Close'], volume=df['Volume']).on_balance_volume()
    df['obv_divergence_10_days'] = df['obv'].diff().rolling(10).sum() - df['Adj Close'].diff().rolling(10).sum()
    df['obv_divergence_20_days'] = df['obv'].diff().rolling(20).sum() - df['Adj Close'].diff().rolling(20).sum()

    # Rapppresenta i rendimenti giornalieri percentuali: calcola la variazione percentuale tra gli elementi successivi
    # della colonna 'Adj Close'. La formula è (valore_corrente - valore_precedente) / valore_precedente.
    df['returns_in_%'] = np.round((df['Adj Close'].pct_change()) * 100, 2)

    # Variabile target (y) che rappresenta il ritorno in % ma spostato in avanti di un giorno al fine di indicare
    # al modello a cosa puntare.
    df['target'] = df['returns_in_%'].shift(-1)

    # Rimuovi i valori null dal DataSet
    df.dropna(inplace=True)

    return df

# Funzione che consente di selezionare le migliori features per addestrare il modello
def select_features(X_train, y_train, X_test):
    # Crea un selettore per le feature migliori utilizzando il test F
    k_best = SelectKBest(score_func=f_regression, k=len(X_train.columns))

    # Addestro (e trasformo) il selettettore sui dati di input
    X_train_kbest = k_best.fit_transform(X_train, y_train)
    X_test_kbest = k_best.transform(X_test)

    # Prende gli indici e i nomi delle features
    feature_indices = k_best.get_support(indices=True)
    feature_names = X_train.columns[feature_indices]

    # Salva i valori p, i quali corrispondono da una feature specifica e rappresenta la probabilità di osservare
    # la statistica del test F osservata, o una statistica ancora più estrema, supponendo che l'ipotesi nulla sia vera.
    p_values = k_best.pvalues_

    # Creating features list
    features = []

    # Seleziona solo le features ceh hanno un valore p minore di 0.2
    for feature, pvalue in zip(feature_names, p_values):
        if pvalue < 0.2:
            features.append(feature)

    # In sintesi, il codice utilizza il test F per valutare la significatività delle feature rispetto alla variabile
    # di output e seleziona solo quelle con valori p al di sotto di una determinata soglia. Questo processo aiuta a
    # identificare le feature più rilevanti per il modello, contribuendo a semplificare e migliorare la precisione
    # del modello stesso.

    return features


# Funzione che testa vari modelli di regressione lineare cercandone uno che minimizzi la radice dell'errore
# quadratico medio e che massimizzi il valore R²
def test_models(X_train, y_train, X_test, y_test):

    # random_state = 42 è un seed che convenzionalmente viene scelto al fine di riprodurre gli stessi risultati
    # in caso di debug
    regressors = [
        LinearRegression(),
        Ridge(random_state=42),
        ExtraTreesRegressor(random_state=42),
        GradientBoostingRegressor(random_state=42),
        KNeighborsRegressor(),
        XGBRegressor(random_state=42),
        LGBMRegressor(random_state=42, verbose=-1),
        CatBoostRegressor(random_state=42, verbose=False),
        AdaBoostRegressor(random_state=42),
    ]

    r2_map, rmse_map = {}, {}

    # Iterating over algorithms and printing scores
    for reg in regressors:
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        # print(f'{type(reg).__name__}: R² = {r2:.2f}, Root Mean Squared Error = {rmse:.2f}')
        r2_map[type(reg).__name__] = float(f'{r2:.2f}')
        rmse_map[type(reg).__name__] = float(f'{rmse:.2f}')

    ordered_r2_map = dict(sorted(r2_map.items(), key=lambda item: item[1], reverse=True))
    ordered_rmse_map = dict(sorted(rmse_map.items(), key=lambda item: item[1]))
    return ordered_r2_map, ordered_rmse_map


# Funzione obbiettivo per il modello
def objective(trial, X_train, y_train, X_test, y_test):

    # Definizione di diversi parametri con cui verrà testato il modello
    params = {
        'loss': trial.suggest_categorical('loss', ['squared_error', 'absolute_error', 'huber', 'quantile']),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 20),
        'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
        'random_state': 42
    }

    # Fitting and predicting
    tuning = GradientBoostingRegressor(**params)
    tuning.fit(X_train, y_train)
    preds = tuning.predict(X_test)

    # Computing RMSE score
    rmse = np.round(mean_squared_error(y_test, preds, squared=False), 3)
    return rmse  # Returining the score


def main(model):

    # Scarica i valori delle azioni di Berkshire Hathaway Inc. (BRK-B) fino alla data odierna
    brk = yf.download('BRK-B')

    # Crea un grafico a candela delle azioni scaricate
    # plots.draw_candlestick_plot(brk)

    # Divide il DataSet in train e test
    train = brk[brk.index.year <= 2016]
    test = brk[brk.index.year >= 2017]

    # Il DataSet ha bisogno di essere arricchito con ulteriori features in modo
    # da migliorare il potere predittivo del modello
    train = feature_engineering(train)
    test = feature_engineering(test)

    # Divide le variabili indipendenti (X) da quelle dipendendenti (y)
    # axis = 1 indica che stiamo eliminando una colonna (1 per le colonne, 0 per le righe)
    X_train = train.drop('target', axis=1)
    y_train = train.target

    X_test = test.drop('target', axis=1)
    y_test = test.target

    # A questo punto disponiamo di ben 39 features, molte delle quali rappresentano valori ridondanti o potrebbero
    # causare overfitting, pertanto scegliamo fra queste le migliori
    features = select_features(X_train, y_train, X_test)

    # Crea un nuovo DataSet utilizzando solo le features selezionate
    X_train_kbest = X_train[features]
    X_test_kbest = X_test[features]

    # A questo punto bisogna scegliere un modello adatto, testiamo vari modelli passando prima il DataSet aggiornato
    # con le feature selezionate e poi il DataSet con tutte le features

    # Se non esiste nessun modello preaddestrato
    if model is None:
        r2_mapk, rmse_mapk = test_models(X_train_kbest, y_train, X_test_kbest, y_test)
        r2_map, rmse_map = test_models(X_train, y_train, X_test, y_test)

        print("\nTest effettuato con il DataSet con feature selezionate: ")
        print("R²: ", r2_mapk)
        print("rmse: ", rmse_mapk)
        print("\nTest effettuato sul DataSet con tutte le features: ")
        print("R²: ", r2_map)
        print("rmse: ", rmse_map)

        # Dal primo test si nota che il LinearRegression, il Ridge e il GradientBoostingRegressor hanno gli stessi valori.
        # Dal secondo test il GradientBoostingRegresso performa meglio di tutti gli altri modelli, pertanto sarà questo il
        # modello scelto.

        # Istanzio il modello di regressione
        model = GradientBoostingRegressor(random_state=42)

        # Il modello riceve in pasto i dati di allenamento
        model.fit(X_train, y_train)

        # Il modello calcola i valori predetti sulla base dei dati di testing
        y_pred = model.predict(X_test)

        # Vengono calcolati di nuovo la radice dell'errore quadratico medio e R²
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        # Disegna dei grafici di dispersione per vedere come sono distribuiti le predizioni rispetto ai valori reali
        #plots.draw_scatter_plot(y_test, y_pred, r2, rmse)
        #plots.draw_scatter_plot2(y_test, y_pred)

        # Verifichiamo quanto sono state incisive le feature per i calcoli delle predizioni
        #plots.draw_feature_importance_plot(model, X_test, y_test)

        # Si può provare a migliorare il modello attaverso il tuning degli iperparametri
        # L'obbiettivo è minimizzare dei valori di errore
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=100, show_progress_bar=True)

        # Estrai i risultati dallo studio
        trials = study.trials

        # Scrivi i risultati in un file di testo
        with open('optuna_results.txt', 'w') as file:
            for trial in trials:
                file.write(f'Trial {trial.number}: Params - {trial.params}, Value - {trial.value}\n')

        # Salvo il modello addestrato
        with open('model3.pkl', 'wb') as file:
            pickle.dump(model, file)

    else:
        # Se sono qui vuol dire che ho già il modello preaddestrato

        y_pred = model.predict(X_test)
        print("y_pred: ", y_pred)


if __name__ == "__main__":

    try:
        with open('modelNO.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
            print(loaded_model)
        main(loaded_model)
    except FileNotFoundError:
        main(None)

    # Salva il modello utilizzando pickle
    # with open('model.pkl', 'wb') as file:
    #    pickle.dump(model, file)



