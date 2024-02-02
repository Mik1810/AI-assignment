# Se non ho già il modello preaddestrato ho  bisogno di queste librerie
try:
    from xgboost import XGBRegressor
except ImportError as e:
    print(f"Errore: {e}")
try:
    from lightgbm import LGBMRegressor
except ImportError as e:
    print(f"Errore: {e}")
try:
    from catboost import CatBoostRegressor
except ImportError as e:
    print(f"Errore: {e}")

# Per caricare un modello preaddestrato
import pickle

# Gestione dei dati
import pandas as pd
import numpy as np

# Financial Data Analysis
import yfinance as yf
import ta

# Machine Learning Metrics
from sklearn.metrics import r2_score, mean_squared_error

# Models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

# Feature Selection
from sklearn.feature_selection import SelectKBest, f_regression

# Hyperparameter Tuning
import optuna

# Importo il file per generare i plot e per salvare i dati
import plots
import data_handler as dh

# Nasconde i warnings
import warnings

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None


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


def return_strategy(_y_pred, time_range):
    start_day, end_day = time_range
    y_pred = _y_pred

    # Applico un treshold alla colonna dei valori predetti
    # Se il valore è positivo allora diventa 1, se è negativo o 0 diventa -1
    y_pred['pred'] = y_pred['pred'].apply(lambda x: 1 if x > 0 else -1)

    # A questo punto ritaglio il dataframe sul range di giorni selezionati
    y_pred = y_pred[start_day:end_day]

    # Creo un array che conterrà i valori della colonna pred
    binary_array = y_pred['pred'].values

    # Creo un valore artificiale per considerare anche l'ultimo elemento, che altrimenti verrebbe
    # tagliato fuori. tale valore artificiale sarà uguale all'ultimo valore reale
    binary_array = np.append(binary_array, binary_array[len(binary_array) - 1])

    # Creo una lista di coppie (valore, azione)
    result_pairs = []

    # Itero sull'array binario e aggiungo le coppie in base alle inversioni
    for i in range(len(binary_array) - 1):
        current_value = binary_array[i]
        next_value = binary_array[i + 1]

        if current_value == next_value:
            # Se il valore corrente è uguale al successivo, vuol dire che non c'è
            # un'inversione di trend
            result_pairs.append((current_value, "N"))
        elif current_value == 1:
            # Vuol dire che il next_value -1, quindi conviene vendere perchè
            # il giorno successivo il prezzo scenderà
            result_pairs.append((current_value, "V"))
        else:
            # current_value = -1 e next_value 1, quindi compro perchè vuol dire
            # che il giorno successivo il prezzo salirà
            result_pairs.append((current_value, "C"))

    # Creo un nuovo dataframe con i valori e le azioni come colonne
    result_df = pd.DataFrame(result_pairs, columns=['Value', "Action"])

    # Al nuovo dataframe aggiungo la data come indice
    result_df = result_df.set_index(y_pred.index)

    # Shifto la colonna delle azioni perchè deve essere fatta un giorno prima
    # result_df['Action'] = result_df['Action'].shift(1)

    return result_df


def compute_actions(result_df):
    # Estraggo l'intervallo di tempo selezionato
    start_day, end_day = str(result_df.index[0].date()), str(result_df.index[-1].date())

    # Estendi la data di fine di un giorno (l'ultimo giorno viene escluso)
    extended_end_day = pd.to_datetime(end_day) + pd.DateOffset(days=1)
    brk = yf.download('BRK-B', start=start_day, end=extended_end_day, progress=False)

    # 1. Partiamo con un'azione in possesso e i soldi in negativo per aver comprato l'azione
    # 2. Utilizziamo la variabile money come "portafoglio" per verificare alla fine quanto
    #    abbiamo guadagnato
    # 3. Supoponiamo di vendere e comprare sempre al prezzo di chiusura
    stocks = 1
    starting_value = brk.loc[start_day, 'Adj Close']
    money = -starting_value
    print(f"\nComprata azione il giorno {start_day} dal valore di {starting_value:.2f}")
    print(f"Portafoglio: {money:.2f}$, Azioni in possesso: {stocks}")

    # Creo un lista di coppie per passare poi i dati alla funzione di plot
    plot_list = [(start_day, starting_value, 'C')]

    # Scorr0 il DataFrame utilizzando iterrows
    for date, row in result_df.iterrows():
        action = row['Action']
        if action == "V":
            # Se l'azione precedentemente scelta è V di Vendi, allora cerco il prezzo di chiusura
            # relativo al giorno dell'azione, aggiungo i soldi della vendita al mio portafoglio
            price = brk.loc[str(date.date()), 'Adj Close']
            money += price
            stocks -= 1
            print(f"\nVenduta azione il giorno {str(date.date())} al prezzo di {price:.2f}")
            print(f"Portafoglio: {money:.2f}$, Azioni in possesso: {stocks}")

            plot_list.append((date, price, action))
        if action == "C":
            # Se l'azione precedentemente scelta è C di Compra, allora cerco il prezzo di chiusura
            # relativo al giorno dell'azione e rimuovo dal mio portafoglio i soldi per l'acquisto
            price = brk.loc[str(date.date()), 'Adj Close']
            money -= price
            stocks += 1
            print(f"\nComprata azione il giorno {str(date.date())} al prezzo di {price:.2f}")
            print(f"Portafoglio: {money:.2f}$, Azioni in possesso: {stocks}")

            plot_list.append((date, price, action))

    # Se sono rimaste azioni, converto il valore dell'azione nel rispettivo prezzo di chiusura aggiustato
    # relativo all'ultimo giorno
    while stocks > 0:
        price = brk.loc[end_day, 'Adj Close']
        money += price
        stocks -= 1
        print(f"\nVenduta azione il giorno {end_day} al prezzo di {price:.2f}")
        print(f"Portafoglio: {money:.2f}$, Azioni in possesso: {stocks}")

        plot_list.append((end_day, price, 'V'))

    print("\nResoconto:")
    print(f"Portafoglio: {money:.2f}$, la strategia ha prodotto {'del guadagno' if money > 0 else 'una perdita'}")

    # Converto la lista di prima in un DataFrame
    planning_df = pd.DataFrame(plot_list, columns=['Date', 'Value', 'Action'])
    planning_df['Date'] = pd.to_datetime(planning_df['Date'])
    planning_df.set_index('Date', inplace=True)

    dh.save_data('data_brk', brk[['Adj Close']])
    dh.save_data('planning_df', planning_df)

    return brk[['Adj Close']], planning_df


def make_planning(y_pred, y_test, time_range):
    # Estendo la colonna delle predizioni aggiungengo la data
    # Crea un dataframe combinato con la data e i valori predetti
    y_pred = pd.DataFrame({
        'date': y_test.index,
        'pred': y_pred,
    })

    # Imposto come indice del dataFrame la data
    y_pred = y_pred.set_index("date")
    # print(y_pred)

    result_df = return_strategy(y_pred, time_range)

    # Stampiamo il dataframe: ora abbiamo un insieme di azioni, vediamo quanto saremmo riusciti a guadagnare
    # avendo eseguito queste azioni
    return result_df


def run_model(_model, display_plot = True):
    # Scarica i valori delle azioni di Berkshire Hathaway Inc. (BRK-B) fino alla data odierna
    brk = yf.download('BRK-B', progress=False)
    dh.save_data('brk', brk)

    # Crea un grafico a candela delle azioni scaricate
    plots.draw_candlestick_plot(brk) if display_plot else None

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

    dh.save_data('X_train', X_train)
    dh.save_data('y_train', y_train)
    dh.save_data('X_test', X_test)
    dh.save_data('y_test', y_test)

    # A questo punto disponiamo di ben 39 features, molte delle quali rappresentano valori ridondanti o potrebbero
    # causare overfitting, pertanto scegliamo fra queste le migliori
    features = select_features(X_train, y_train, X_test)

    # Crea un nuovo DataSet utilizzando solo le features selezionate
    X_train_kbest = X_train[features]
    X_test_kbest = X_test[features]

    # A questo punto bisogna scegliere un modello adatto, testiamo vari modelli passando prima il DataSet aggiornato
    # con le feature selezionate e poi il DataSet con tutte le features

    model = _model
    # Se non esiste nessun modello preaddestrato
    if _model is None:
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
        model_now = GradientBoostingRegressor(random_state=42)

        # Il modello riceve in pasto i dati di allenamento
        model_now.fit(X_train, y_train)

        # Il modello calcola i valori predetti sulla base dei dati di testing
        y_pred = model_now.predict(X_test)

        # Vengono calcolati di nuovo la radice dell'errore quadratico medio e R²
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        # Disegna dei grafici di dispersione per vedere come sono distribuiti le predizioni rispetto ai valori reali
        plots.draw_scatter_plot(y_test, y_pred, r2, rmse) if display_plot else None
        plots.draw_frequency_plot(y_test, y_pred) if display_plot else None

        # Verifichiamo quanto sono state incisive le feature per i calcoli delle predizioni
        plots.draw_feature_importance_plot(model_now, X_test, y_test) if display_plot else None

        # Si può provare a migliorare il modello attaverso il tuning degli iperparametri
        # L'obbiettivo è minimizzare dei valori di errore
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=100,
                       show_progress_bar=True)

        # Stampo i valori dei parametri nel miglior trial
        print('Best parameters:', study.best_params)

        # Stampo il minimo valore di RMSE trovato
        print('Best score:', study.best_value)

        # Estrae i risultati dallo studio
        trials = study.trials

        # Scrive i risultati in un file di testo
        with open('resources/optuna_results.txt', 'w') as file:
            for trial in trials:
                file.write(f'Trial {trial.number}: Params - {trial.params}, Value - {trial.value}\n')

        # Ristanzio il modello con i parametri ottimizzati
        model_now = GradientBoostingRegressor(**study.best_params)
        model_now.fit(X_train, y_train)

        # Salvo il modello addestrato
        with open('model_now.pkl', 'wb') as file:
            pickle.dump(model_now, file)

        model = model_now
    # Da questo momento in poi il codice è condiviso,sia che venga addestrato che caricato

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    dh.save_data('model', model)
    dh.save_data('y_pred', y_pred)
    dh.save_data('r2', r2)
    dh.save_data('rmse', rmse)

    # A questo punto ridisegnamo i plot di dispersione
    plots.draw_scatter_plot(y_test, y_pred, r2, rmse) if display_plot else None
    plots.draw_frequency_plot(y_test, y_pred) if display_plot else None

    # Rivisualizziamo come è cambiata l'importanza dele varie feature dopo l'ottimizzazione
    plots.draw_feature_importance_plot(model, X_test, y_test) if display_plot else None

    # Salvo le predizioni e i valori di test in dei file
    # np.savetxt('resources/y_pred.csv', y_pred, fmt='%f')
    # y_test.to_csv('resources/y_test.csv', index=True)

    # Al fine di comprendere quanto le predizioni siano aderenti alla realtà,
    # creiamo due nuovi dataframe utilizzando come treshold lo 0. Tali array
    # presenteranno solo valori binari (1 quando il valore è maggiore di 0
    # e -1 quando è minore di 0)
    y_pred_class = np.where(y_pred > 0, 1, -1)
    y_test_class = np.where(y_test > 0, 1, -1)

    dh.save_data('y_pred_class', y_pred_class)
    dh.save_data('y_test_class', y_test_class)

    # Visualizziamo la matrice di confusione
    plots.draw_confusion_matrix(y_test_class, y_pred_class) if display_plot else None

    # A questo punto abbiamo i valori delle predizioni, sviluppiamo una strategia che li sfrutti
    # print(y_pred, y_test)

    # Scegliamo un range temporale di cui vogliamo sapere la strategia (formato 'YYYY-MM-DD')
    time_range = ("2018-07-23", "2018-07-30")
    time_range2 = ("2018-07-19", "2018-07-24")
    time_range3 = ("2020-12-18", "2021-01-19")
    time_range_fail = ("2020-03-04", "2020-03-20")

    # Ottengo i risultati dell'algoritmo
    result = make_planning(y_pred, y_test, time_range3)

    # Ottengo i prezzi e le azioni messe in atto
    prices, planning= compute_actions(result)

    # Creo il grafico della strategia
    plots.plot_strategy(prices, planning) if display_plot else print("OK")


def load_model():
    try:
        with open('models/model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        print("Modello caricato: ", loaded_model)
        return loaded_model
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    model = load_model()
    run_model(model)
