from bottle import Bottle, run, template, response, static_file
import sys
from io import StringIO

from project import load_model, run_model, make_planning
import plots
from data_handler import shared_values as sv

app = Bottle()


@app.route('/')
def index():
    # http://localhost:8080/
    with open("web/index.html", "r") as file:
        html_home = file.read()
    return html_home


def cleanup():
    sys.stdout = sys.__stdout__


# Definisci una route per gestire i file statici nella cartella "web"
@app.route('/web/<filepath:path>')
def server_static(filepath):
    return static_file(filepath, root='web')


@app.route('/draw_candlestick')
def draw_candlestick_plot():
    # http://localhost:8080/draw_candlestick
    return plots.draw_candlestick_plot(sv['brk'], 'c')


@app.route('/draw_scatter_plot')
def draw_scatter_plot():
    # http://localhost:8080/draw_scatter_plot
    img_buf = plots.draw_scatter_plot(sv['y_test'], sv['y_pred'], sv['r2'], sv['rmse'], 'c')
    # Imposta l'header della risposta HTTP con il giusto tipo di contenuto
    response.set_header('Content-Type', 'image/png')
    return img_buf.getvalue()


@app.route('/draw_frequency_plot')
def draw_frequency_plot():
    # http://localhost:8080/draw_frequency_plot
    return plots.draw_frequency_plot(sv['y_test'], sv['y_pred'], 'c')


@app.route('/draw_feature_importance_plot')
def draw_feature_importante_plot():
    # http://localhost:8080/draw_feature_importance_plot
    img_buf = plots.draw_feature_importance_plot(sv['model'], sv['X_test'], sv['y_test'], 'c')
    # Imposta l'header della risposta HTTP con il giusto tipo di contenuto
    response.set_header('Content-Type', 'image/png')
    return img_buf.getvalue()


@app.route('/draw_confusion_matrix')
def draw_confusion_matrix():
    # http://localhost:8080/draw_confusion_matrix
    img_buf = plots.draw_confusion_matrix(sv['y_test_class'], sv['y_pred_class'], 'c')
    # Imposta l'header della risposta HTTP con il giusto tipo di contenuto
    response.set_header('Content-Type', 'image/png')
    return img_buf.getvalue()


@app.route('/make_planning/start=<start_day>&end=<end_day>')
def make_plannig(start_day, end_day):
    # http://localhost:8080/make_planning/start=2020-12-18&end=2021-01-19

    # Cattura l'output di print
    output_buffer = StringIO()
    sys.stdout = output_buffer

    time_range = (start_day, end_day)
    make_planning(sv['y_pred'], sv['y_test'], time_range)

    # Ottengo la stringa dell'output del planning
    # (cancellando l'output del caricamento del modello)
    output = output_buffer.getvalue()
    output_arr = output.split("\n")
    output = "\n".join(output_arr[4:])

    # Reimposta l'output buffer
    output_buffer.truncate(0)
    output_buffer.seek(0)

    with open("web/planning.html", "r") as file:
        html_home = file.read()

    return template(html_home, output=output)


@app.route('/draw_planning')
def draw_frequency_plot():
    # http://localhost:8080/draw_planning
    return plots.plot_strategy(sv['data_brk'], sv['planning_df'], 'C')


@app.route('/data_table')
def show_data_table():
    # http://localhost:8080/data_table
    # Converti il DataFrame in un formato HTML utilizzando DataTables
    table_html = sv['brk'].to_html(classes='table table-striped table-bordered',
                                   table_id='myTable', index=False, escape=False)

    # Apro il file HTML della tabella
    with open("web/data_table.html", "r") as file:
        html_file = file.read()

    html_file = html_file.replace('<span>placeholder</span>', table_html)
    return html_file


if __name__ == "__main__":

    # Esegui l'app Bottle
    try:
        model = load_model()
        run_model(model, display_plot = False)
        run(app, host='0.0.0.0', port=8080, debug=True)
    finally:
        cleanup()
