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


@app.route('/draw_wave_plot')
def draw_wave_plot():
    # http://localhost:8080/draw_scatter_plot2
    return plots.draw_wave_plot(sv['y_test'], sv['y_pred'], 'c')


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
    time_range = (start_day, end_day)
    make_planning(sv['y_pred'], sv['y_test'], time_range)

    # Ottieni l'output e sostituisci i \n con <br>
    output = output_buffer.getvalue().replace('\n', '<br>')

    # Reimposta l'output buffer
    output_buffer.truncate(0)
    output_buffer.seek(0)

    return output


@app.route('/data_table')
def show_data_table():
    # http://localhost:8080/data_table
    # Converti il DataFrame in un formato HTML utilizzando DataTables
    table_html = sv['brk'].to_html(classes='table table-striped table-bordered', index=False, escape=False,
                                   table_id='myTable')

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
        {table_html}
    </body>
    </html>
    '''

    return template(html_content)


if __name__ == "__main__":

    # Cattura l'output di print
    output_buffer = StringIO()
    sys.stdout = output_buffer

    # Esegui l'app Bottle
    try:
        model = load_model()
        run_model(model)
        run(app, host='0.0.0.0', port=8080, debug=True)
    finally:
        cleanup()
