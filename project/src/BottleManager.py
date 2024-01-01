from bottle import Bottle, run, template
import sys
from io import StringIO

from Project import load_model, main
import Plots as plots
from DataManager import shared_values as sv
import DataManager as dm

app = Bottle()


@app.route('/')
def index():

    run_app()

    # Ottieni l'output e sostituisci i \n con <br>
    output = output_buffer.getvalue().replace('\n', '<br>')

    # Reimposta l'output buffer
    output_buffer.truncate(0)
    output_buffer.seek(0)

    return output


def cleanup():
    sys.stdout = sys.__stdout__


def run_app():
    model = load_model()
    main(model)

@app.route('/draw_candlestick')
def draw_candlestick_plot():
    return plots.draw_candlestick_plot(sv['brk'], 'c')


@app.route('/data_table')
def show_data_table():
    # Converti il DataFrame in un formato HTML utilizzando DataTables
    table_html = sv['brk'].to_html(classes='table table-striped table-bordered', index=False, escape=False, table_id='myTable')

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


if __name__ == "__main__":

    # Cattura l'output di print
    output_buffer = StringIO()
    sys.stdout = output_buffer

    # Esegui l'app Bottle
    try:
        run(app, host='0.0.0.0', port=8080, debug=True)
    finally:
        cleanup()
