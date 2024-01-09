FROM python:3.10-slim

LABEL authors="michaelpic"

RUN mkdir web
RUN mkdir resources

COPY project.py .
COPY data_handler.py .
COPY server.py .
COPY plots.py .
COPY requirements.txt .
COPY models/model.pkl ./models/
COPY web/ ./web/

# COPY frozen-requirements.txt .

# Aggiungo il pacchetto libgomp1 necessaria per un modello
# RUN apt-get update && apt-get install -y libgomp1

# Aggiorno pip e installo le liberie necessarie
RUN pip install --upgrade pip
RUN pip install --no-cache-dir   -r requirements.txt
# RUN pip install -r frozen-requirements.txt

EXPOSE 8080
ENTRYPOINT ["python", "server.py"]
