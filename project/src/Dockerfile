FROM python:3.10-slim

LABEL authors="michaelpic"

WORKDIR /project

COPY main_with_bottle.py .
COPY requirements.txt .

# Aggiungo il pacchetto libgomp1 necessaria per un modello
RUN apt-get update && apt-get install -y libgomp1

# Aggiorno pip e installo le liberie necessarie
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8080
ENTRYPOINT ["python", "main.py"]
