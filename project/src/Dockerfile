FROM python:3.10-slim

LABEL authors="michaelpic"

# WORKDIR /project

COPY Project.py .
COPY DataManager.py .
COPY BottleManager.py .
COPY requirements.txt .
COPY models/model.pkl ./models/
COPY Plots.py .

# COPY frozen-requirements.txt .

RUN mkdir resources

# Aggiungo il pacchetto libgomp1 necessaria per un modello
# RUN apt-get update && apt-get install -y libgomp1

# Aggiorno pip e installo le liberie necessarie
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# RUN pip install -r frozen-requirements.txt

EXPOSE 8080
ENTRYPOINT ["python", "BottleManager.py"]
