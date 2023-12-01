# Gebruik een officiële Python-runtime als basisimage
FROM python:3.11.4

# Zet omgevingsvariabelen
ENV PYTHONUNBUFFERED=1

# Stel de werkmap in de container in
WORKDIR /app

RUN apt-get update \
    && apt-get install -y ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Kopieer alleen requirements.txt eerst om de Docker cache beter te benutten
COPY requirements_worker.txt /app/
RUN pip install --no-cache-dir -r requirements_worker.txt
RUN python -m spacy download nl_core_news_lg

# Kopieer de huidige mapinhoud naar de container
COPY . /app/

# Start de applicatie
CMD ["python", "worker_service.py"]
