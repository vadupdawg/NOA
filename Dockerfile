# Gebruik een officiële Python-runtime als basisimage
FROM python:3.11.4

# Zet omgevingsvariabelen
ENV PYTHONUNBUFFERED=1

# Stel de werkmap in de container in
WORKDIR /app

# Installeer git
RUN apt-get update && apt-get install -y git
RUN pip install --upgrade pip
RUN pip install gunicorn

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Kopieer de huidige mapinhoud naar de container
COPY . /app/

# Start de applicatie
CMD ["gunicorn", "-b", "0.0.0.0:8080", "--timeout", "3600", "app:app"]