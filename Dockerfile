FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app/code ./code
COPY ./models ./models
COPY ./data ./data


CMD ["python","code/app.py"]
