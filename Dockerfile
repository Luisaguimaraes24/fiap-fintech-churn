FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY mlruns/ ./mlruns/

RUN mkdir -p logs

ENV MLFLOW_TRACKING_URI=file:///app/mlruns

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]