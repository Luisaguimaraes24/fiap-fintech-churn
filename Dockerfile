FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

RUN mkdir -p logs

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Como usar ---
# docker build -t fiap-churn-api .
# docker run -p 8000:8000 fiap-churn-api
#
# Teste rápido:
# curl -X POST http://localhost:8000/predict \
#   -H "Content-Type: application/json" \
#   -d '{"idade":35,"renda":5000,"satisfacao":2,"historico_meses":24,"num_produtos":1}'