FROM python:3.13-slim

WORKDIR /app

RUN pip install uv --no-cache-dir

COPY pyproject.toml .
RUN uv pip install --system --no-cache -e .

COPY app/    ./app/
COPY static/ ./static/
COPY data/   ./data/

RUN mkdir -p artifacts data/chroma data/documents

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]