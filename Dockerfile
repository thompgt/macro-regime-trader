FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY . .

ENV MRT_DATA_CACHE_DIR=/app/data_cache
EXPOSE 8501

ENTRYPOINT ["mrt"]
CMD ["dashboard", "--host", "0.0.0.0", "--port", "8501"]
