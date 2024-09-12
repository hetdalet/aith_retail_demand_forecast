FROM python:3.11.8-slim-bullseye

COPY ./ /app
WORKDIR /
ENV PYTHONPATH=/app
RUN pip install --no-cache-dir -r /app/requirements.txt
HEALTHCHECK \
    --interval=5s \
    --timeout=10s \
    --retries=5 \
    CMD if [[ $(curl -o /dev/null -s -w "%{http_code}" 0.0.0.0:${APP_PORT}/health/live) == '200' ]]; then echo 0; else echo 1; fi

CMD ["python", "/app/run.py"]
