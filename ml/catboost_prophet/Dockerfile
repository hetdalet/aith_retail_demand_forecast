FROM python:3.11.8-slim-bullseye

COPY ./ /model
WORKDIR /model
ENV PYTHONPATH=/
RUN pip install --no-cache-dir -r requirements.txt

CMD python -u worker.py
