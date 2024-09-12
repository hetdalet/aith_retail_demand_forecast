FROM python:3.11.8-slim-bullseye

COPY ./ /streamlit_ui
WORKDIR /streamlit_ui
ENV PYTHONPATH=/
RUN pip install --no-cache-dir -r requirements.txt

CMD streamlit run --server.headless true main.py
