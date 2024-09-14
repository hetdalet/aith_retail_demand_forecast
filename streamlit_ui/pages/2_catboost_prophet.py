import itertools
import json
import time
from datetime import datetime
from uuid import UUID

import pandas as pd
import streamlit as st
from streamlit_ui.backend import get_api


api = get_api()


def poll_task(key: UUID):
    while True:
        time.sleep(1)
        response = api.get(f"/tasks/{key}")
        status = response.status_code
        if status != 200 or response.json()["end"] is not None:
            break
    return response


def gen_features(df, window_size=3, limit=None):
    limit = limit or len(df)
    items = list(df['item_id'].unique())
    stores = list(df['store_id'].unique())
    count = 0
    for item_id, store_id in itertools.product(items, stores):
        features_df = (
            df
            .query('item_id == @item_id and store_id == @store_id')
            .sort_values(by='date', ascending=True)
        )[:window_size]
        if features_df.empty:
            continue
        count += 1
        if count > limit:
            break

        features_json = [{'item_id': item_id[8:], 'store_id': store_id}]
        cashback_col = f'CASHBACK_{store_id}'
        for _, row in features_df.iterrows():
            features_json.append({
                'ds': row['date'],
                'sell_price': str(row['sell_price']),
                'cashback': str(row[cashback_col]),
            })
        yield features_json


st.title("Demand forecasting model")

uploaded_sales = st.file_uploader(
    "shop_sales.csv",
    type='csv',
    accept_multiple_files=False
)
uploaded_sales_dates = st.file_uploader(
    "shop_sales_dates.csv",
    type='csv',
    accept_multiple_files=False
)
uploaded_sales_prices = st.file_uploader(
    "shop_sales_prices.csv",
    type='csv',
    accept_multiple_files=False
)
if (uploaded_sales is not None and
        uploaded_sales_dates is not None and
        uploaded_sales_prices is not None):
    sales = pd.read_csv(uploaded_sales)
    sales_dates = pd.read_csv(uploaded_sales_dates)
    sales_prices = pd.read_csv(uploaded_sales_prices)
    window_size = st.number_input("Forecast window size", value=3)
    if st.button("Forecast"):
        with st.spinner('Loading data...'):
            merged = (
                sales
                .merge(sales_dates, on='date_id', how='left')
                .merge(sales_prices, on=['item_id', 'store_id', 'wm_yr_wk'], how='left')
                .loc[:, ['item_id', 'store_id', 'date', 'sell_price', 'CASHBACK_STORE_1', 'CASHBACK_STORE_2', 'CASHBACK_STORE_3']]
            )
        # data_size = len(merged)
        data_size = 10
        forecasts = []
        progress_bar = st.progress(0, text="Forecasting")
        count = 0
        for features in gen_features(merged, window_size=window_size, limit=data_size):
            data = json.dumps(features)
            response = api.post(
                "/models/catboost_prophet/task",
                data={"input": data}
            )
            if response.status_code != 200:
                msg = response.json()["detail"]
                st.error(msg)
                st.stop()

            task = response.json()
            response = poll_task(task["key"])
            if response.status_code != 200:
                st.error(response.json())
                st.stop()

            task = response.json()
            output = task["output"]
            output = json.loads(output)

            forecast = features[0].copy()
            dates = [item['ds'] for item in features[1:]]
            forecast.update(dict(zip(dates, output)))
            forecasts.append(forecast)

            count += 1
            completed = count / data_size
            progress_bar.progress(
                completed,
                text=f"Completed: {100 * completed:.2f}%"
            )
        forecast = pd.DataFrame(forecasts)
        st.write(forecast)

