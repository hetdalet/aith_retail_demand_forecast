import json
from datetime import datetime
import time
from uuid import UUID

import streamlit as st
from streamlit_ui.backend import get_api


api = get_api()
user = st.session_state.user


def poll_task(key: UUID):
    while True:
        time.sleep(1)
        response = api.get(f"/tasks/{key}")
        status = response.status_code
        if status != 200 or response.json()["end"] is not None:
            break
    return response


st.title("Wine Quality Prediction")
st.header("Wine params")

fixed_acidity = st.number_input(
    "Fixed Acidity",
    min_value=0.0,
    value=7.4,
    step=0.1
)
volatile_acidity = st.number_input(
    "Volatile Acidity",
    min_value=0.0,
    value=0.7,
    step=0.01
)
citric_acid = st.number_input(
    "Citric Acid",
    min_value=0.0,
    value=0.0,
    step=0.01
)
residual_sugar = st.number_input(
    "Residual Sugar",
    min_value=0.0,
    value=1.9,
    step=0.1
)
chlorides = st.number_input(
    "Chlorides",
    min_value=0.0,
    value=0.076,
    step=0.001
)
free_sulfur_dioxide = st.number_input(
    "Free Sulfur Dioxide",
    min_value=0,
    value=11,
    step=1
)
total_sulfur_dioxide = st.number_input(
    "Total Sulfur Dioxide",
    min_value=0,
    value=34,
    step=1
)
density = st.number_input(
    "Density",
    min_value=0.0,
    value=0.9978,
    step=0.0001
)
pH = st.number_input(
    "pH",
    min_value=0.0,
    value=3.51,
    step=0.01
)
sulphates = st.number_input(
    "Sulphates",
    min_value=0.0,
    value=0.56,
    step=0.01
)
alcohol = st.number_input(
    "Alcohol",
    min_value=0.0,
    value=9.4,
    step=0.1
)

if st.button("Predict Quality"):
    wine_params = {
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }
    wine_params = json.dumps(wine_params)
    with st.status("Processing...") as status:
        response = api.post(
            f"/services/wineq/task",
            data={"input": wine_params}
        )
        if response.status_code != 200:
            msg = response.json()["detail"]
            status.update(label=msg, state="error", expanded=False)
            st.stop()
        task = response.json()
        response = poll_task(task["key"])
        if response.status_code != 200:
            st.error(response.json())
            st.stop()
        task = response.json()
        output = task["output"]
        status.update(label=output, state="complete", expanded=False)
