from datetime import datetime
import time
from uuid import UUID

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


st.header("Echo")
st.write("The state of art NLP model. It can convert any text to uppercase7")

response = api.get(f"/models/echo/tasks")
if response.status_code == 200:
    tasks = response.json()
else:
    st.error(f"Error: {response.json()}")
    st.stop()


messages = []
for task in tasks:
    messages.append({"role": "user", "content": task["input"]})
    messages.append({"role": "assistant", "content": task["output"]})

if "messages" not in st.session_state:
    st.session_state.messages = messages

for message in st.session_state.messages:
    role = message["role"]
    if role == "assistant":
        st.status(message["content"], state="complete", expanded=False)
    else:
        with st.chat_message(role):
            st.markdown(message["content"])

with st.chat_message("assistant"):
    st.markdown("Send a text message and bot will convert it to upper case.")

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.status("Processing...") as status:
        response = api.post(f"/models/echo/task", data={"input": prompt})
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
        st.session_state.messages.append({
            "role": "assistant",
            "content": output
        })
