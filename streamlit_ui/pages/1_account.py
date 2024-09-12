from datetime import datetime
import streamlit as st
from streamlit_ui.backend import get_api


api = get_api()
user = st.session_state.user

st.header("Account")
user_id = user['id']
response = api.get(f"/users/{user_id}/account")
if response.status_code == 200:
    st.subheader("General info")
    account = response.json()
    st.write(f"Email: {user['email']}")
else:
    st.error(f"{response.status_code}: {response.text}")
    st.stop()

st.subheader("Link Telegram")
with st.form(key="telegram"):
    st.write(
        "To use the service through our Telegram bot, "
        "you need a Telegram account linked to your account. "
        "To link one, please, click \"Get Link\" and then follow it."
    )
    if st.form_submit_button("Get link"):
        response = api.post(f"/users/{user_id}/tg/link")
        if response.status_code != 200:
            st.error(response.json()['detail'])
        else:
            st.write(f"[Link a Telegram account]({response.json()['link']}).")

st.subheader("Balance")
st.write(f"Credits: {account['balance']}")
with st.form(key="account"):
    amount = st.number_input("Amount", value=100)
    if st.form_submit_button("Deposit"):
        response = api.post(
            f"/transactions/",
            data={"type": 1, "amount": amount},
        )
        st.rerun()

response = api.get(f"/users/{user['id']}/transactions")
data = [
    {
        "Time": datetime.fromisoformat(r["timestamp"]),
        "Type": r["type"],
        "Amount": r["amount"]
    }
    for r in response.json()
]
st.table(data)
