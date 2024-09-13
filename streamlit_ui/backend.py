import os
import urllib.parse as urlparse
from urllib.parse import unquote

import httpx
import streamlit as st
# https://github.com/streamlit/streamlit/pull/5457
# from streamlit.web.server.websocket_headers import _get_websocket_headers


COOKIE_NAME = os.getenv("COOKIE_NAME")
LOGIN_URL = f"http://localhost:8080/auth/login"


class API:

    def __init__(self, host, port, cookies=None):
        self._host = host
        self._port = port
        self._cookies = cookies or {}
        self._api_authority = f"http://{host}:{port}"

    def get(self, route, params=None, cookie=None):
        cookie = cookie or self._cookies
        return httpx.get(self._get_url(route, params), cookies=cookie)

    def post(self, route, data=None, cookie=None):
        data = data or {}
        cookie = cookie or self._cookies
        return httpx.post(self._get_url(route), json=data, cookies=cookie)

    def _get_url(self, route, params=None):
        params = [
            f"{name}={val}"
            for name, val in (params or {}).items()
        ]
        if params:
            params = "&".join(params)
            route = f"{route}?{params}"
        url = urlparse.urljoin(self._api_authority, route)
        return url


# def get_all_cookies():
#     """
#     WARNING: This uses unsupported feature of Streamlit
#     Returns the cookies as a dictionary of kv pairs
#
#     Source: https://discuss.streamlit.io/t/streamlit-cookies-which-package-to-use-so-many-of-them/50500/3
#     """
#
#     headers = _get_websocket_headers()
#     if headers is None:
#         return {}
#
#     if "Cookie" not in headers:
#         return {}
#
#     cookie_string = headers["Cookie"]
#     # A sample cookie string: "K1=V1; K2=V2; K3=V3"
#     items = cookie_string.split(";")
#
#     cookie_dict = {}
#     for item in items:
#         key, val = item.split("=")
#         key = key.strip()
#         val = val.strip()
#         #e.g. Convert name%40company.com to name@company.com
#         cookie_dict[key] = unquote(val)
#
#     return cookie_dict


def login_msg():
    st.error(f"[Log in]({LOGIN_URL}) please.")
    st.stop()


def get_api():
    # cookies = get_all_cookies()
    # check_login(cookies)
    api = API(
        host=os.getenv("APP_HOST"),
        port=os.getenv("APP_PORT"),
        # cookies={COOKIE_NAME: cookies[COOKIE_NAME]}
    )
    return api
