import os
import urllib.parse as urlparse
from urllib.parse import unquote

import httpx
import streamlit as st


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
        response = httpx.get(
            self._get_url(route, params),
            cookies=cookie,
            timeout=10.0
        )
        return response

    def post(self, route, data=None, cookie=None):
        data = data or {}
        cookie = cookie or self._cookies
        resource = httpx.post(
            self._get_url(route),
            json=data,
            cookies=cookie,
            timeout=10.0
        )
        return resource

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


def login_msg():
    st.error(f"[Log in]({LOGIN_URL}) please.")
    st.stop()


def get_api():
    api = API(
        host=os.getenv("APP_HOST"),
        port=os.getenv("APP_PORT"),
    )
    return api
