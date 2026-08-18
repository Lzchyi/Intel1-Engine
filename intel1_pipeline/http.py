from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


USER_AGENT = "Intel1/0.1 (+https://github.com/Lzchyi/Intel1)"


class FetchError(RuntimeError):
    pass


def fetch_text(url: str, timeout: int = 15) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")
    except (HTTPError, URLError, TimeoutError) as error:
        raise FetchError(str(error)) from error


def fetch_json(url: str, timeout: int = 15) -> dict[str, Any]:
    return json.loads(fetch_text(url, timeout=timeout))
