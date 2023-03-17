from pathlib import Path

from cached_path import cached_path
from requests import HTTPError

def cached_path_that_ignores_405(*args, retries: int = 5, **kwargs) -> Path:
    for _ in range(retries):
        try:
            return cached_path(*args, **kwargs)
        except HTTPError as e:
            # Dropbox many times returns 405 Method Not Allowed when `cached_path` calls `HEAD` on a URL.
            # This is a bug in Dropbox, but we can work around it by retrying.
            if e.response.status_code == 405:
                print("Got 405 Method Not Allowed for `cached_path`, retrying…")
            else:
                raise e
