import contextlib
import os
from typing import Iterator


# From https://stackoverflow.com/a/34333710/1165181
@contextlib.contextmanager
def set_env(**kwargs) -> Iterator[None]:
    """Temporarily set the environment variables."""
    old_environ = dict(os.environ)
    os.environ.update(kwargs)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)
