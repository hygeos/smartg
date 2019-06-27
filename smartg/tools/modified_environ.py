import contextlib
import os

'''
A context to temporarily modify environment variables

https://stackoverflow.com/questions/2059482/python-temporarily-modify-the-current-processs-environment

>>> with modified_environ('HOME', LD_LIBRARY_PATH='/my/path/to/lib'):
...     home = os.environ.get('HOME')
...     path = os.environ.get("LD_LIBRARY_PATH")
>>> home is None
True
>>> path
'/my/path/to/lib'

>>> home = os.environ.get('HOME')
>>> path = os.environ.get("LD_LIBRARY_PATH")
>>> home is None
False
>>> path is None
True
'''

@contextlib.contextmanager
def modified_environ(*remove, **update):
    """
    Temporarily updates the ``os.environ`` dictionary in-place.

    The ``os.environ`` dictionary is updated in-place so that the modification
    is sure to work in all situations.

    :param remove: Environment variables to remove.
    :param update: Dictionary of environment variables and values to add/update.
    """
    env = os.environ
    update = update or {}
    remove = remove or []

    # List of environment variables being updated or removed.
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # Environment variables and values to restore on exit.
    update_after = {k: env[k] for k in stomped}
    # Environment variables and values to remove on exit.
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)
        [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(update_after)
        [env.pop(k) for k in remove_after]
