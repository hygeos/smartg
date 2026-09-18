"""Context manager for temporarily modifying the process environment.

This module provides a context manager that allows environment variables
to be added, updated, or removed for the duration of a ``with`` block.
On exit, the original environment is restored: variables that existed
before are reset to their previous values, variables that were newly
introduced are removed, and variables that were explicitly removed are
restored.

It is used, for instance, to temporarily set the ``CUDA_DEVICE``
environment variable when initializing a pycuda context (see
:func:`smartg.smartg.Smartg.__init__`).

References
----------
.. [1] Stack Overflow: temporarily modify the current process's
       environment.

Examples
--------
>>> with modified_environ('HOME', SMARTG_EXAMPLE='/my/path/to/lib'):
...     home = os.environ.get('HOME')
...     path = os.environ.get('SMARTG_EXAMPLE')
>>> home is None
True
>>> path
'/my/path/to/lib'

>>> os.environ.get('HOME') is None
False
>>> os.environ.get('SMARTG_EXAMPLE') is None
True

Key Functions
-------------
modified_environ
    Context manager that temporarily updates os.environ in-place.
"""

import contextlib
import os
from collections.abc import Iterator
from typing import Any


@contextlib.contextmanager
def modified_environ(*remove: str, **update: Any) -> Iterator[None]:
    """Temporarily update ``os.environ`` in-place.

    The environment is modified in-place so that the change is visible
    to all subprocesses and C extensions that consult ``os.environ``
    directly. The original state is restored when the context block
    exits, even if an exception is raised inside it.

    Parameters
    ----------
    *remove : str
        Names of environment variables to remove for the duration of
        the context. Variables that did not exist before are simply
        ignored.
    **update : str
        Environment variables to set or override, given as keyword
        arguments (e.g. ``LD_LIBRARY_PATH='/my/path'``). Values are
        converted to strings by :meth:`os.environ.update`.

    Yields
    ------
    None
        No value is yielded; the context only provides the side effect
        of a modified environment.

    Notes
    -----
    Variables that are both updated and already present are saved and
    restored to their original value on exit. Variables introduced by
    ``update`` that were not previously defined are removed on exit.
    Variables listed in ``remove`` are restored on exit if they
    existed before, otherwise they stay absent.

    See Also
    --------
    smartg.smartg.Smartg : uses this context manager to select a CUDA
        device at initialization time.
    """
    env = os.environ
    update = update or {}
    remove = remove or ()

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
