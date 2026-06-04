#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module facilitates the inclusion of images in pytest-html reports.

Generate images with matplotlib and use conftest.savefig(request)
instead of matplotlib's savefig.
Example:

    import conftest
    from matplotlib import pyplot as plt

    def test_html_report(request):
        plt.figure()
        plt.plot()
        conftest.savefig(request)


How to use it
-------------

- To use this module, please create the file `tests/conftest.py` in your repository,
  with the following content:
    from core.conftest import *

- Run pytest with the following options (can be added to pytest.ini):
    --html=tests/test_report.html --self-contained-html

    Sample pytest.ini:
        [pytest]
        addopts= -n auto --html=test_report.html --self-contained-html

Note: this requires pytest-html


Other features
--------------
    
- If you generate an image as a ByteIO buffer, you can add it
  to the report like so:

    def test_with_image(request):
        plt.plot(...)
        fp = io.BytesIO()
        plt.savefig(fp)
        conftest.add_image_to_report(request, fp)

- If you generate a file in your test, you can link it in the report using:

    conftest.add_link_to_report(request, path)

- The images embedded in the html file can be dragged in a browser to view
  them in full size.

Parameters
----------

By using this conftest, the following parameters are being supported in pytest.ini

img_collapsible
    Whether to use a collapsible section to embed the images (default true).
    Ex: true, false
    Otherwise, the standard image integration is used.
img_size
    Image size, when using img_collapsible mode
    Ex: 100%, 250px (default)
"""

from pathlib import Path
import base64
import io
import pytest


def add_image_to_report(request, fp):
    """
    Appends image data to request.node.images

    request: pytest `request` fixture

    fp: BytesIO
    """
    if not hasattr(request.node, 'images'):
        request.node.images = []
    fp.seek(0)
    data = fp.read()
    request.node.images.insert(0, data)


def add_extra_to_report(request, *args):
    """
    Add extra content to pytest-html report (through request.node.extras)

    Examples:
        add_extra_to_report(request, 'sample text', 'text')
        add_extra_to_report(request, 'sample html', 'html')
        add_extra_to_report(request, 'https://www.wikipedia.org/', 'url', 'Link text')
    """
    if not hasattr(request.node, 'extras'):
        request.node.extras = []
    request.node.extras.insert(0, args)


def add_link_to_report(request, path, name='Link'):
    """
    Add a link to a local file `path`

    Makes the link relative to the directory of html output
    """
    assert Path(path).exists()
    htmlo = request.config.getoption("--html")
    if htmlo is None:
        return
    html_output = Path(htmlo).resolve()
    try:
        url = Path(path).resolve().relative_to(html_output.parent)
    except ValueError:
        raise Exception(f'path ({path}) must be a subfolder of output html ({html_output})')
    add_extra_to_report(request, str(url), 'url', name)


def savefig(request, **kwargs):
    """
    Wraps matplotlib's savefig to add image data to request.node.images

    `kwargs` are passed to `plt.savefig`
    """
    from matplotlib import pyplot as plt

    fp = io.BytesIO()
    plt.savefig(fp, **kwargs)
    add_image_to_report(request, fp)
    plt.close('all')

def pytest_addoption(parser):
    parser.addini('img_collapsible',
                  'Whether to use a collapsible section to embed the images (default true).')
    parser.addini('img_size',
                  'Image size, when using img_collapsible mode. Ex: 100%, 250px (default)')


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item):
    # pytest-html 4.x moved extras to a top-level import; 3.x used the plugin object.
    try:
        from pytest_html import extras as html_extras  # 4.x
    except ImportError:
        html_extras = None

    # 3.x fallback
    pytest_html_plugin = item.config.pluginmanager.getplugin('html')

    outcome = yield
    report = outcome.get_result()
    extra = getattr(report, 'extras', [])
    img_size = item.config.getini('img_size') or '250px'
    img_use_extra = {'true': True, 'false': False}[
        (item.config.getini('img_collapsible') or 'true').lower()
        ]

    has_html_plugin = html_extras is not None or pytest_html_plugin is not None

    if (report.when == 'call') and has_html_plugin:
        def _html(content):
            if html_extras is not None:
                return html_extras.html(content)
            return pytest_html_plugin.extras.html(content)

        def _image(b64data):
            if html_extras is not None:
                return html_extras.image(b64data, mime_type='image/png')
            return pytest_html_plugin.extras.image(b64data)

        # add docstring
        doc = item.function.__doc__
        if doc is not None:
            extra.append(_html(f'<pre>{doc}</pre>'))

        # add images
        # pytest-html 4.x HTML-escapes the content of extras.html(), so
        # embedding <img> tags via html() produces raw text, not images.
        # Always use the native image() API when on 4.x.
        use_native_image = html_extras is not None
        img_content = ''
        for image in getattr(item, 'images', []):
            b64data = base64.b64encode(image).decode('ascii')
            if use_native_image:
                extra.append(_image(b64data))
            elif img_use_extra:
                img_content += (f'<img src="data:image/png;base64,{b64data}"'
                                f' style="max-width:{img_size};">')
            else:
                extra.append(_image(b64data))

        if img_content:
            extra.append(_html(
                f'<details open><summary>Images</summary>{img_content}</details>'
            ))

        # add other extras (added via add_extra_to_report)
        for a in getattr(item, 'extras', []):
            content, fmt = a[0], (a[1] if len(a) > 1 else 'text')
            if html_extras is not None:
                if fmt == 'html':
                    extra.append(html_extras.html(content))
                elif fmt == 'url':
                    extra.append(html_extras.url(content, name=a[2] if len(a) > 2 else 'Link'))
                elif fmt == 'text':
                    extra.append(html_extras.text(content))
                else:
                    extra.append(html_extras.text(str(content)))
            else:
                extra.append(pytest_html_plugin.extras.extra(*a))

        report.extras = extra
