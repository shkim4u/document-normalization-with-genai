# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""General helper utilities the workshop notebooks"""
# Python Built-Ins:
from io import StringIO
import sys
import textwrap
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_ww(*args, width: int = 100, **kwargs):
    """Like print(), but wraps output to `width` characters (default 100)"""
    buffer = StringIO()
    try:
        _stdout = sys.stdout
        sys.stdout = buffer
        logger.info(*args, **kwargs)
        output = buffer.getvalue()
    finally:
        sys.stdout = _stdout
    for line in output.splitlines():
        logger.info("\n".join(textwrap.wrap(line, width=width)))
