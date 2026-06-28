# -*- coding: utf-8 -*-
"""
Legacy compatibility wrapper for the old ``gflownet_main.py`` entrypoint.

The actively maintained streaming pipeline now lives in ``train_streaming.py``.
Keeping this file importable avoids syntax errors in environment-wide checks and
preserves the historical CLI entrypoint by forwarding execution.
"""

from train_streaming import main


if __name__ == "__main__":
    main()
