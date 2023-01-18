#!/bin/sh
PYTHONPATH=. gunicorn web.app:server --workers 4 --timeout 180
