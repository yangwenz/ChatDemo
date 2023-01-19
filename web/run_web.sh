#!/bin/sh
PYTHONPATH=. gunicorn web.app:server --workers 4 --timeout 180 --bind 0.0.0.0:8080
