#!/bin/sh
# PYTHONPATH=. streamlit run --server.port 8080 ./web/demo.py
PYTHONPATH=. gunicorn web.app:server --workers 4 --timeout 180
