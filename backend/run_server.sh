#!/bin/sh
PYTHONPATH=. gunicorn backend.server:app --workers 4 --timeout 180 --bind 0.0.0.0:8081
